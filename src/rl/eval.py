import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from src.models.eval import load_model, get_transforms
from src.rl.env import WasteRLMultiStepEnv
from src.rl.agent import DQNAgent
import torch.nn.functional as F
from PIL import Image

def evaluate_single_image(model, device, image_path, true_label, class2idx, transform=get_transforms()):
    image = Image.open(image_path).convert("RGB")   # читаем изображение в RGB
    image_tensor = transform(image).unsqueeze(0).to(device)  # превращаем в батч из 1 элемента

    with torch.no_grad():
        outputs = model(image_tensor)               # предсказываем класс
        probs = F.softmax(outputs, dim=1)           # превращаем логиты в вероятности
        predicted_idx = torch.argmax(probs, dim=1).item()

    true_idx = class2idx[true_label]
    correct = int(predicted_idx == true_idx)        # 1 если предсказал верно
    confidence = probs[0, predicted_idx].item()     # вероятность предсказанного класса

    return correct, confidence, predicted_idx

def setup_paths():
    root = Path(__file__).parent.parent.parent
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    return root, reports_dir


def load_data(root):
    manifest_path = root / "data" / "unified" / "manifest.csv"
    df = pd.read_csv(manifest_path)
    classes = sorted(df["unified_class"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    return df, classes, class2idx


def load_models(root, df, classes, class2idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_path = root / "src" / "models" / "rl_agent_multistep.pth"
    data_root = root / "data" / "raw"

    baseline_model = load_model(device=device, num_classes=len(classes))
    
    env = WasteRLMultiStepEnv(df, baseline_model, device, class2idx, str(data_root))
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load_state_dict(torch.load(agent_path, map_location=device))
    agent.eval()
    
    return baseline_model, env, agent, device


def run_evaluation(env, agent, baseline_model, device, class2idx, num_samples=50):
    baseline_correct = 0
    rl_correct = 0
    all_actions = []

    for i in range(num_samples):
        state = env.reset()
        img_path = env.current_image_path
        true_label = env.true_label

        base_correct, _, _ = evaluate_single_image(
            baseline_model, device, img_path, true_label, class2idx
        )

        done = False
        step_count = 0
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            all_actions.append(env.actions[action])
            next_state, _, done, info = env.step(action)
            state = next_state
            step_count += 1

        rl_correct_now, _, _ = evaluate_single_image(
            baseline_model, device, env.current_image_path, true_label, class2idx
        )

        baseline_correct += base_correct
        rl_correct += rl_correct_now

        print(f"[{i+1:03d}] Steps={step_count} | Baseline={base_correct} | RL={rl_correct_now}")

    return baseline_correct, rl_correct, num_samples, all_actions


def save_results(baseline_acc, rl_acc, improvement, total, all_actions, reports_dir):
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    print(f"RL Accuracy:       {rl_acc:.4f}")
    print(f"Improvement:       {improvement:+.4f}")

    with open(reports_dir / "rl_vs_baseline.txt", "w") as f:
        f.write(f"Baseline Accuracy: {baseline_acc:.4f}\n")
        f.write(f"RL Accuracy: {rl_acc:.4f}\n")
        f.write(f"Improvement: {improvement:+.4f}\n")
        f.write(f"Total Samples: {total}\n")

    action_counts = Counter(all_actions)
    plt.figure(figsize=(10, 5))
    plt.bar(action_counts.keys(), action_counts.values())
    plt.xticks(rotation=45)
    plt.title("RL Agent Actions Distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "action_distribution.png", dpi=300)
    plt.close()

    print(f"Results saved to {reports_dir}")


def evaluate_rl_vs_baseline():
    root, reports_dir = setup_paths()
    df, classes, class2idx = load_data(root)
    baseline_model, env, agent, device = load_models(root, df, classes, class2idx)
    
    baseline_correct, rl_correct, total, all_actions = run_evaluation(
        env, agent, baseline_model, device, class2idx
    )
    
    baseline_acc = baseline_correct / total
    rl_acc = rl_correct / total
    improvement = rl_acc - baseline_acc
    
    save_results(baseline_acc, rl_acc, improvement, total, all_actions, reports_dir)


if __name__ == "__main__":
    evaluate_rl_vs_baseline()
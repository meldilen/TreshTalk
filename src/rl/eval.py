# rl/eval.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from ..models.eval import load_model, get_transforms
from src.rl.env import WasteRLMultiStepEnv
from src.rl.agent import DQNAgent
import torch.nn.functional as F
from PIL import Image

def evaluate_single_image(model, device, image_path, true_label, class2idx, transform=None):
    if transform is None:
        transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)               # [1, num_classes]
        probs = F.softmax(outputs, dim=1)           # [1, num_classes]
        predicted_idx = int(torch.argmax(probs, dim=1).item())

    true_idx = int(class2idx[true_label])
    correct = int(predicted_idx == true_idx)
    confidence = float(probs[0, predicted_idx].item())

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
    agent_path = root / "src" / "rl" / "rl_agent_multistep.pth"
    
    # Проверяем разные возможные пути к данным
    possible_data_paths = [
        root / "data" / "raw",
        root / "data",
        root / "data" / "unified"
    ]
    
    data_root = None
    for path in possible_data_paths:
        if path.exists():
            data_root = path
            break
    
    if data_root is None:
        raise FileNotFoundError(f"Data directory not found. Checked: {[str(p) for p in possible_data_paths]}")
    
    print(f"Using data root: {data_root}")  # для отладки

    baseline_model = load_model(device=device, num_classes=len(classes))
    
    env = WasteRLMultiStepEnv(df, baseline_model, device, class2idx, data_root=str(data_root))
    # action_dim fallback if env.action_space has no attribute n (type-checkers)
    action_dim = int(getattr(env.action_space, "n", len(env.actions)))
    agent = DQNAgent(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=action_dim,
        device=str(device)
    )
    # загружаем если есть
    if (agent_path).exists():
        agent.load(str(agent_path), map_location=device)
    agent.eval()
    
    return baseline_model, env, agent, device

def run_evaluation(env, agent, baseline_model, device, class2idx, num_samples=50):
    baseline_correct = 0
    rl_correct = 0
    all_actions = []

    for i in range(num_samples):
        state, info = env.reset()
        
        img_path = env.original_image_path
        true_label = env.true_label

        base_correct, _, _ = evaluate_single_image(
            baseline_model, device, img_path, true_label, class2idx
        )

        done = False
        step_count = 0

        # Эпизод: агент действует без эпсилона (детерминированно)
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            all_actions.append(env.actions[action])
            next_state, reward, terminated, truncated, step_info = env.step(action)
            state = next_state
            step_count += 1
            done = bool(terminated or truncated)

        # Используем текущий путь после всех преобразований
        rl_correct_now, _, _ = evaluate_single_image(
            baseline_model, device, env.current_image_path, true_label, class2idx
        )

        baseline_correct += base_correct
        rl_correct += rl_correct_now

        print(f"[{i+1:03d}] Steps={step_count} | Baseline={base_correct} | RL={rl_correct_now} | Image={Path(img_path).name}")

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
    keys = list(action_counts.keys())
    vals = list(action_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(keys, vals)
    plt.xticks(rotation=45)
    plt.title("RL Agent Actions Distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "action_distribution.png", dpi=300)
    plt.close()

    print(f"Results saved to {reports_dir}")

def evaluate_rl_vs_baseline(num_samples=50):
    root, reports_dir = setup_paths()
    df, classes, class2idx = load_data(root)
    baseline_model, env, agent, device = load_models(root, df, classes, class2idx)
    
    baseline_correct, rl_correct, total, all_actions = run_evaluation(
        env, agent, baseline_model, device, class2idx, num_samples=num_samples
    )
    
    baseline_acc = baseline_correct / total
    rl_acc = rl_correct / total
    improvement = rl_acc - baseline_acc
    
    save_results(baseline_acc, rl_acc, improvement, total, all_actions, reports_dir)


if __name__ == "__main__":
    evaluate_rl_vs_baseline(num_samples=1000)

import pandas as pd
import torch
from src.models.eval import load_model
from src.rl.env import WasteRLMultiStepEnv
from src.rl.agent import DQNAgent
import os

def train_multistep_rl(num_episodes: int = 300, max_steps: int = 4, model_save_path: str = "src/rl/rl_agent_multistep.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("data/unified/manifest.csv")
    classes = sorted(df["unified_class"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}

    # Загружаем классификатор (frozen)
    model = load_model(device=device, num_classes=len(classes))
    model.eval()

    # === Среда ===
    env = WasteRLMultiStepEnv(df, model, device, class2idx, data_root="data/raw", max_steps=max_steps)

    # === Агент DQN ===
    agent = DQNAgent(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(getattr(env.action_space, "n", len(env.actions))),
        device=str(device)
    )

    last_info = {}
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        last_info = info

        while not done:
            action = agent.select_action(state, epsilon=0.1)  # eps-greedy
            next_state, reward, terminated, truncated, step_info = env.step(action)
            agent.update(state, action, reward, next_state, bool(terminated))
            state = next_state
            total_reward += reward
            done = bool(terminated or truncated)
            last_info = step_info

        if episode % 20 == 0 or episode == num_episodes - 1:
            print(f"Ep {episode:04d} | total_reward={total_reward:.4f} | last_action={last_info.get('action')} | steps={last_info.get('step')}")
            # Можно сохранять промежуточные чекпоинты
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            agent.save(model_save_path)

    # финальное сохранение
    agent.save(model_save_path)
    print(f"Saved agent to {model_save_path}")


if __name__ == "__main__":
    train_multistep_rl(num_episodes=300, max_steps=4)

import pandas as pd
import torch
from src.models.eval import load_model
from src.rl.env import WasteRLMultiStepEnv
from src.rl.agent import DQNAgent

def train_multistep_rl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("data/unified/manifest.csv")
    classes = sorted(df["unified_class"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}

    model = load_model(device=device, num_classes=len(classes))

    # === Среда ===
    env = WasteRLMultiStepEnv(df, model, device, class2idx, data_root="data/raw", max_steps=4)

    # === Агент DQN ===
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    # === Обучение ===
    for episode in range(300):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 20 == 0:
            print(f"Ep {episode:03d} | total_reward={total_reward:.4f} | last_action={info['action']} | steps={info['step']}")

    torch.save(agent.state_dict(), "src/models/rl_agent_multistep.pth")


if __name__ == "__main__":
    train_multistep_rl()

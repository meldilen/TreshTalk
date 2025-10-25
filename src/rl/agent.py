import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Optional

class DQNAgent(nn.Module):
    """
    Простой DQN-агент (online, без replay-buffer) для небольшого дискретного action-space.
    - state_dim: размер вектора состояния (int)
    - action_dim: количество дискретных действий (int)
    - lr: learning rate
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, device: Optional[str] = None):
        super().__init__()
        assert isinstance(state_dim, int) and isinstance(action_dim, int)

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Простая Q-network: state_dim -> hidden -> action_dim
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 0.9  # дисконт-фактор
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_net(x)

    def select_action(self, state, epsilon: float = 0.1) -> int:
        """
        Эпсилон-жадный выбор действия.
        state: numpy array или torch tensor (1D)
        """
        if random.random() < epsilon:
            return int(random.randrange(self.action_dim))  # случайное действие

        # детерминированный выбор по сети
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state_t = state.float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_t)  # shape [1, action_dim]
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Один шаг gradient descent для DQN:
        loss = (Q(state,action) - target)^2
        target = r + gamma * max_a' Q(next_state, a') * (1 - done)
        """
        # Приводим к тензорам и устройству
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)      # [1, state_dim]
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)              # [1]
        done_mask = torch.tensor([float(done)], dtype=torch.float32, device=self.device)        # [1]

        # Текущие Q
        q_vals = self.q_net(state_t)              # [1, action_dim]
        q_val = q_vals[0, action]                 # scalar

        # Целевой Q (target)
        with torch.no_grad():
            next_q_vals = self.q_net(next_state_t)   # [1, action_dim]
            next_q_max = torch.max(next_q_vals, dim=1)[0]  # [1]
            target = reward_t + self.gamma * next_q_max * (1.0 - done_mask)

        loss = self.loss_fn(q_val, target.detach())  # scalar

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str):
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(ckpt["q_net_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                # optimizer shape mismatch — игнорируем
                pass

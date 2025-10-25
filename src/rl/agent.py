import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 0.9

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.q_net[-1].out_features - 1)
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state).float().unsqueeze(0))
            return int(torch.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state).float().unsqueeze(0)
        next_state = torch.tensor(next_state).float().unsqueeze(0)
        reward = torch.tensor([reward]).float()
        q_values = self.q_net(state)
        next_q_values = self.q_net(next_state)

        target = reward + self.gamma * torch.max(next_q_values) * (1 - done)
        loss = (q_values[0, action] - target.detach()) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

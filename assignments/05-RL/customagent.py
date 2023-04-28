import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List


class QNet(nn.Module):
    "docstring"

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(QNet, self).__init__()
        dims = [state_dim] + hidden_dims + [action_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x


class Agent:
    "docstring"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [32,32],
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9,
        min_epsilon: float = 0.01,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.q_net = QNet(state_dim, action_dim, hidden_dims).to(device)
        self.q_target_net = QNet(state_dim, action_dim, hidden_dims).to(device)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device

    def act(self, state: gym.spaces.Box) -> int:
        "docstring"
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        action = q_values.argmax().item()
        return action

    def learn(
        self,
        state: gym.spaces.Box,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        "docstring"
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.q_target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self) -> None:
        "docstring"
        for param, target_param in zip(
            self.q_net.parameters(), self.q_target_net.parameters()
        ):
            target_param.data.copy_(
                self.gamma * target_param.data + (1 - self.gamma) * param.data
            )

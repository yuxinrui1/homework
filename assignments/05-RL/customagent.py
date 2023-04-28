import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define Q network
class QNet(nn.Module):
    "Docstring"

    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define Q-learning agent
class Agent:
    "Docstring"

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNet(state_dim, action_dim).to(self.device)
        self.q_target_net = QNet(state_dim, action_dim).to(self.device)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.q_target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        "Docstring"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def learn(
            self,
            observation: gym.spaces.Box,
            reward: float,
            terminated: bool,
            truncated: bool,
    ) -> None:
        "Docstring"
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
        "Docstring"
        for target_param, param in zip(
            self.q_target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data * 0.001 + target_param.data * 0.999)

import gym
import numpy as np


class Agent:
    "this is the class function"

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        alpha: float = 0.1,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        "this is the agent function"
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        "this is the act function"
        if np.random.uniform() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[observation, :])

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        "this is the learn function"
        action = self.act(observation)
        next_q = np.max(self.q_table[observation, :])
        self.q_table[observation, action] += self.alpha * (
            reward + self.gamma * next_q - self.q_table[observation, action]
        )
        if terminated or truncated:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            self.q_table[observation, action] = reward

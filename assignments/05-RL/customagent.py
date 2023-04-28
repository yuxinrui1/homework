import gymnasium as gym
import numpy as np
import random


class Agent:
    "docstring"

    def __init__(
        self,
        action_space,
        observation_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def act(self, s: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Choose an action to take given the current state.
        Uses an epsilon-greedy policy based on the Q-table.

        Args:
            state: The current state of the environment.

        Returns:
            The index of the action to take.
        """
        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
                a = 2
            elif angle_todo < -0.05:
                a = 3
            elif angle_todo > +0.05:
                a = 1
        return a

    def learn(
        self,
        state: gym.spaces.Box,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update the Q-value table based on the observed experience.

        Args:
            state: The state of the environment at the start of the observed experience.
            action: The action taken during the observed experience.
            reward: The reward received during the observed experience.
            next_state: The state of the environment after the observed experience.
            done: Whether the episode terminated after the observed experience.
        """
        # Calculate the target Q-value for the observed experience
        q_value = self.q_table[state][action]
        max_next_q_value = np.max(self.q_table[next_state])
        target_q_value = reward + self.discount_factor * max_next_q_value * (not done)

        # Update the Q-value table using the Q-learning update rule
        self.q_table[state][action] = q_value + self.learning_rate * (
            target_q_value - q_value
        )

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

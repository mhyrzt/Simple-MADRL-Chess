import gym
import numpy as np
from abc import ABC


class Learning(ABC):
    def __init__(
        self,
        environment: gym.Env,
        episodes: int,
        gamma: float = 0.99,
    ) -> None:
        self.environment = environment
        self.state_dim = ...
        self.action_dim = ...
        
        self.episodes = episodes
        self.gamma = gamma

    def take_action(self, state: np.ndarray):
        pass

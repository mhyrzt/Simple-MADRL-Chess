import gym
import torch as T
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class Learning(nn.Module, ABC):
    def __init__(
        self,
        environment: gym.Env,
        state_dim: int,
        action_dim: int,
        epochs: int,
        gamma: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.environment = environment

        self.gamma = gamma
        self.epochs = epochs

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    @abstractmethod
    def take_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, *args):
        pass

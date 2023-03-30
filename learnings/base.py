import gym
import torch as T
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class Learning(nn.Module, ABC):
    def __init__(
        self,
        environment: gym.Env,
        epochs: int,
        gamma: float,
        learning_rate: float
    ) -> None:
        super().__init__()
        self.state_dim = environment.observation_space.shape[0]
        self.action_dim = environment.action_space.n

        self.gamma = gamma
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    @abstractmethod
    def take_action(self, state: np.ndarray, *args):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, *args):
        pass
    
    @abstractmethod
    def save(self, folder: str):
        pass

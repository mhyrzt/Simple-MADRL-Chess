import gym
import numpy as np
import torch as T
from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic


class PPO(Learning):
    def __init__(
        self,
        environment: gym.Env,
        hidden_layers: tuple[int],
        episodes: int,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(environment, episodes, gamma)
        self.hidden_layers = hidden_layers
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, self.action_dim, hidden_layers)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    
    def take_action(self, state: np.ndarray):
        state = T.Tensor([state]).to(self.device)
        dist  = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = T.squeeze(dist.log_prob(action)).item()
        actions = T.squeeze(action).item()
        value = T.squeeze(value).item()
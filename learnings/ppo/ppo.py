import os
import gym
import numpy as np
import torch as T
import torch.optim as optim

from tqdm import tqdm
from buffer.ppo import BufferPPO
from buffer.episode import Episode

from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic


class PPO(Learning):
    def __init__(
        self,
        environment: gym.Env,
        hidden_layers: tuple[int],
        epochs: int,
        buffer_size: int,
        batch_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        learning_rate: float = 0.003,
    ) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)

        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer = BufferPPO(
            gamma=gamma,
            max_size=buffer_size,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )

        self.hidden_layers = hidden_layers
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.to(self.device)

    def take_action(self, state: np.ndarray, action_mask: np.ndarray):
        state = T.Tensor(state).unsqueeze(0).to(self.device)
        action_mask = T.Tensor(action_mask).unsqueeze(0).to(self.device)
        dist = self.actor(state, action_mask)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(self.critic(state)).item()
        action = T.squeeze(action).item()
        return action, probs, value

    def epoch(self):
        (
            states_arr,
            actions_arr,
            rewards_arr,
            goals_arr,
            old_probs_arr,
            values_arr,
            masks_arr,
            advantages_arr,
            batches,
        ) = self.buffer.sample()

        for batch in batches:
            masks = T.Tensor(masks_arr[batch]).to(self.device)
            values = T.Tensor(values_arr[batch]).to(self.device)
            states = T.Tensor(states_arr[batch]).to(self.device)
            actions = T.Tensor(actions_arr[batch]).to(self.device)
            old_probs = T.Tensor(old_probs_arr[batch]).to(self.device)
            advantages = T.Tensor(advantages_arr[batch]).to(self.device)

            dist = self.actor(states, masks)
            critic_value = T.squeeze(self.critic(states))

            new_probs = dist.log_prob(actions)
            prob_ratio = (new_probs - old_probs).exp()

            weighted_probs = advantages * prob_ratio
            weighted_clipped_probs = (
                T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages
            )

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            critic_loss = ((advantages + values - critic_value) ** 2).mean()
            total_loss = actor_loss + 0.5 * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def learn(self):
        for epoch in tqdm(range(self.epochs), desc="PPO Learning...", ncols=64, leave=False):
            self.epoch()
        self.buffer.clear()

    def remember(self, episode: Episode):
        self.buffer.add(episode)

    def save(self, folder: str, name: str):
        T.save(self, os.path.join(folder, f"{name}.pt"))
        
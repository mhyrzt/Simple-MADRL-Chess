import gym
import numpy as np
import torch as T
import torch.optim as optim
from tqdm.autonotebook import tqdm

from buffer.ppo import BufferPPO
from buffer.episode import Episode

from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic


class PPO(Learning):
    def __init__(
        self,
        environment: gym.Env,
        state_dim: int,
        action_dim: int,
        hidden_layers: tuple[int],
        epochs: int,
        buffer_size: int,
        batch_size: int,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(environment, state_dim, action_dim, epochs, gamma)
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer = BufferPPO(
            gamma=gamma,
            max_size=buffer_size,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )

        self.hidden_layers = hidden_layers
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers, self.device)
        self.critic = Critic(self.state_dim, self.action_dim, hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.to(self.device)

    def take_action(self, state: np.ndarray):
        state = T.Tensor([state]).to(self.device)
        value = T.squeeze(self.critic(state)).item()
        prob, action, _ = self.actor(state)
        return action[0], prob[0], value

    def epoch(self):
        (
            states_arr,
            actions_arr,
            rewards_arr,
            goals_arr,
            old_probs_arr,
            values_arr,
            advantages_arr,
            batches,
        ) = self.buffer.sample()

        for batch in batches:
            values = T.Tensor(values_arr[batch]).to(self.device)
            states = T.Tensor(states_arr[batch]).to(self.device)
            old_probs = T.Tensor(old_probs_arr[batch]).to(self.device)
            advantages = T.Tensor(advantages_arr[batch]).to(self.device)

            _, _, calc_log_probs = self.actor(states)
            critic_value = T.squeeze(self.critic(states))

            new_probs = calc_log_probs(actions_arr[batch])
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
        for epoch in tqdm(range(self.epochs), desc="PPO Learning..."):
            self.epoch()
        self.buffer.clear()

    def remember(self, episode: Episode):
        self.buffer.add(episode)

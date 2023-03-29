import utils
from buffer.base import Buffer
from buffer.episode import Episode
from collections import deque
import numpy as np

class BufferPPO(Buffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        shuffle: bool = True,
    ) -> None:
        super().__init__(max_size, batch_size, shuffle)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episodes = deque(maxlen=max_size)
        self.advantages = deque(maxlen=max_size)

    def add(self, episode: Episode):
        self.episodes.append(episode)
        self.advantages.append(episode.calc_advantage(self.gamma, self.gae_lambda))

    def clear(self) -> None:
        self.episodes.clear()
        self.advantages.clear()

    def get_len(self) -> int:
        return len(self.episodes)

    def sample(self):
        probs = sum(map(lambda x: x.probs, self.episodes), [])
        goals = sum(map(lambda x: x.goals, self.episodes), [])
        masks = sum(map(lambda x: x.masks, self.episodes), [])
        values = sum(map(lambda x: x.values, self.episodes), [])
        states = sum(map(lambda x: x.states, self.episodes), [])
        actions = sum(map(lambda x: x.actions, self.episodes), [])
        rewards = sum(map(lambda x: x.rewards, self.episodes), [])
        advantages = sum(self.advantages, [])

        batches = utils.make_batch_ids(
            n=len(states), batch_size=self.batch_size, shuffle=self.shuffle
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(goals),
            np.array(probs),
            np.array(values),
            np.array(masks),
            np.array(advantages),
            batches,
        )

import os
import numpy as np
import chess.pieces as Pieces
import chess.info_keys as InfoKeys
from buffer.episode import Episode
from learnings.base import Learning
from tqdm import tqdm
from chess import Chess


class Index:
    STATE = 0
    ACTION = 1
    PROB = 2
    VALUE = 3
    NEXT_STATE = 4
    REWARDS = 5
    DONE = 6
    INFO = 7


class SingleAgentChess:
    def __init__(
        self, env: Chess, learner: Learning, episodes: int, train_on: int
    ) -> None:
        self.env = env
        self.learner = learner
        self.episodes = episodes
        self.train_on = train_on
        self.current_ep = 0

        self.mates_win = np.zeros((2, episodes), dtype=np.int64)
        self.checks_win = np.zeros((2, episodes), dtype=np.int64)
        self.mates_lose = np.zeros((2, episodes), dtype=np.int64)
        self.checks_lose = np.zeros((2, episodes), dtype=np.int64)
        self.rewards = np.zeros((2, episodes))

    def update_stats(self, infos: list[dict]):
        for turn, info in enumerate(infos):
            if InfoKeys.CHECK_MATE_WIN in info:
                self.mates_win[turn, self.current_ep] += 1

            if InfoKeys.CHECK_MATE_LOSE in info:
                self.mates_lose[turn, self.current_ep] += 1

            if InfoKeys.CHECK_WIN in info:
                self.checks_win[turn, self.current_ep] += 1

            if InfoKeys.CHECK_LOSE in info:
                self.checks_lose[turn, self.current_ep] += 1

    def take_action(self, turn: int, episode: Episode):
        state = self.env.get_state(turn)
        _, _, mask = self.env.get_all_actions(turn)

        action, prob, value = self.learner.take_action(state, mask)
        rewards, done, infos = self.env.step(action)

        self.update_stats(infos)
        goal = InfoKeys.CHECK_MATE_WIN in infos[turn]
        episode.add(state, rewards[turn], action, goal, prob, value, mask)

        return [state, rewards, action, goal, prob, value, mask]

    def train_episode(self, render: bool):
        renders = []

        def render_fn():
            if render:
                renders.append(self.env.render())

        self.env.reset()
        episode_white = Episode()
        episode_black = Episode()
        white_data: list = None
        black_data: list = None
        render_fn()
        while not self.env.is_game_done():
            try:
                white_data = self.take_action(Pieces.WHITE, episode_white)
                render_fn()

                if black_data:
                    black_copy = black_data.copy()
                    black_copy[1] = white_data[1][Pieces.BLACK]
                    episode_black.add(*black_copy)

                black_data = self.take_action(Pieces.BLACK, episode_black)
                render_fn()

                white_copy = white_data.copy()
                white_copy[1] = black_data[1][Pieces.WHITE]
                episode_white.add(*white_copy)

            except Exception as e:

                done = True

        self.learner.remember(episode_white)
        self.learner.remember(episode_black)
        if render:
            np.save(f"results/renders/episode_{self.current_ep}.npy", np.array(renders))
        return sum(episode_white.rewards), sum(episode_black.rewards)

    def log(self, episode: int):
        return "\n".join(
            [
                f"+ Results:",
                f"\t- Reward = {self.rewards[:, episode]}",
                f"\t- Checks (win, lose) = {self.checks_win[:, episode]} - {self.checks_lose[:, episode]}",
                f"\t- Mates (win, lose) = {self.mates_win[:, episode]} - {self.mates_lose[:, episode]}",
            ]
        )

    def train(self):
        for ep in range(self.episodes):
            print("-" * 64)
            print(f"Episode: {ep + 1}")
            white_reward, black_reward = self.train_episode(
                ep % 100 == 0 or ep == self.episodes - 1
            )
            self.rewards[0, ep] = black_reward
            self.rewards[1, ep] = white_reward
            self.current_ep += 1
            print(self.log(ep))
            if (ep + 1) % self.train_on == 0:
                self.learner.learn()

    def save(self, folder: str):
        np.save(os.path.join(folder, "rewards.npy"), self.rewards)
        np.save(os.path.join(folder, "mates_win.npy"), self.mates_win)
        np.save(os.path.join(folder, "mates_lose.npy"), self.mates_lose)
        np.save(os.path.join(folder, "checks_win.npy"), self.checks_win)
        np.save(os.path.join(folder, "checks_lose.npy"), self.checks_lose)

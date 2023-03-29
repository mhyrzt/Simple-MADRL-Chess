import os
import numpy as np
import chess.pieces as Pieces
import chess.info_keys as InfoKeys
from buffer.episode import Episode
from learnings.base import Learning
from tqdm import tqdm
from chess import Chess
import torch as T
from utils import save_to_video
import traceback


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

        self.moves = np.zeros((2, episodes), dtype=np.int64)
        self.rewards = np.zeros((2, episodes))
        self.mates_win = np.zeros((2, episodes), dtype=np.int64)
        self.checks_win = np.zeros((2, episodes), dtype=np.int64)
        self.mates_lose = np.zeros((2, episodes), dtype=np.int64)
        self.checks_lose = np.zeros((2, episodes), dtype=np.int64)

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
        self.moves[turn, self.current_ep] += 1

        self.update_stats(infos)
        goal = InfoKeys.CHECK_MATE_WIN in infos[turn]
        episode.add(state, rewards[turn], action, goal, prob, value, mask)
        return done, [state, rewards, action, goal, prob, value, mask]

    def update_enemy(self, prev: list, episode: Episode, reward: int):
        if prev is None:
            return
        prev[1] = reward
        episode.add(*prev)

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
        while True:
            done, white_data = self.take_action(Pieces.WHITE, episode_white)
            self.update_enemy(black_data, episode_black, white_data[1][Pieces.BLACK])
            render_fn()
            if done:
                break

            done, black_data = self.take_action(Pieces.BLACK, episode_black)
            self.update_enemy(white_data, episode_white, black_data[1][Pieces.WHITE])
            render_fn()
            if done:
                break

        self.learner.remember(episode_white)
        self.learner.remember(episode_black)
        if render:
            path = f"results/renders/episode_{self.current_ep}.mp4"
            save_to_video(path, np.array(renders))
            print(f"*** EPISODE SAVED TO: {path} ***")

        return sum(episode_white.rewards), sum(episode_black.rewards)

    def log(self, episode: int):
        return "\n".join(
            [
                f"+ Episode {episode} Results [B | w]:",
                f"\t- Moves  = {self.moves[:, episode]}",
                f"\t- Reward = {self.rewards[:, episode]}",
                f"\t- Checks = {self.checks_win[:, episode]}",
                f"\t- Mates  = {self.mates_win[:, episode]}",
                "-" * 64,
            ]
        )

    def train(self, save_each: int = 10):
        for ep in range(self.episodes):
            white_reward, black_reward = self.train_episode(
                ep % save_each == 0 or ep == self.episodes - 1
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
        T.save(self.learner, os.path.join(folder, "learner.pt"))

import gym
import numpy as np
from chess.pieces import Pieces
from chess.info_keys import InfoKeys
from buffer.episode import Episode
from learnings.base import Learning
from time import sleep

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
    def __init__(self, env: gym.Env, learner: Learning, episodes: int, train_on: int) -> None:
        self.env = env
        self.learner = learner
        self.episodes = episodes
        self.train_on = train_on
        self.current_ep = 0

        self.mates = np.zeros(episodes)
        self.checks = np.zeros(episodes)
        self.rewards = np.zeros(episodes)
        self.wrong_move = np.zeros(episodes)
        self.empty_select = np.zeros(episodes)

    def update_stats(self, infos: list[dict]):
        i = self.current_ep
        for d in infos:
            if InfoKeys.WRONG_MOVE in d:
                self.wrong_move[i] += 1
            if InfoKeys.EMPTY_SELECT in d:
                self.empty_select[i] += 1
            if InfoKeys.CHECK_MATE_WIN in d:
                self.mates[i] += 1
            if InfoKeys.CHECK_WIN in d:
                self.checks[i] += 1

    def take_action(self, turn: int, episode: Episode):
        _turn = self.env.turn
        state = self.env.get_state(turn)

        action, prob, value = self.learner.take_action(state)
        rewards, done, infos = self.env.step(action)
        self.update_stats(infos)
        if _turn == self.env.turn:
            episode.add(state, rewards[turn], action, False, prob, value)
            return self.take_action(turn, episode)

        goal = InfoKeys.CHECK_MATE_WIN in infos[turn]
        episode.add(state, rewards[turn], action, goal, prob, value)
        return done

    def train_episode(self):
        self.env.reset()
        sleep(1)
        done = False
        total_reward = 0
        episode_white = Episode()
        episode_black = Episode()
        while not done:
            done = self.take_action(Pieces.WHITE, episode_white)
            done = self.take_action(Pieces.BLACK, episode_black)
        self.learner.remember(episode_white)
        self.learner.remember(episode_black)
        total_reward = sum(episode_white.rewards) + sum(episode_black.rewards)
        return total_reward

    def log(self):
        i = self.current_ep
        print(
            f"#{i} | Reward = {self.rewards[i]} | wrong = {self.wrong_move[i]} | empty = {self.empty_select[i]} | check = {self.checks[i]} | mate = {self.mates[i]}"
        )

    def train(self):
        for ep in range(self.episodes):
            self.rewards[ep] = self.train_episode()
            self.log()
            self.current_ep += 1
            if ep > 0 and ep % self.train_on == 0:
                self.learner.learn()

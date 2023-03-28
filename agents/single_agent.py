import gym
import numpy as np
from chess.pieces import Pieces
from chess.info_keys import InfoKeys
from buffer.episode import Episode
from learnings.base import Learning
from tqdm import tqdm


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
        self, env: gym.Env, learner: Learning, episodes: int, train_on: int
    ) -> None:
        self.env = env
        self.learner = learner
        self.episodes = episodes
        self.train_on = train_on
        self.current_ep = 0

        self.mates = np.zeros(episodes, dtype=np.int64)
        self.checks = np.zeros(episodes, dtype=np.int64)
        self.rewards = np.zeros(episodes)
        self.wrong_move = np.zeros(episodes, dtype=np.int64)
        self.empty_select = np.zeros(episodes, dtype=np.int64)

    def update_stats(self, infos: list[dict]):
        for info in infos:
            if InfoKeys.WRONG_MOVE in info:
                self.wrong_move[self.current_ep] += 1

            if InfoKeys.EMPTY_SELECT in info:
                self.empty_select[self.current_ep] += 1

            if InfoKeys.CHECK_MATE_WIN in info:
                self.env.render()
                self.mates[self.current_ep] += 1

            if InfoKeys.CHECK_WIN in info:
                self.checks[self.current_ep] += 1

    def take_action(self, turn: int, episode: Episode):
        state = self.env.get_state(turn)
        counter = 0
        cond = turn == self.env.turn
        while cond:
            counter += 1
            if counter % 1024 == 0:
                print(f"action try counter = {counter}")
            action, prob, value = self.learner.take_action(state)
            rewards, done, infos = self.env.step(action)
            self.update_stats(infos)
            goal = InfoKeys.CHECK_MATE_WIN in infos[turn]
            if turn != self.env.turn:
                episode.add(state, rewards[turn], action, goal, prob, value)
                cond = False
        print(f"took action after {counter} attempts")
        self.env.render()
        
        return done

    def train_episode(self):
        self.env.reset()
        self.env.render()
        done = False
        action_count = 0
        episode_white = Episode()
        episode_black = Episode()
        while not done:
            print(f"Action = {action_count}")
            done = self.take_action(Pieces.WHITE, episode_white)
            action_count += 1
            
            print(f"Action = {action_count}")
            done = self.take_action(Pieces.BLACK, episode_black)
            action_count += 1


        self.learner.remember(episode_white)
        self.learner.remember(episode_black)
        total_reward = sum(episode_white.rewards) + sum(episode_black.rewards)
        return total_reward

    def log(self, episode: int):
        return "\n".join([
            f"+ Results:",
            f"\t- Reward = {self.rewards[episode]}",
            f"\t- Wrong Moves = {self.wrong_move[episode]}",
            f"\t- Empty Selects = {self.empty_select[episode]}",
            f"\t- Checks = {self.checks[episode]}",
            f"\t- Mates = {self.mates[episode]}"
        ])

    def train(self):
        for ep in range(self.episodes):
            print("-" * 64)
            print(f"Episode: {ep + 1}")
            self.rewards[ep] = self.train_episode()
            self.current_ep += 1
            print(self.log(ep))
            if ((ep + 1) % self.train_on == 0):
                self.learner.learn()

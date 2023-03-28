import numpy as np
from chess import Chess
from agents import SingleAgentChess
from learnings.ppo import PPO
import sys

sys.setrecursionlimit(10 ** 6)

buffer_size = 10
if __name__ == "__main__":
    chess = Chess(window_size=512)
    ppo = PPO(
        chess,
        state_dim=8 * 8 * 2,
        action_dim=8 * 2 * 2,
        hidden_layers=(1024, 1024, 1024, 1024),
        epochs=100,
        buffer_size=buffer_size,
        batch_size=64,
    )

    agent = SingleAgentChess(env=chess, learner=ppo, episodes=100, train_on=1)
    print(ppo.device)
    agent.train()


    def save(x: SingleAgentChess):
        np.save("results/mates.npy", x.mates)
        np.save("results/checks.npy", x.checks)
        np.save("results/rewards.npy", x.rewards)
        np.save("results/wrong_move.npy", x.wrong_move)
        np.save("results/empty_select.npy", x.empty_select)


    save(agent)
    chess.close()

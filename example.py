import numpy as np
from chess import Chess
from agents import SingleAgentChess
from learnings.ppo import PPO

buffer_size = 16
if __name__ == "__main__":
    chess = Chess(window_size=512, max_steps=90, render_mode="rgb_array")
    chess.reset()
    ppo = PPO(
        chess,
        hidden_layers=(1024, 1024, 1024, 1024),
        epochs=50,
        buffer_size=buffer_size,
        batch_size=512,
    )
    print(ppo.device)
    print(ppo)

    agent = SingleAgentChess(
        env=chess, learner=ppo, episodes=1_500, train_on=buffer_size
    )
    agent.train()
    agent.save("results")

    chess.close()

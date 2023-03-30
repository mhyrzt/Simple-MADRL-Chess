import numpy as np
from chess import Chess
from agents import SingleAgentChess, DoubleAgentsChess
from learnings.ppo import PPO

buffer_size = 16
if __name__ == "__main__":
    chess = Chess(window_size=512, max_steps=128, render_mode="rgb_array")
    chess.reset()

    ppo = PPO(
        chess,
        hidden_layers=(2048,) * 4,
        epochs=100,
        buffer_size=buffer_size,
        batch_size=256,
    )

    print(ppo.device)
    print(ppo)
    print("-" * 64)

    agent = SingleAgentChess(
        env=chess,
        learner=ppo,
        episodes=40,
        train_on=buffer_size,
        result_folder="results",
    )
    agent.train(
        render_each=10, 
        save_on_learn=True
    )
    agent.save()
    chess.close()

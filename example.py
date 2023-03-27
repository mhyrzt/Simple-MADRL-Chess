from chess import Chess
from agents import SingleAgentChess
from learnings.ppo import PPO
import sys
sys.setrecursionlimit(4096 * 2)

buffer_size = 10
chess = Chess(window_size=512)
ppo = PPO(
    chess,
    state_dim=8 * 8 * 2,
    action_dim=8 * 2 * 2,
    hidden_layers=(256, 256, 256),
    epochs=1000,
    buffer_size=buffer_size,
    batch_size=64,
)
agent = SingleAgentChess(chess, ppo, 500, buffer_size)
agent.train()
chess.close()
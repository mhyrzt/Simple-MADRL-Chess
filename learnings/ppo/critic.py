import torch as T
import torch.nn as nn
from utils import build_base_model


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.model = build_base_model(state_dim, hidden_layers, 1)

    def forward(self, state: T.Tensor):
        return self.model(state)

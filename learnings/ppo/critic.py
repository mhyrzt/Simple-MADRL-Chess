import torch
import torch.nn as nn
import utils


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.model = utils.build_model(state_dim, hidden_layers, 1)

    def forward(self, state: torch.Tensor):
        return self.model(state)

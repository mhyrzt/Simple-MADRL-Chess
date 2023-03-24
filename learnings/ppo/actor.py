import utils
import torch as T
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_layers: tuple[int]
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.model = utils.build_model(state_dim, hidden_layers, action_dim)

    def forward(self, state: T.Tensor):
        return Categorical(self.model(state))

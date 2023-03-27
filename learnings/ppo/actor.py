import numpy as np
import torch as T
import torch.nn as nn
from utils import build_base_model, tensor_to_numpy
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_layers: tuple[int], device: str
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.base_model = build_base_model(state_dim, hidden_layers)

        self.select_row = nn.Sequential(
            nn.Linear(hidden_layers[-1], action_dim // 4), nn.Softmax(dim=1)
        )
        self.select_col = nn.Sequential(
            nn.Linear(hidden_layers[-1], action_dim // 4), nn.Softmax(dim=1)
        )
        self.moveto_row = nn.Sequential(
            nn.Linear(hidden_layers[-1], action_dim // 4), nn.Softmax(dim=1)
        )
        self.moveto_col = nn.Sequential(
            nn.Linear(hidden_layers[-1], action_dim // 4), nn.Softmax(dim=1)
        )

        self.to(device)

    def convert_to_cell(self, rows: T.Tensor, cols: T.Tensor) -> np.ndarray:
        rows = rows.unsqueeze(1)
        cols = cols.unsqueeze(1)
        cell = T.cat([rows, cols], dim=1)
        return tensor_to_numpy(cell)

    def convert_to_action(
        self, select_cell: np.ndarray, moveto_cell: np.ndarray
    ) -> np.ndarray:
        action = np.concatenate((select_cell, moveto_cell), axis=1).reshape(-1, 2, 2)
        return action

    def forward(self, states: T.Tensor):
        base_model = self.base_model(states)
        select_row_dist = Categorical(self.select_row(base_model))
        select_col_dist = Categorical(self.select_col(base_model))
        moveto_row_dist = Categorical(self.moveto_row(base_model))
        moveto_col_dist = Categorical(self.moveto_col(base_model))

        select_row_act = select_row_dist.sample()
        select_col_act = select_col_dist.sample()
        moveto_row_act = moveto_row_dist.sample()
        moveto_col_act = moveto_col_dist.sample()

        select_cell = self.convert_to_cell(select_row_act, select_col_act)
        moveto_cell = self.convert_to_cell(moveto_row_act, moveto_col_act)
        action = self.convert_to_action(select_cell, moveto_cell)

        log_probs_sum = tensor_to_numpy(
            select_row_dist.log_prob(select_row_act)
            + select_col_dist.log_prob(select_col_act)
            + moveto_row_dist.log_prob(moveto_row_act)
            + moveto_col_dist.log_prob(moveto_col_act)
        )

        def calc_log_probs(actions: np.ndarray) -> np.ndarray:
            sr = T.Tensor(actions[:, 0, 0]).to(self.device)
            sc = T.Tensor(actions[:, 0, 1]).to(self.device)
            mr = T.Tensor(actions[:, 1, 0]).to(self.device)
            mc = T.Tensor(actions[:, 1, 1]).to(self.device)
            return (
                select_row_dist.log_prob(sr)
                + select_col_dist.log_prob(sc)
                + moveto_row_dist.log_prob(mr)
                + moveto_col_dist.log_prob(mc)
            )

        return log_probs_sum, action, calc_log_probs

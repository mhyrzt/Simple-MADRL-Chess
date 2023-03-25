import numpy as np
import torch as T
import torch.nn as nn


def build_model(
    input_size: int, hidden_layers: tuple[int], output_size: int
) -> nn.Module:
    layers = [
        nn.Linear(input_size, hidden_layers[0]),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_layers[0]),
    ]

    for i in range(len(hidden_layers) - 1):
        in_features = hidden_layers[i]
        out_features = hidden_layers[i + 1]
        layers += [
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
        ]

    layers += [nn.Linear(hidden_layers[-1], output_size), nn.Softmax(dim=1)]

    return nn.Sequential(*layers)


def make_batch_ids(n :int, batch_size: int, shuffle: bool = True) -> np.ndarray:
    starts = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    if shuffle:
        np.random.shuffle(indices)
    return [indices[i: i + batch_size] for i in starts]
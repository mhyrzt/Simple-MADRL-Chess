import torch as T
import torch.nn as nn


def get_batch_norm(
    num_features: int, affine: bool = False, track_running_stats: bool = False
):
    return nn.BatchNorm1d(
        num_features,
        affine=affine,
        track_running_stats=track_running_stats,
    )


def build_model(
    input_size: int, hidden_layers: tuple[int], output_size: int
) -> nn.Module:
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, hidden_layers[0]),
        nn.ReLU(),
        get_batch_norm(hidden_layers[0]),
    ]

    for i in range(len(hidden_layers) - 1):
        in_features = hidden_layers[i]
        out_features = hidden_layers[i + 1]
        layers += [
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            get_batch_norm(out_features),
        ]

    layers += [nn.Linear(hidden_layers[-1], output_size), nn.Softmax(dim=1)]

    return nn.Sequential(*layers)

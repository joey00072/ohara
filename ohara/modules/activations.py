import torch
import torch.nn.functional as F


def relu_squared(x):
    return torch.relu(x) ** 2


ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu_squared": relu_squared,
    "relu2": relu_squared,
}

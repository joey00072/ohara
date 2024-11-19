from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from ohara.modules.activations import ACT2FN


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        activation_fn: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation_fn = ACT2FN[activation_fn]

        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.up(x)
        x = self.activation_fn(x)
        x = self.down(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        activation_fn: str = "silu",
        bias: bool = False,
        *args,
        **kwargs,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1

        order in which W1,W2,W3 are multiplied is as per llama (for compatiblity)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

        self.activation = ACT2FN[activation_fn]
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        up = self.up(x)
        gate = self.gate(x)
        down = self.down(F.silu(gate) * up)
        return self.dropout(down)


class BiLinear(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.w2(self.w1(x) * self.w3(x)))


class ReGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x)) * self.w3(x)))


# This might not me most efficient implementation of MOE
# but it is easy to understand
# TODO: Write a more efficient implementation


class GEGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        bias: bool = False,
    ):
        """
        GLU Variants Improve Transformer
        https://arxiv.org/abs/2002.05202v1
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.gate(x)) * self.up(x)))


MLP_MAP = {
    "swiglu": SwiGLU,
    "mlp": MLP,
    "glu": GLU,
    "bilinear": BiLinear,
    "reglu": ReGLU,
    "geglu": GEGLU,
}

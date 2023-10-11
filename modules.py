import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass





class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        dropout: float,
        bias: bool = True,
    ):
        """
        This is named Feed Forward but this is acually SwiGlu layer  ## Dont confuse with F.gelu
        SwiGlu: https://arxiv.org/abs/2002.05202v1
        Llama: https://arxiv.org/abs/2302.13971 (used in)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))





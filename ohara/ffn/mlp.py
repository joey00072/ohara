import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim:int=None,
        multiple_of: int = 4,
        dropout: float = None,
        activation_fn:Callable=None,
        bias: bool = True,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation_fn = activation_fn if activation_fn else F.silu

        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.w1(x)
        x = self.activation_fn(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

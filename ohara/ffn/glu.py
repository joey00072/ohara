
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
        activation:Callable=None,
        bias: bool = False,
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

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation  = activation if activation else F.silu 
        self.dropout = nn.Dropout(dropout) if dropout else lambda x:x

    def forward(self, x):
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
        bias: bool = False,
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

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x:x

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class BiLinear(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
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
        self.dropout = nn.Dropout(dropout) if dropout else lambda x:x

    def forward(self, x):
        return self.dropout(self.w2(self.w1(x) * self.w3(x)))


class ReGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
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
        self.dropout = nn.Dropout(dropout) if dropout else lambda x:x

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x)) * self.w3(x)))
    
class GEGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
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
        self.dropout = nn.Dropout(dropout) if dropout else lambda x:x

    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))
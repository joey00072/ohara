import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else 4 * dim

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor):
        u = self.up(x)
        g = self.gate(x)
        h = F.silu(g) * u
        out = self.down(h)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else 4 * dim
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        x = self.up(x)
        x = F.silu(x)
        return self.down(x)


class DeGLU(nn.Module):
    def __init__(self, dim, hidden_dim, mask_blocks=None, mask_size=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else 4 * dim
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

        self.mask_blocks = mask_blocks if mask_blocks is not None else 4
        self.mask_size = mask_size
        if mask_size is None:
            mask_size = 2 * hidden_dim

        self.register_buffer("mask_proj", torch.randn(hidden_dim, self.mask_blocks))
        self.mask_proj: Tensor

    def get_mask(self, x: Tensor) -> Tensor:
        out = x.detach() @ self.mask_proj
        out = F.softmax(out, dim=1)
        out = out / out.max().detach()
        out[out < 1] = 0
        out = torch.repeat_interleave(out, x.shape[-1] // self.mask_blocks, dim=1)
        return out.detach()

    def forward(self, x):
        x = self.up(x)
        x = F.silu(x)
        g = self.get_mask(x)
        return self.down(x * g)


if __name__ == "__main__":
    bsz = 1
    dim = 4
    hdim = 8
    x = torch.randn(bsz, dim)
    m = DeGLU(dim, hdim)
    print(m(x).shape)

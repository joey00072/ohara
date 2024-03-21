import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ohara.modules.pscan import pscan

torch.manual_seed(0)
# def scan(x: Tensor, h: Tensor) -> Tensor:
#     # mamba uses B,L,D,N
#     # we have B,L,D. thats why this squeezy ness
#     # TODO: use mambas offical cuda scan to make gpu go burr
#     return pscan(x.unsqueeze(-1), h.unsqueeze(-1)).squeeze(-1)


class RG_LRU(nn.Module):
    def __init__(self, dim):
        self.input_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.forget_lambda = nn.Parameter(torch.linspace(-4.323, -9, dim))

        # Why this Constant is 8 Paper offer no explaintion
        self.C = 8
        with torch.no_grad():
            self.input_proj.weight.normal_(std=dim**-0.5)
            self.gate_proj.weight.normal_(std=dim**-0.5)

    def forward(self, x):
        input_gate: torch.Tensor = self.input_proj(x)
        recurrence_gate: torch.Tensor = self.gate_proj(x)

        # ℎ𝑡    =  𝛼(𝑟𝑡)ℎ𝑡−1 + 𝛽(𝑟𝑡)𝑥𝑡             ...1
        # xbeta =  𝛽(𝑟𝑡)𝑥𝑡                        ...2
        # rest recurrace will calcuate with scan
        # h(t) = parallel_scan( a(rt), xbeta )   ...3
        alpha = (-self.C * F.softplus(self.forget_lambda) * recurrence_gate.sigmoid()).exp()

        beta = (1 - alpha**2 + 1e-6).sqrt()
        xbeta: Tensor = beta * input_gate.sigmoid() * x

        h = pscan(alpha.mT.contiguous(), xbeta.mT.contiguous()).mT
        return h


class RetNext(nn.Module):
    def __init__(self, dim):
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.num_heads = 4
        self.head_dim = dim // self.num_heads

    def forward(self, x):
        bsz, seq_len, x_dim = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k: Tensor = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        q: Tensor = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        v: Tensor = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        states = k.mT @ v
        out = q @ states
        return out


def retation(
    k: Tensor, q: Tensor, v: Tensor, num_heads: int, gamma: float = 0.9, state: Tensor = None
):
    bsz, x_dim = k.shape
    head_dim = x_dim // num_heads

    k = k.view(bsz, num_heads, head_dim).unsqueeze(-2)
    q = q.view(bsz, num_heads, head_dim).unsqueeze(-2)
    v = v.view(bsz, num_heads, head_dim).unsqueeze(-2)

    new_state = ((k.mT @ v) / head_dim) + gamma * state
    out = q @ new_state

    return out.reshape(bsz, x_dim), new_state


def parallel_retation(k: Tensor, q: Tensor, v: Tensor, num_heads: int, gamma: float = 0.9):
    bsz, seq_len, x_dim = k.shape
    head_dim = x_dim // num_heads

    k = k.view(bsz, seq_len, num_heads, head_dim).unsqueeze(-2)
    q = q.view(bsz, seq_len, num_heads, head_dim).unsqueeze(-2)
    v = v.view(bsz, seq_len, num_heads, head_dim).unsqueeze(-2)

    print(f"{k.shape=} , {v.shape=} {(k.mT@v).shape=}")

    new_state = (k.mT @ v) / head_dim

    gammas = torch.ones_like(new_state) * gamma
    new_state = pscan(gammas, new_state)
    out = q @ new_state

    print(f"{k.shape=} , {new_state.shape=} , {out.shape=}")

    return out.reshape(bsz, seq_len, x_dim).float()


B, T, C = 1, 3, 4
num_heads = 2

x = torch.randn(B, T, C)

k = q = v = x

head_dim = C // num_heads
state = torch.zeros(B, head_dim, head_dim)

outs = []
for i in range(T):
    out, state = retation(k[:, i], q[:, i], v[:, i], num_heads, 0.9, state)
    outs.append(out)

result = torch.stack(outs, dim=1)
print(result)

print("=" * 100)

out = parallel_retation(k, q, v, num_heads, 0.9)
print(out)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pscan import pscan


def scan(x: Tensor, h: Tensor) -> Tensor:
    # mamba uses B,L,D,N
    # we have B,L,D. thats why this squeezy ness
    # TODO: use mambas offical cuda scan to make gpu go burr
    return pscan(x.unsqueeze(-1), h.unsqueeze(-1)).squeeze(-1)


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

        h = scan(alpha.mT.contiguous(), xbeta.mT.contiguous()).mT
        return h


class Hawk(nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=1.5, kernel_size=4):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.conv = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            bias=True,
            kernel_size=kernel_size,
            groups=hidden,
            padding=kernel_size - 1,
        )
        self.linear_rnn = RG_LRU(hidden)
        self.output = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.proj.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        B, T, C = x.shape
        # So linear rnn + conv can gets you close to transformer
        # to ssm hippo theory required :)

        gate, x = self.proj(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT
        h = self.linear_rnn(x)
        x = self.output(F.gelu(gate) * h)
        return x


class Griffin(nn.Module):
    ...
    # TODO
    # also craete new file under modules/hybrid.py  move griffin there

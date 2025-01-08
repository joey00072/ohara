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
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.forget_lambda = nn.Parameter(torch.linspace(-4.323, -9, dim))

        # Why this Constant is 8 Paper offer no explaintion
        self.C = 8
        self.reset_parameters()


    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
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
        # TODO: wirte recurrence for inference
        return h

    def reset_parameters(self, init_std=None):
        init_std = init_std or (self.dim ** (-0.5))
        
        for w in [self.input_proj, self.gate_proj]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


class Hawk(nn.Module):
    def __init__(self, *, dim: int = 1024, expansion_factor: float = 1.5, kernel_size: int = 4):
        super().__init__()
        self.dim = dim
        self.hidden = int(dim * expansion_factor)
        self.proj = nn.Linear(dim, 2 * self.hidden, bias=False)
        self.conv = nn.Conv1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            bias=True,
            kernel_size=kernel_size,
            groups=self.hidden,
            padding=kernel_size - 1,
        )
        self.linear_rnn = RG_LRU(self.hidden)
        self.output = nn.Linear(self.hidden, dim, bias=False)

        self.reset_parameters()



    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # So linear rnn + conv can gets you close to transformer
        # to ssm hippo theory required :)

        gate, x = self.proj(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT
        h = self.linear_rnn(x)
        x = self.output(F.gelu(gate) * h)
        return x

    def reset_parameters(self, init_std=None):
        init_std = init_std or (self.dim ** (-0.5))
        
        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=(self.hidden ** (-0.5)),
            a=-3 * init_std,
            b=3 * init_std,
        )
        
class Griffin(nn.Module):
    ...
    # TODO
    # also craete new file under modules/hybrid.py  move griffin there

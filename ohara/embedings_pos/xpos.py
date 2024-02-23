from __future__ import annotations

import torch
import torch.nn as nn


class XPos(nn.Module):
    def __init__(self, dim, num_heads):
        """
        https://arxiv.org/abs/2212.10554
        """
        super().__init__()

        angle = 1.0 / (10000 ** torch.linspace(0, 1, dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def forward(self, slen, recurrent=False):
        # todo chukwire retation
        if recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


if __name__ == "__main__":
    xpos = XPos(64, 4)
    ((sin, cos), decay) = xpos.forward(8)
    print(decay)

    for i in range(1, 9):
        ((sin, cos), decay) = xpos.forward(8, recurrent=True)
        print(i, decay)

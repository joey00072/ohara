from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        """
        warning I am assuing bias=False
        """

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
            self.scaling = self.lora_alpha / self.rank
            self.reset_parameters()

    def reset_lora_parameters(self, lora_only=False):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # funny story behind math.sqrt(5) I'll write blog later

    def reset_parameters(self, lora_only=False):
        nn.init.kaiming_normal_(self.linear.weight, a=math.sqrt(5))
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self.reset_lora_parameters()

    def merge(self):
        if not self.merged and self.rank > 0:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return pretrained + lora


if __name__ == "__main__":
    model = LoRALinear(2, 2, rank=4)
    print(model.linear.weight)
    model.reset_parameters()
    print(model.linear.weight)

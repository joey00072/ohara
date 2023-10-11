import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        )
        self.merged = False

        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self, lora_only=False):
        if hasattr(self, "lora_A"):
            # math.sqrt(5)?? https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        if not lora_only:
            nn.init.kaiming_normal_(self.linear.weight, a=math.sqrt(5))

    def merge(self):
        if not self.merged and self.r > 0:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return pretrained + lora


l = LoRALinear(2, 2)
print(l.linear.weight)
l.reset_parameters()
print(l.linear.weight)

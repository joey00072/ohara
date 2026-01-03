from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
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
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False
        self.enable_dora = True

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
            self.magnitude = nn.Parameter(torch.ones((out_features, 1)))
            self.scaling = self.lora_alpha / self.rank
            self.reset_parameters()

    def reset_magnitude(self):
        with torch.no_grad():
            weight = self.linear.weight
            weight_norm = torch.norm(weight, p=2, dim=1, keepdim=True)
            self.magnitude.copy_(weight_norm)

    def reset_dora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.reset_magnitude()
        # funny story behind math.sqrt(5) I'll write blog later

    def reset_parameters(self, lora_only=False):
        nn.init.kaiming_normal_(self.linear.weight, a=math.sqrt(5))
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self.reset_dora_parameters()

    def dora_trainable_only(self):
        self.linear.train(False)
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        self.magnitude.requires_grad = True

    def merge(self):
        if not self.merged and self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            weight = self.linear.weight + delta
            weight_norm = torch.norm(weight, p=2, dim=1, keepdim=True)
            weight_norm = torch.clamp(weight_norm, min=1e-6)
            weight = weight * (self.magnitude / weight_norm)
            self.linear.weight.data = weight
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.rank == 0 or self.merged:
            return self.linear(x)

        delta = (self.lora_B @ self.lora_A) * self.scaling
        weight = self.linear.weight + delta
        weight_norm = torch.norm(weight, p=2, dim=1, keepdim=True)
        weight_norm = torch.clamp(weight_norm, min=1e-6)
        scale = self.magnitude / weight_norm

        base = F.linear(x, self.linear.weight, bias=None)
        lora = F.linear(self.lora_dropout(x), delta, bias=None)
        out = (base + lora) * scale.T
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out


def dora_from_linear(linear: nn.Linear, lora_alpha: int = 1, lora_dropout: float = 0.0, rank: int = 16):
    device = linear.weight.device
    dtype = linear.weight.dtype
    dora = DoRALinear(
        linear.in_features,
        linear.out_features,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        rank=rank,
    )
    dora = dora.to(device)
    dora.load_state_dict(linear.state_dict(), strict=False)
    if rank > 0:
        dora.reset_magnitude()
    return dora.to(device).to(dtype)


def replace_with_dora(
    model: nn.Module,
    target_layer: list[str] | None = None,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    rank: int = 16,
):
    if isinstance(model, nn.Linear):
        return dora_from_linear(model, lora_alpha, lora_dropout, rank)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and (target_layer is None or name in target_layer):
                setattr(model, name, dora_from_linear(module, lora_alpha, lora_dropout, rank))
            else:
                replace_with_dora(module, target_layer, lora_alpha, lora_dropout, rank)
    return model


def mark_dora_as_trainable(model: nn.Module, target_layer: list[str] | None = None):
    # freeze hole model
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, DoRALinear):
            module.dora_trainable_only()
    return model


def merge_dora(model: nn.Module, target_layer: list[str] | None = None):
    for module in model.modules():
        if isinstance(module, DoRALinear):
            module.merge()
    return model


if __name__ == "__main__":

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
            self.seq = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        def forward(self, x):
            return self.linear(x)

    model = Network()
    model = replace_with_dora(model)
    print(model)

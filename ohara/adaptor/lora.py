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
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False
        self.enable_lora = True

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

    def lora_trainable_only(self):
        self.linear.train(False)
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    def merge(self):
        if not self.merged and self.rank > 0:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        pretrained = self.linear(x)
        if self.rank == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return pretrained + lora


def lora_from_linear(linear: nn.Linear, lora_alpha: int = 1, lora_dropout: float = 0.0, rank: int = 16):
    device = linear.weight.device
    dtype = linear.weight.dtype
    lora = LoRALinear(
        linear.in_features,
        linear.out_features,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        rank=rank,  # Pass rank 16
    )
    lora = lora.to(device)
    lora.load_state_dict(linear.state_dict(), strict=False)  # Ensure only matching keys are loaded
    return lora.to(device).to(dtype)


def replace_with_lora(
    model: nn.Module,
    target_layer: list[str] | None = None,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    rank: int = 16,  # Pass rank 16
):
    if isinstance(model, nn.Linear):
        return lora_from_linear(model, lora_alpha, lora_dropout, rank)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and (target_layer is None or name in target_layer):
                setattr(model, name, lora_from_linear(module, lora_alpha, lora_dropout, rank))
            else:
                replace_with_lora(module, target_layer, lora_alpha, lora_dropout, rank)
    return model


def mark_lora_as_trainable(model: nn.Module, target_layer: list[str] | None = None):
    # freeze hole model
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_trainable_only()
    return model


def merge_lora(model: nn.Module, target_layer: list[str] | None = None):
    for module in model.modules():
        if isinstance(module, LoRALinear):
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
    model = replace_with_lora(model)
    print(model)

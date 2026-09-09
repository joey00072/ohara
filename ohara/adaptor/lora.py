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

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
            self.scaling = self.lora_alpha / self.rank
            self.reset_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # funny story behind math.sqrt(5) I'll write blog later

    def reset_parameters(self):
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self.reset_lora_parameters()

    def lora_trainable_only(self):
        self.linear.requires_grad_(False)
        if self.rank > 0:
            self.lora_A.requires_grad_(True)
            self.lora_B.requires_grad_(True)

    @torch.no_grad()
    def merge(self):
        if not self.merged and self.rank > 0:
            self.linear.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
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
        rank=rank,
        bias=linear.bias is not None,
    )
    lora = lora.to(device=device, dtype=dtype)
    lora.linear.load_state_dict(linear.state_dict())
    return lora.train(linear.training)


def replace_with_lora(
    model: nn.Module,
    target_layer: list[str] | None = None,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    rank: int = 16,  # Pass rank 16
):
    if isinstance(model, nn.Linear) and target_layer is None:
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
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and (
            target_layer is None or name in target_layer or name.rsplit(".", 1)[-1] in target_layer
        ):
            module.lora_trainable_only()
    return model


def merge_lora(model: nn.Module, target_layer: list[str] | None = None):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and (
            target_layer is None or name in target_layer or name.rsplit(".", 1)[-1] in target_layer
        ):
            module.merge()
    return model

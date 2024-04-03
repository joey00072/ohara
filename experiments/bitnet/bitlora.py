from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ohara.utils import auto_accelerator, random_name, model_summary
from torch import Tensor

from bitnet import BitLinear, activation_quant, weight_quant

### lora


class BitLora(BitLinear):
    def __init__(self, rank=4, lora_alpha=1, *args, **kwargs):
        super(BitLora, self).__init__(*args, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.merged = False

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.rms_norm(x)
        x_quant = activation_quant(x_norm)

        if not self.merged and self.rank > 0:
            lora = self.lora_A @ self.lora_B
            w = w + lora * self.scaling

        w_quant, scale = weight_quant(w)

        output = nn.functional.linear(x_quant, w_quant, self.bias)
        return output * scale

    def merge(self):
        if not self.merged and self.rank > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


def _get_bitlora(linear: nn.Linear, rank=4, lora_alpha=1):
    return BitLora(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        rank=rank,
        lora_alpha=lora_alpha,
    )


def apply_bitlora(
    model: nn.Module,
    target_layers: list[str] | None = None,
    rank=4,
    lora_alpha=1,
):
    if isinstance(model, nn.Linear):
        return _get_bitlora(model, rank=rank, lora_alpha=lora_alpha)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for key, value in model._modules.items():
            if isinstance(value, nn.Linear) and (target_layers is None or key in target_layers):
                model._modules[key] = _get_bitlora(value, rank=rank, lora_alpha=lora_alpha)
            else:
                apply_bitlora(value, target_layers=target_layers, rank=rank, lora_alpha=lora_alpha)

    if isinstance(model, (nn.ModuleList, nn.Sequential)):
        for i, sub_model in enumerate(model):
            if isinstance(sub_model, nn.Linear) and (target_layers is None or i in target_layers):
                model[i] = _get_bitlora(sub_model, rank=rank, lora_alpha=lora_alpha)
            else:
                apply_bitlora(
                    sub_model, target_layers=target_layers, rank=rank, lora_alpha=lora_alpha
                )

    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


if __name__ == "__main__":
    # Create a sample input tensor of shape (n, d)
    # For example, let n = batch size and d = features dimension
    n, d, k = 10, 5, 1024  # n: batch size, d: input features, k: output features
    input_tensor: Tensor = torch.randn(n, d)
    parameterized = True
    # Initialize the BitLinear layer with input features d and output features k
    bit_linear_layer: BitLora = BitLora(d, k, bias=False)
    print("bit_linear_layer: ", model_summary(bit_linear_layer))

    linaer: BitLora = nn.Linear(d, k, bias=False)
    print("linaer: ", model_summary(linaer))

    bilinaer: BitLora = apply_bitlora(linaer)
    print("bilinaer: ", model_summary(bilinaer))

    # Run the sample input through the BitLora layer
    output: Tensor = bit_linear_layer(input_tensor)

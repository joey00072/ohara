from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ohara.utils import auto_accelerator, random_name, model_summary
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        """
        Paper: https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        # self.weight = nn.Parameter(torch.ones(dim))
        # if you uncomment this model size will increase a lot

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output  # * self.weight


@torch.jit.script  # https://colab.research.google.com/drive/1B_-PfHKzSmuwF3TETx_ZMlFSE5PNcr1k?usp=sharing
def activation_quant(x: Tensor) -> Tensor:
    scale: Tensor = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
    y: Tensor = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()


@torch.jit.script
def weight_quant(w: Tensor) -> tuple[Tensor, Tensor]:
    scale: Tensor = 1.0 / w.abs().mean().clamp(min=1e-5)
    quant: Tensor = (w * scale).round().clamp(-1, 1) / scale
    w_quant: Tensor = w + (quant - w).detach()
    scale = abs(w_quant).max().detach()
    w_quant = w_quant / scale
    return w_quant, scale


class BitLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BitLinear, self).__init__(*args, **kwargs)
        self.rms_norm = RMSNorm(self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.rms_norm(x)

        x_quant = activation_quant(x_norm)
        w_quant, scale = weight_quant(w)

        output = nn.functional.linear(x_quant, w_quant)
        return output * scale


def _get_bitlinear(linear: nn.Linear):
    return BitLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )


def apply_bitlinear(
    model: nn.Module,
    target_layers: list[str] | None = None,
):
    if isinstance(model, nn.Linear):
        return _get_bitlinear(model)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for key, value in model._modules.items():
            if isinstance(value, nn.Linear) and (target_layers is None or key in target_layers):
                model._modules[key] = _get_bitlinear(value)
            else:
                apply_bitlinear(value)

    if isinstance(model, (nn.ModuleList, nn.Sequential)) :
        for sub_model in model:
            if isinstance(sub_model, nn.Linear) and (target_layers is None or sub_model in target_layers):
                sub_model = _get_bitlinear(sub_model)
            else:
                apply_bitlinear(sub_model)

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
    bit_linear_layer: BitLinear = BitLinear(d, k, bias=False)
    print("bit_linear_layer: ", model_summary(bit_linear_layer))

    linaer: BitLinear = nn.Linear(d, k, bias=False)
    print("linaer: ", model_summary(linaer))

    bilinaer: BitLinear = apply_bitlinear(linaer)
    print("bilinaer: ", model_summary(bilinaer))

    # Run the sample input through the BitLinear layer
    output: Tensor = bit_linear_layer(input_tensor)

"""Precision-controlled expert routing projections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterLinear(nn.Linear):
    """Compute routing logits in FP32, including under mixed-precision autocast.

    Casting the result of an autocast linear is too late: rounding can change
    the winning expert. Promote the operands before the projection instead.
    Autograd keeps the projection's backward GEMMs in FP32, then casts gradients
    back to each operand's storage dtype. Parameters and checkpoint keys retain
    the ordinary Linear layout.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(
                input.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float(),
            )

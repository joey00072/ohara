import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor

class XLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        w = w / w.square().sum(dim=-1, keepdim=True)
        output = nn.functional.linear(x, w, self.bias)
        return output 

def monkey_patch_model(model, target_layers):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_layers:
            setattr(model, name, XLinear(module.in_features, module.out_features, bias=module.bias is not None))
        else:
            monkey_patch_model(module, target_layers)


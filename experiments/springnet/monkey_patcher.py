import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        activation_fn: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation_fn = nn.SELU()

        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.up(x)
        x = self.activation_fn(x)
        x = self.down(x)
        x = self.dropout(x)
        return x
    
    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(self.up.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.trunc_normal_(self.down.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)


class StackedMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        activation_fn: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [MLP(dim, hidden_dim, multiple_of, dropout, activation_fn, bias) for _ in range(4)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class DMLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.scale = nn.Parameter(torch.ones(1, out_features))

    def forward(self, x):
        w = self.weight

        w = w / w.norm(dim=1, keepdim=True)

        out = F.linear(x, w) * self.scale

        if self.bias is not None:
            out = out + self.bias
        return out


def monkey_patch_layer(
    new_layer: nn.Module,
    model: nn.Module,
    target_layers: list[str],
    ignore_layers: list[str] = ["lm_head"],
    parent_name="",
) -> nn.Module:
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, nn.Linear) and name not in ignore_layers and name in target_layers:
            new_linear = new_layer(module.in_features, module.out_features, module.bias is not None)
            setattr(model, name, new_linear)
        else:
            monkey_patch_layer(new_layer, module, target_layers, ignore_layers, full_name)

    return model


def dm_linear_monkey_patch(*args, **kwargs):
    return monkey_patch_layer(DMLinear, *args, **kwargs)


if __name__ == "__main__":
    model = StackedMLP(64)
    dm_linear_monkey_patch(model, ["up"])
    print(model)

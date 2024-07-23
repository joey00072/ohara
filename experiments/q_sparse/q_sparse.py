import torch
import torch.nn as nn
import torch.nn.functional as F


class QSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, K):
        topk_values, topk_indices = torch.topk(tensor, K, dim=-1)

        masked_tensor = torch.zeros_like(tensor)
        masked_tensor.scatter_(-1, topk_indices, topk_values)

        output = masked_tensor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def q_sparse(tensor, K):
    return QSparse.apply(tensor, K)


class QSparseLinear(nn.Linear):
    def __init__(self, pct, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = int(pct * self.weight.shape[1])
        print(self.in_features, self.out_features, self.K)

    def forward(self, x: torch.Tensor):
        x = QSparse.apply(x, self.K)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return F.linear(x, self.weight, self.bias)


def monkey_patch_model(model: nn.Module, target_layers: list[str], pct: float = 0.7):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_layers:
            setattr(
                model,
                name,
                QSparseLinear(
                    pct, module.in_features, module.out_features, bias=module.bias is not None
                ),
            )
        else:
            monkey_patch_model(module, target_layers, pct)


if __name__ == "__main__":
    # Example usage
    tensor = torch.randn(4, 10, requires_grad=True)
    K = 3
    output = QSparse.apply(tensor, K)
    output.sum().backward()

    print(tensor.grad)

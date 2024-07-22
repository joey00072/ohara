import torch
import torch.nn as nn
import torch.nn.functional as F

class QSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, K):
        topk_values, topk_indices = torch.topk(tensor, K, dim=-1)
        
        masked_tensor = torch.zeros_like(tensor)
        masked_tensor.scatter_(-1, topk_indices, topk_values)
        
        norm = torch.norm(masked_tensor, dim=-1, keepdim=True)
        output = masked_tensor / norm
        
        ctx.save_for_backward(norm)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        norm, = ctx.saved_tensors
        return grad_output/norm, None
    
    
def q_sparse(tensor, K):
    return QSparse.apply(tensor, K)

class QSparseLinear(nn.Linear):
    def __init__(self,K,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
    def forward(self, x):
        x = QSparse.apply(x, self.K)
        return F.linear(x, self.weight, self.bias)
    
    
def monkey_patch_model(model: nn.Module,K:int, target_layers:list[str]):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_layers:
            setattr(
                model,
                name,
                QSparseLinear(K, module.in_features, module.out_features, bias=module.bias is not None),
            )
        else:
            monkey_patch_model(module, K,target_layers)

if __name__=="__main__":
    # Example usage
    tensor = torch.randn(4, 10, requires_grad=True)
    K = 3
    output = QSparse.apply(tensor, K)
    output.sum().backward()

    print(tensor.grad)

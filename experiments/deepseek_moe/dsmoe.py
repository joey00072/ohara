import torch
import torch.nn as nn

from ohara.modules.mlp import MLP_MAP, MLP
from torch import Tensor
import torch.nn.functional as F

from dataclasses import dataclass
from collections import OrderedDict

class FFNType:
    DSMoE = "dsmoe"
    SparseMoE = "sparse_moe"
    Dense = "dense"

@dataclass
class Config(OrderedDict):
    vocab_size: int
    seq_len: int

    d_model: int
    hidden_dim: int

    head_dim: int
    num_heads: int
    num_kv_heads: int = 0
    
    num_experts: int = 4
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1

    expert_update_rate: float = 0.001
    train_experts_biases: bool = True
    
    aux_free_loadbalancing: bool = True
    use_aux_loss: bool = True

    num_layers: int = 4
    dense_layers: int = 1

    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False

    activation: str = "silu"  # "relu", "gelu", "silu" etc
    mlp: str = "GLU"  # MLP or GLU
    
    ffn_type: str = FFNType.DSMoE
    
    use_spda: bool = False

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)   
        
        assert self.aux_free_loadbalancing != self.use_aux_loss, f"aux_free_loadbalancing: {self.aux_free_loadbalancing} and use_aux_loss: {self.use_aux_loss} cannot both be True"

def maximal_violation(expert_indices: Tensor, num_experts: int):
    """
    Maximal violation     
    https://arxiv.org/pdf/2408.15664
    eq 4 page 6
    """
    expert_indices = expert_indices.clone().detach().reshape(-1)
    expert_frequencies = torch.bincount(
        expert_indices, minlength=num_experts
    ) / len(expert_indices)
    mean_load = expert_frequencies.mean()
    max_load = expert_frequencies.max()
    return (max_load - mean_load) / mean_load
    

class DSMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 1,
        expert_update_rate: float = 0.001,
        train_experts_biases: bool = True,
        aux_free_loadbalancing: bool = False,
        use_aux_loss: bool = False,
        activation_fn: str = "silu",
        dropout: float = 0.2,
        bias: bool = False,
        mlp: str = "swiglu",
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_update_rate = expert_update_rate
        self.train_experts_biases = train_experts_biases
        self.aux_free_loadbalancing = aux_free_loadbalancing
        self.use_aux_loss = use_aux_loss
        
        assert self.aux_free_loadbalancing != self.use_aux_loss, "only one of aux_free_loadbalancing and use_aux_loss can be True"

        mlp_block = MLP_MAP[mlp.lower()]  # SwiGLU is default

        self.experts = nn.ModuleList(
            [mlp_block(dim, hidden_dim) for i in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [mlp_block(dim, hidden_dim) for i in range(self.num_shared_experts)]
            )

        if train_experts_biases:
            # not part of graph
            self.e_expert_biases = nn.Parameter(torch.zeros(self.num_experts))
            self.expert_update_rate = expert_update_rate
            self.train_experts_biases = train_experts_biases

    def forward(self, x: torch.Tensor, return_with_info: bool = False, *args, **kwargs):
        batch_size, seq_len, dim = x.shape  # (batch_size, seq_len, dim)

        if self.num_shared_experts > 0:
            shared = 0
            for shared_expert in self.shared_experts:
                shared += shared_expert(x)

        # (batch_size , seq_len, dim) -> (batch_size * seq_len, dim)
        x = x.view(batch_size * seq_len, dim)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len, num_experts)
        scores = self.gate(x)

        # expert_indices -> (batch_size * seq_len, num_experts_per_tok)
        _, expert_indices = torch.topk(
            scores + self.e_expert_biases, self.num_experts_per_tok, dim=-1
        )

        #  -> (batch_size * seq_len * num_experts_per_tok ) 1D
        flat_expert_indices = expert_indices.view(-1)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len * num_experts_per_tok, dim)
        # create copied of inputs for each expert
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)

        # (total_tokens,dim)
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)

        for idx, expert in enumerate(self.experts):
            # filtered_x - selected toks that to be sent to nth expert
            filtered_x = x[flat_expert_indices == idx]
            output[flat_expert_indices == idx] = expert(filtered_x).to(x.dtype)

        ## adding router to graph, so model can learn routing
        # router_probs -> (batch_size * seq_len, num_experts_per_tok)
        router_probs, _ = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        if self.use_aux_loss:
            aux_loss = self.aux_loss(router_probs)
        else:
            aux_loss = 0
        # -> (batch_size * seq_len, num_experts_per_tok)
        router_probs:Tensor = router_probs.sigmoid()
        router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        output = output.view(*router_probs.shape, -1)
        router_probs = router_probs.unsqueeze(-1)
        output = output * router_probs

        # sum up experts outputs
        # batch_size * seq_len, num_experts_per_tok, dim -> batch_size * seq_len, dim
        output = output.sum(dim=1)

        ## aux loss free loadbalancing by updating expert biases
        if self.training and self.train_experts_biases:
            self.update_experts_biases(expert_indices)

        output = shared + output.view(batch_size, seq_len, dim)

        if return_with_info:
            return output, router_probs
        
        if self.use_aux_loss:
            return output, aux_loss, maximal_violation(expert_indices, self.num_experts)

        return output, 0, maximal_violation(expert_indices, self.num_experts)

    def update_experts_biases(self, expert_indices: Tensor):
        expert_indices = expert_indices.clone().detach().reshape(-1)
        expert_frequencies = torch.bincount(
            expert_indices, minlength=self.num_experts
        ) / len(expert_indices)
        mean = expert_frequencies.mean()
        error = mean - expert_frequencies
        self.e_expert_biases.data = (
            self.e_expert_biases.data + self.expert_update_rate * torch.sign(error)
        )
        
    def aux_loss(self, router_probs: Tensor):
        total_tokens, _ = router_probs.shape
        # avg over batch and seq_len -> (num_experts)
        expert_load = router_probs.sum(dim=0) / total_tokens
        ideal_load = router_probs.mean(dim=0)
        return (ideal_load * expert_load).sum() / self.num_experts
    

    def reset_parameters(self, init_std=None, factor:float=1.0):
        # Initialize gate weights
        gate_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.gate.weight,
            mean=0.0,
            std=gate_std,
            a=-3 * gate_std,
            b=3 * gate_std,
        )

        self.experts: list[MLP]

        # Reset parameters for each expert
        for expert in self.experts:
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters(init_std=init_std, factor=factor)



class SparseMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp: str = "swiglu",
    ):
        """
        Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers
        https://arxiv.org/abs/2303.01610
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        mlp_block = MLP_MAP[mlp]  # SwiGLU is default

        self.experts = nn.ModuleList([mlp_block(dim, hidden_dim) for i in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)



    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape  # (batch_size, seq_len, dim)

        # (batch_size , seq_len, dim) -> (batch_size * seq_len, dim)
        x = x.view(batch_size * seq_len, dim)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len, num_experts)
        scores = self.gate(x)

        # expert_weights -> (batch_size * seq_len, num_experts_per_tok)
        # expert_indices -> (batch_size * seq_len, num_experts_per_tok)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)

        # -> (batch_size * seq_len, num_experts_per_tok)
        expert_weights = expert_weights.softmax(dim=-1)

        #  -> (batch_size * seq_len * num_experts_per_tok ) 1D
        flat_expert_indices = expert_indices.view(-1)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len * num_experts_per_tok, dim)
        # create copied of inputs for each expert
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)

        # (total_tokens,dim)
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)

        for idx, expert in enumerate(self.experts):
            # filtered_x - selected toks that to be sent to nth expert
            filtered_x = x[flat_expert_indices == idx]
            output[flat_expert_indices == idx] = expert(filtered_x)

        # ->B,T,num_experts_per_tok,dim
        output = output.view(*expert_weights.shape, -1)
        expert_weights = expert_weights.unsqueeze(-1)

        output = output * expert_weights

        # sum up experts outputs
        # batch_size * seq_len, num_experts_per_tok, dim -> batch_size * seq_len, dim
        output = output.sum(dim=1)

        return output.view(batch_size, seq_len, dim), self.aux_loss(expert_indices)  #  batch_size, seq_len, dim

    
    def aux_loss(self, router_probs: Tensor):
        batch_size, seq_len, num_experts_per_tok = router_probs.shape
        
        # avg over batch and seq_len -> (num_experts)
        expert_load = router_probs.sum(dim=0).sum(dim=0) / (batch_size * seq_len)
        
        ideal_load = router_probs.mean(dim=0).mean(dim=0)
        
        return (ideal_load * expert_load).sum()
    
    def reset_parameters(self, init_std=None):
        # Initialize gate weights
        gate_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.gate.weight,
            mean=0.0,
            std=gate_std,
            a=-3 * gate_std,
            b=3 * gate_std,
        )
        
        self.experts:list[MLP]
        
        # Reset parameters for each expert
        for expert in self.experts:
            if hasattr(expert, 'reset_parameters'):
                expert.reset_parameters(init_std=init_std)
                
                
                
                
if __name__ == "__main__":
    B, T, C = 8, 16, 32 
    import lightning as L
    import time
    fabric = L.Fabric(accelerator="cuda", devices=1)
    with torch.device("cuda"):
        import torch.optim as optim
        torch.set_float32_matmul_precision('high')
        model = DSMoE(C, C,aux_free_loadbalancing=True)
        model = fabric.setup(model)
        print(model)
        time.sleep(4)
        model = torch.compile(model)
        optimizer = optim.AdamW(model.parameters())

        x = torch.randn(B, T, C)
        y = torch.randn(B, T, C)

        iters = 1000

        DEBUG = True
        model(x)
        DEBUG = False

        for _ in range(iters):
            p = model(x)
            loss = torch.nn.functional.mse_loss(p,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"{loss.item()=}")

        DEBUG = True
        model(x)
        DEBUG = False


        print(y.shape)

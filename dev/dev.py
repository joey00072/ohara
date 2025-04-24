from torch import nn
import torch

from ohara.modules.mlp import MLP_MAP

class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp: str = "swiglu",
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        mlp_block = MLP_MAP[mlp]  # SwiGLU is default

        self.experts = nn.ModuleList([mlp_block(dim, hidden_dim) for i in range(num_experts)])
    
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
    def expert_paralle(self):
        self.num_devices = torch.cuda.device_count()
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.num_devices)]
        for i in range(len(self.experts)):
            self.experts[i] = self.experts[i].to(self.devices[i%self.num_devices])

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
            filtered_x.to(self.devices[idx%self.num_devices])
            output[flat_expert_indices == idx] = expert(filtered_x)

        print(f"{output.shape=}, {expert_weights.shape=}")
        # ->B,T,num_experts_per_tok,dim
        output = output.view(*expert_weights.shape, -1)
        expert_weights = expert_weights.unsqueeze(-1)

        
        output = output * expert_weights

        # sum up experts outputs
        # batch_size * seq_len, num_experts_per_tok, dim -> batch_size * seq_len, dim
        output = output.sum(dim=1)

        return output.view(batch_size, seq_len, dim)  #  batch_size, seq_len, dim

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
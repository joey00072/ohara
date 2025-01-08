from re import DEBUG
import torch
import torch.nn as nn

from ohara.modules.mlp import MLP_MAP, MLP


# This might not me most efficient implementation of MOE
# but it is easy to understand
# TODO: Write a more efficient implementation and more types or moe

DEBUG = False

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



    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape  # (batch_size, seq_len, dim)

        # (batch_size , seq_len, dim) -> (batch_size * seq_len, dim)
        x = x.view(batch_size * seq_len, dim)

        # (batch_size * seq_len, dim) -> (batch_size * seq_len, num_experts)
        scores = self.gate(x)

        # expert_weights -> (batch_size * seq_len, num_experts_per_tok)
        # expert_indices -> (batch_size * seq_len, num_experts_per_tok)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)

        if DEBUG:
            lst = [0]*4
            for _idx in expert_indices.reshape(-1):
                lst[_idx]+=1
            for idx,item in enumerate(lst):
                print(f"{idx=} {item=}")
            print("-------------")

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
                
if __name__ == "__main__":
    B, T, C = 8, 16, 32 

    with torch.device("cuda"):
        import torch.optim as optim
        torch.set_float32_matmul_precision('high')
        model = MoE(C, C)
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
            # print(f"{loss.item()=}")

        DEBUG = True
        model(x)
        DEBUG = False


        print(y.shape)

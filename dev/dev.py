import torch
import torch.nn.functional as F
from torch import Tensor

from ohara.modules.pscan import pscan

torch.manual_seed(0)


def scan(a: Tensor, b: Tensor):
    return pscan(a.unsqueeze(-1), b.unsqueeze(-1)).squeeze(-1)


def build_mask(seq_len, sliding_window_attention=False, window_size=1):
    mask = torch.full((seq_len, seq_len), float("-inf"))

    assert window_size != 0, "window_size cannot be 0"
    if not sliding_window_attention:
        window_size = seq_len

    row_indices = torch.arange(seq_len).unsqueeze(-1)
    col_indices = torch.arange(seq_len)
    distance = row_indices - col_indices

    mask[(distance >= 0) & (distance <= (window_size - 1))] = 0

    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


B, T, C = 1, 8, 2

x = torch.randn(B, T, C)
y = torch.randn(B, T, C)
h = torch.zeros(B, 1, C)

hs = []

for idx in range(T):
    h = h * x[:, idx] + y[:, idx]
    hs.append(h)

print(torch.stack(hs, dim=1).squeeze(-1))

print(scan(x, y).half())

import torch.nn as nn

resolve_ffn_act_fn = lambda x: F.silu


class DbrxExpertGLU(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, ffn_act_fn: dict
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.v1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.activation_fn = resolve_ffn_act_fn(ffn_act_fn)

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[
            expert_idx
        ]
        expert_v1 = self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[
            expert_idx
        ]
        expert_w2 = self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[
            expert_idx
        ]

        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1


class HydraLora(nn.Module):
    def __init__(self, moe_num_experts, ffn_hidden_size, hidden_size, rank):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_size = hidden_size
        self.rank = rank

        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.lora_a = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, hidden_size))

    def forward(self, x: Tensor, expert_idx: int) -> Tensor:
        expert_w1 = self.w1.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.hidden_size,
        )[expert_idx]
        expert_lora_a = self.lora_a.view(
            self.moe_num_experts,
            self.ffn_hidden_size,
            self.rank,
        )[expert_idx]
        expert_lora_b = self.lora_b.view(
            self.moe_num_experts,
            self.rank,
            self.hidden_size,
        )[expert_idx]

        wx = x.matmul(expert_w1.t())
        lora = x.matmul(expert_lora_a.t()).matmul(expert_lora_b.t())
        return wx + lora

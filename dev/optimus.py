import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass


@dataclass
class Config:
    vocab_size = 65
    seq_len = 64
    d_model = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.2
    multiple_of = 4
    bias = True


class Attention(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        g = self.gate(x)

        k = k.view(
            seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)
        g = g.view(seq_len, self.num_heads, self.head_dim)

        k = k.transpose(0, 1)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)
        g = g.transpose(0, 1)

        attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
        attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
        attn_mtx = self.attn_dropout(attn_mtx)

        v = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)
        g = torch.matmul(attn_mtx, g)
        # restore time as batch dimension and concat heads
        v = v.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        g = g.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        output = self.proj(F.silu(g) * v)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output

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
    num_head = 4
    num_layer = 4
    dropout = 0.2
    multiple_of = 4
    bias = True


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 4,
        dropout: float = None,
        bias: bool = False,
    ):
        """
        This is named Feed Forward but this is acually SwiGlu layer  ## Dont confuse with F.gelu
        SwiGlu: https://arxiv.org/abs/2002.05202v1
        Llama: https://arxiv.org/abs/2302.13971 (used in)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        d_model = model_args.d_model
        self.num_head = model_args.num_head
        self.head_dim = model_args.d_model // model_args.num_head

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
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

        k = k.view(
            seq_len, self.num_head, self.head_dim
        )  # shape = (B, seq_len, num_head, head_dim)
        q = q.view(seq_len, self.num_head, self.head_dim)
        v = v.view(seq_len, self.num_head, self.head_dim)

        k = k.transpose(0, 1)  # shape = (B, num_head, seq_len, head_dim)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()

        self.attn = Attention(model_args)
        self.ff = FeedForward(
            dim=model_args.d_model,
            multiple_of=model_args.multiple_of,
            dropout=model_args.dropout,
        )

        self.norm1 = nn.LayerNorm(model_args.d_model)
        self.norm2 = nn.LayerNorm(model_args.d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, model_args: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.word_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.pos_emb = nn.Embedding(model_args.seq_len, model_args.d_model)

        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layer)]
        )

        self.norm = nn.LayerNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(
            model_args.d_model, model_args.vocab_size, bias=False
        )

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, model_args.seq_len, model_args.seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, x):
        x = self.word_emb(x) + self.pos_emb(x)

        for layer in self.layers:
            x = layer(x, self.mask)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

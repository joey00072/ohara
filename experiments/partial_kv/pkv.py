import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections import OrderedDict

from ohara.modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope

from torch import Tensor

import math
from rich import print, traceback
traceback.install()


@dataclass
class Config(OrderedDict):
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int = None
    head_dim: int = None
    hidden_dim: int = None
    num_kv_heads: int = None
    num_layers: int = 4
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = False
    activation: str = "silu"
    mlp: str = "GLU"
    rope_head_dim: int = None
    kv_lora_rank: int = None
    attn_type: str = "mla"
    attention_type: str = "attention"

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)




class KAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_heads 
        # assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.query = nn.Linear(d_model, self.head_dim * self.num_heads, config.bias)
        self.proj = nn.Linear(self.head_dim * self.num_heads, d_model, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = k

        k = k.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )  # shape = (B, seq_len, num_kv_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)

        
        # Setting up v to be the same as k
        v = k
        
        # print(f"{q.shape=} {k.shape=} {v.shape=}")
        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)
        # print(f"{output.shape=}")
        output = output.transpose(1, 2).contiguous().view(batch, seq_len,self.head_dim * self.num_kv_heads )

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class PartialKVAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_heads

        # print(f"{int(1.5 * self.head_dim * self.num_kv_heads)=} {self.d_model=}")
        self.keyvalue = nn.Linear(self.d_model, int(1.5 * self.head_dim * self.num_kv_heads), config.bias)
        self.query = nn.Linear(self.d_model, self.head_dim * self.num_heads, config.bias)
        self.proj = nn.Linear(self.head_dim * self.num_heads, self.d_model, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        kv = self.keyvalue(x)
        q = self.query(x)
        
        # print(f"{self.num_heads * self.head_dim=} {kv.shape=} {x.shape=}")
        split_size = int(self.num_heads * self.head_dim* 1.5 / 3)
        # print(f"{split_size=}")
        k,s,v = kv.split([split_size, split_size, split_size], dim=-1)
        q1,q2 = q.split([split_size, split_size], dim=-1)
        
        # print(f"{q.shape=} {q1.shape=} {q2.shape=} {split_size=}")
        q1 = q1.view(batch, seq_len, self.num_heads, self.head_dim//2)
        q2 = q2.view(batch, seq_len, self.num_heads, self.head_dim//2)
        
        k = k.view(
            batch, seq_len, self.num_kv_heads, self.head_dim//2
        )  # shape = (B, seq_len, num_kv_heads, head_dim)
        
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim//2)
        
        s = s.view(batch, seq_len, self.num_kv_heads, self.head_dim//2)

        q2, k = apply_rope(q2, k, freqs_cis)
        
        k = torch.cat([s,k], dim=-1)
        v = torch.cat([s,v], dim=-1)
        q = torch.cat([q1,q2], dim=-1)
        

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.head_dim * self.num_kv_heads)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output



class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_heads
   
        # print(f"{self.num_heads=} {self.head_dim=}")
        self.key = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.query = nn.Linear(d_model, self.head_dim * self.num_heads, config.bias)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.proj = nn.Linear(self.head_dim * self.num_kv_heads, d_model, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )  # shape = (B, seq_len, num_kv_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.head_dim * self.num_kv_heads )

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


if __name__ == "__main__":
    
    d_model = 1024
    num_heads = 16
    head_dim = d_model // num_heads
    kv_lora_rank = 64
    q_lora_rank = 3 * kv_lora_rank
    rope_head_dim = 32
    num_kv_heads = num_heads
    
    config = Config(
        vocab_size=30522,
        d_model=d_model,
        seq_len=2048,
        num_heads=num_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
        num_kv_heads=num_heads,
    )
    
    x = torch.randn(2, 10, d_model)
    freqs_cis = precompute_freqs_cis(config.head_dim, config.seq_len)
    
    attn = Attention(config)
    print(f"Model Size: {sum(p.numel() for p in attn.parameters())/1e6:.4f}M ")
    output = attn(x,None, freqs_cis)
    print(f"Attention KV cache size: {int(config.num_heads * config.head_dim * 2)}")
    assert output.shape == (2,10,d_model)
    
    
    config.num_heads = 21
    kattn = KAttention(config)
    print(f"Model Size: {sum(p.numel() for p in kattn.parameters())/1e6:.4f}M ")
    output = kattn(x,None, freqs_cis)
    print(f"KAttention KV cache size: {int(config.num_heads * config.head_dim)}")
    assert output.shape == (2,10,d_model)
    
    print("-"*100)
    config.num_heads = 18
    pkv = PartialKVAttention(config)
    freqs_cis = precompute_freqs_cis(config.head_dim//2, config.seq_len)
    print(f"Model Size: {sum(p.numel() for p in pkv.parameters())/1e6:.4f}M ")
    print(f"PKVAttention KV cache size: {int(1.5*config.num_heads * config.head_dim)}")
    output = pkv(x,None, freqs_cis)
    assert output.shape == (2,10,d_model)
    
    
    
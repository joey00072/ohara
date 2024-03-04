import torch
import torch.nn as nn

from dataclasses import dataclass

from ohara.modules.mlp import SwiGLU
from ohara.modules.norm import RMSNorm
from ohara.embedings_pos.xpos import XPos


@dataclass
class Config:
    vocab_size: int = 65
    seq_len: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = False
    eps: float = 1e-5


# rotary embedding


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] for odd number truncation
    # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector

    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real
    freqs_sin = torch.sin(freqs)  # imaginary
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    assert 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # keep 2nd (T) and last(freq) dim same else make dim 1 for freq_cis
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # print(shape)
    return freqs_cis.view(shape)


def apply_rope(k, q, freqs_sin, freqs_cos):
    # Idea suppose vector v = [x,y,x1,y1,...] # v.shape = dim
    # convert vetor into complex num # ie two vec one real, one imagery
    # [x,y,x1,y1,...] -> x+iy, x1+iy1
    # Multiplying by complex num == roatate vector
    # => (x + iy) * (cos + isin) -> x'+iy'
    # restack
    # x'+iy' -> [x',y',x1',y1'...]
    # you roated vector in chunks of two lfg!!!

    #  rehsape a shape (...,n )-> (..., n//2,2)
    q_cis = q.float().reshape(
        q.shape[:-1] + (-1, 2)
    )  # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(
        k.shape[:-1] + (-1, 2)
    )  # (B,T,nhead,C) -> (B,T,nhead,Cc,2)

    xq_r, xq_i = q_cis.unbind(
        -1
    )  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc)) split into two tuple
    xk_r, xk_i = k_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freqs_cos, xq_r)

    # e+if = (a+ib) * (c+di) = (ac-bd) + i (ad+bc)
    # a = xq_r , b = xq_i
    # c = fcos , d = fsin
    # ...
    # e = (ac-bd) = xq_r * freqs_cos - xq_i * freqs_sin
    # f = (c+di)  = xq_r * freqs_sin + xq_i * freqs_cos

    xq_out_r = (
        xq_r * freqs_cos - xq_i * freqs_sin
    )  # (ac-bd)   # shape =  # (B,T,nhead,Cc)
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # (ad+bc) * i
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # (ac-bd)
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # (ad+bc) * i

    # now we stack r,i -> [r,i,r2,i2]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)

    # flatten last two dimensions
    xq_out = xq_out.flatten(3)  # (B,T,nhead,C)
    xk_out = xk_out.flatten(3)  # (B,T,nhead,C)

    return xq_out.type_as(q), xk_out.type_as(q)


class Retation(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        self.d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.key_dim = self.d_model // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.key = nn.Linear(self.d_model, self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.gate = nn.Linear(self.d_model, self.d_model)
        self.proj = nn.Linear(self.d_model, self.d_model)

        self.norm = RMSNorm(self.head_dim, model_args.e)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(
        self, x: torch.Tensor, freqs_cos, freqs_sin, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        g = self.gate(x)

        k = k * self.scaling

        k = k.view(
            seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        k = k.transpose(0, 1)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)

        ret_mtx = torch.matmul(q, k.transpose(2, 3))
        ret_mtx = ret_mtx / ret_mtx.detach().abs().sum(dim=-1, keepdim=True).clamp(
            min=1, max=5e4
        )
        ret_mtx = ret_mtx + mask[:, :, :seq_len, :seq_len]

        output = torch.matmul(ret_mtx, v)  # (batch, n_head, seq_len, head_dim)

        output = self.norm(output)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        output = output * g

        output = self.proj(output)
        return output


class Block(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()

        self.attn = Retation(model_args)
        self.ff = SwiGLU(
            dim=model_args.d_model,
            multiple_of=model_args.multiple_of,
            dropout=model_args.dropout,
        )

        self.norm1 = nn.LayerNorm(model_args.d_model)
        self.norm2 = nn.LayerNorm(model_args.d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, mask)
        x = x + self.ff(self.norm2(x))
        return x


class RoFormer(nn.Module):
    def __init__(self, model_args: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.word_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.pos_emb = nn.Embedding(model_args.seq_len, model_args.d_model)

        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )

        self.norm = nn.LayerNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(
            model_args.d_model, model_args.vocab_size, bias=False
        )

        (cos, sin), mask = XPos(model_args.d_model, model_args.seq_len).forward(
            slen=model_args.seq_len
        )
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x):
        B, T = x.shape
        x = self.word_emb(x)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, self.mask)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

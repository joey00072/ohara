from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import math

from ohara.embeddings_pos.rotary import apply_rope

TensorTuple = tuple[Tensor, ...]


class CasualAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.idx = idx

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        self.reset_parameters()


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        freqs_cis: TensorTuple | None = None,
        verbose: bool = False,
        **kwargs: dict,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn and not verbose:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx_dropout = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx_dropout, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)

        if verbose:
            return output, {"idx": self.idx, "attn_mtx": attn_mtx}
        return output


    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in [self.key, self.query, self.value]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        idx: int | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.idx = idx

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        self.reset_parameters()



    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        freqs_cis: TensorTuple | None = None,
        verbose: bool = False,
        **kwargs: dict,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is None and self.flash_attn:
            ...

        if self.flash_attn and not verbose:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx_dropout = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx_dropout, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)

        if verbose:
            return output, {"idx": self.idx, "attn_mtx": attn_mtx}
        return output

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in [self.key, self.query, self.value]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )
        
        
class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.idx = idx
        self.query = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        self.reset_parameters()


    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        freqs_cis: TensorTuple | None = None,
        is_casual: bool = True,
        verbose: bool = False,
        **kwargs: dict,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor = self.key(x)
        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = query.view(batch, seq_len, self.num_heads, self.head_dim)
        v = value.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn and not verbose:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx_dropout = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx_dropout, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)

        if verbose:
            return output, {"idx": self.idx, "attn_mtx": attn_mtx}
        return output


    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in [self.query]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )
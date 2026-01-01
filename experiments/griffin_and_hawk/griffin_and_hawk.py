from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ohara.embeddings_pos.rotary import apply_rope, precompute_freqs_cis
from ohara.modules.pscan import pscan
from ohara.modules.mlp import GEGLU
from ohara.modules.norm import RMSNorm

from dataclasses import dataclass

from huggingface_hub import PyTorchModelHubMixin


## paper https://arxiv.org/abs/2402.19427


def scan(x: Tensor, h: Tensor) -> Tensor:
    # mamba uses B,L,D,N to resues existing pscan
    # we need B,L,D. thats why this squeezy ness
    # TODO: use triton scan to make gpu go burr
    return pscan(x.unsqueeze(-1), h.unsqueeze(-1)).squeeze(-1)


@dataclass
class ModelType:
    MQA: str = "MQA"
    Hawk: str = "Hawk"
    Griffin: str = "Griffin"


@dataclass
class HnGConfig:
    model_type: str = ModelType.Griffin  # MQA, Hawk, Griffin
    vocab_size: int = 51200
    seq_len: int = 2048  # keep in power of 2 for faster scan
    dim: int = 256  # keep in muliple of 128 as per paper
    hidden_dim: int = dim * (3 / 2)
    num_heads: int = 32
    num_layers: int = 32
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    num_kv_heads: int = 1
    kernel_size: int = 4
    sliding_window_attention = False
    window_size: int = 128
    weight_tying: bool = True


class RG_LRU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.forget_lambda = nn.Parameter(torch.linspace(-4.323, -9, dim))

        # Why this Constant is 8 Paper offer no explaintion
        self.C = 8
        with torch.no_grad():
            self.input_proj.weight.normal_(std=dim**-0.5)
            self.gate_proj.weight.normal_(std=dim**-0.5)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        input_gate: torch.Tensor = self.input_proj(x)
        recurrence_gate: torch.Tensor = self.gate_proj(x)

        # ℎ𝑡    =  𝛼(𝑟𝑡)ℎ𝑡−1 + 𝛽(𝑟𝑡)𝑥𝑡             ...1
        # xbeta =  𝛽(𝑟𝑡)𝑥𝑡                        ...2
        # rest recurrace will calcuate with scan
        # h(t) = parallel_scan( a(rt), xbeta )   ...3
        alpha = (-self.C * F.softplus(self.forget_lambda) * recurrence_gate.sigmoid()).exp()

        beta = (1 - alpha**2 + 1e-6).sqrt()
        xbeta: Tensor = beta * input_gate.sigmoid() * x
        h = scan(alpha.mT.contiguous(), xbeta.mT.contiguous()).mT
        # TODO: wirte recurrence for inference
        return h


class HawkMixer(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 1024,
        hidden_dim: int | None = None,
        expansion_factor: float = 1.5,
        kernel_size: int = 4,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else int(dim * expansion_factor)

        self.input_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            bias=True,
            kernel_size=kernel_size,
            groups=hidden_dim,
            padding=kernel_size - 1,
        )
        self.linear_rnn = RG_LRU(hidden_dim)
        self.output = nn.Linear(hidden_dim, dim, bias=False)

        with torch.no_grad():
            self.input_proj.weight.normal_(std=dim**-0.5)
            self.gate_proj.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=hidden_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.shape
        # So linear rnn + conv can gets you close to transformer
        # to ssm hippo theory required :)

        gate = self.gate_proj(x)
        x = self.input_proj(x)

        x = self.conv(x.mT)[..., :seq_len].mT
        h = self.linear_rnn(x)
        x = self.output(F.gelu(gate) * h)
        return x


class CasualMultiQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        num_kv_heads: int = None,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        print(f"{num_kv_heads=}")
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is None else num_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.idx = idx

        self.key = nn.Linear(dim, self.head_dim * self.num_heads)
        self.query = nn.Linear(dim, self.head_dim * self.num_kv_heads)
        self.value = nn.Linear(dim, self.head_dim * self.num_kv_heads)
        self.proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        freqs_cis: tuple[Tensor, ...] | None = None,
        verbose: bool = False,
        **kwargs: dict,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape

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

        if self.num_kv_heads != self.num_heads:
            print(f"{self.idx=} {k.shape=} {q.shape=} {v.shape=}")
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)
            print(f"{self.idx=} {k.shape=} {q.shape=} {v.shape=}")
            exit(0)

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
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)

        if verbose:
            return output, {"idx": self.idx, "attn_mtx": attn_mtx}
        return output


class MQABlock(nn.Module):
    def __init__(
        self,
        cfg: HnGConfig,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        self.idx = idx

        self.mqa = CasualMultiQueryAttention(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, idx=idx
        )
        self.mlp = GEGLU(dim=cfg.dim, hidden_dim=cfg.hidden_dim)

        self.norm1 = RMSNorm(dim=cfg.dim, eps=cfg.eps)
        self.norm2 = RMSNorm(dim=cfg.dim, eps=cfg.eps)

    def forward(
        self, x: Tensor, mask: Tensor, freqs_cis: tuple[Tensor, ...] | None = None, **kwargs
    ):
        x = x + self.mqa(self.norm1(x), mask, freqs_cis, **kwargs)
        x = x + self.mlp(self.norm2(x))
        return x


class HawkBlock(nn.Module):
    def __init__(
        self,
        cfg: HnGConfig,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        self.idx = idx

        self.hawk = HawkMixer(dim=cfg.dim, hidden_dim=cfg.hidden_dim, kernel_size=cfg.kernel_size)
        self.mlp = GEGLU(dim=cfg.dim, hidden_dim=cfg.hidden_dim)

        self.norm1 = RMSNorm(dim=cfg.dim, eps=cfg.eps)
        self.norm2 = RMSNorm(dim=cfg.dim, eps=cfg.eps)

    def forward(self, x: Tensor, *args, **kwargs):
        x = x + self.hawk(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GriffiBlock(nn.Module):
    def __init__(
        self,
        cfg: HnGConfig,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        self.idx = idx

        self.mqa = CasualMultiQueryAttention(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, idx=idx
        )
        self.hawk = HawkMixer(dim=cfg.dim, hidden_dim=cfg.hidden_dim, kernel_size=cfg.kernel_size)
        self.mlp = GEGLU(dim=cfg.dim, hidden_dim=cfg.hidden_dim)

        self.norm1 = RMSNorm(dim=cfg.dim, eps=cfg.eps)
        self.norm2 = RMSNorm(dim=cfg.dim, eps=cfg.eps)
        self.norm3 = RMSNorm(dim=cfg.dim, eps=cfg.eps)

        # In paper they did not specify if they shared norm is mqa and hawk for griffin
        # Assume they did

    def forward(
        self, x: Tensor, mask: Tensor, freqs_cis: tuple[Tensor, ...] | None = None, **kwargs
    ):
        x = x + self.mqa(self.norm1(x), mask, freqs_cis, **kwargs)
        x = x + self.hawk(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x


class HawkAndGriffin(nn.Module, PyTorchModelHubMixin):  # they refer this as MQA Base line
    def __init__(self, cfg: HnGConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        MODEL_TYPE = {"mqa": MQABlock, "hawk": HawkBlock, "griffin": GriffiBlock}
        Block = MODEL_TYPE[cfg.model_type.lower()]

        self.layers = nn.ModuleList([Block(cfg, idx=idx) for idx in range(cfg.num_layers)])

        self.norm = nn.LayerNorm(cfg.dim)
        self.vocab_proj = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # they did not use weight tying in paper woth I dont think it matter for small models
        # its keeps model size small (for small models lol)
        if cfg.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight
        self.mask = self.build_mask(cfg.seq_len, cfg.sliding_window_attention, cfg.window_size)

        self.cis = precompute_freqs_cis(cfg.dim // cfg.num_heads, cfg.seq_len * 2)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        _, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.cis[0][:seqlen].to(device), self.cis[1][:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_mask(self, seq_len, sliding_window_attention=False, window_size=1):
        mask: Tensor = torch.full((seq_len, seq_len), float("-inf"))

        assert window_size != 0, "window_size cannot be 0"
        if not sliding_window_attention:
            window_size = seq_len

        row_indices: Tensor = torch.arange(seq_len).unsqueeze(-1)
        col_indices: Tensor = torch.arange(seq_len)
        distance = row_indices - col_indices

        mask[(distance >= 0) & (distance <= (window_size - 1))] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

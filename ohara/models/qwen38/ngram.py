"""Bigram/trigram features injected at the second layer.

Hash layout follows the Apache-2.0 Transformers Qwen4Exp implementation
(Copyright 2026 The Qwen Team and HuggingFace Inc.); see docs/qwen38.md.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .context_parallel import causal_conv
from .ops import RMSNorm


def hash_multipliers(vocab_size, seed):
    mask = (1 << 64) - 1
    gamma = 0x9E3779B97F4A7C15
    bound = max(1, ((1 << 63) - 1) // vocab_size // 2)
    multipliers = []
    for i in range(3):
        value = (seed + gamma * (i + 2)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        multipliers.append(2 * ((value ^ (value >> 31)) % bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def prime_sizes(start, count):
    sizes = []
    candidate = max(2, start)
    while len(sizes) < count:
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            sizes.append(candidate)
        candidate += 1
    return sizes


class NGram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        head_sizes = prime_sizes(cfg.ngram_vocab, 2 * cfg.ngram_heads)
        sizes = torch.tensor(head_sizes)
        offsets = sizes.cumsum(0) - sizes
        self.register_buffer("sizes", sizes)
        self.register_buffer("offsets", offsets)
        self.register_buffer("multipliers", hash_multipliers(cfg.vocab_size, cfg.ngram_seed))
        total = math.ceil(sum(head_sizes) / 128) * 128
        self.embedding = nn.Embedding(total, cfg.ngram_dim // (2 * cfg.ngram_heads))
        self.key = nn.Linear(cfg.ngram_dim, cfg.branches * cfg.hidden_size, bias=False)
        self.value = nn.Linear(cfg.ngram_dim, cfg.hidden_size, bias=False)
        self.key_norm = RMSNorm((cfg.branches, cfg.hidden_size), cfg.rms_eps)
        self.query_norm = RMSNorm((cfg.branches, cfg.hidden_size), cfg.rms_eps)
        self.conv_norm = RMSNorm((cfg.branches, cfg.hidden_size), cfg.rms_eps)
        self.conv = nn.Conv1d(
            cfg.branches * cfg.hidden_size, cfg.branches * cfg.hidden_size,
            cfg.conv_size, groups=cfg.branches * cfg.hidden_size, dilation=3, bias=False,
        )

    def indices(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)[None, :]
        eos = torch.where(tokens == self.cfg.eos_token_id, positions, -1)
        starts = F.pad(eos.cummax(1).values[:, :-1], (1, 0), value=-1) + 1
        shifted = [tokens]
        for shift in (1, 2):
            previous = F.pad(tokens, (shift, 0), value=self.cfg.eos_token_id)[:, :tokens.size(1)]
            shifted.append(torch.where(positions - starts >= shift, previous, self.cfg.eos_token_id))
        mixed = shifted[0] * self.multipliers[0]
        result = []
        for order in (1, 2):
            mixed = mixed ^ (shifted[order] * self.multipliers[order])
            heads = slice((order - 1) * self.cfg.ngram_heads, order * self.cfg.ngram_heads)
            result.append(mixed[..., None].remainder(self.sizes[heads]) + self.offsets[heads])
        return torch.cat(result, dim=-1)

    def lookup(self, tokens, context=None):
        history = (tokens if context is None else context.halo(tokens, 2, self.cfg.eos_token_id))
        indices = self.indices(history)[:, -tokens.size(1):]
        # Moving just this table to CPU permits host-resident capacity at inference.
        return self.embedding(indices.to(self.embedding.weight.device)).flatten(-2).to(tokens.device)

    def forward(self, residual, embeddings, context=None):
        cfg = self.cfg
        key = self.key_norm(self.key(embeddings).unflatten(-1, (cfg.branches, cfg.hidden_size)))
        query = self.query_norm(residual)
        score = (key.float() * query.float()).sum(-1, keepdim=True) / math.sqrt(cfg.hidden_size)
        score = score.sign() * score.abs().clamp_min(1e-6).sqrt()
        value = self.value(embeddings).unsqueeze(-2) * score.sigmoid().to(residual.dtype)
        conv_input = self.conv_norm(value).flatten(-2)
        local = F.silu(causal_conv(self.conv, conv_input, context)).reshape_as(value)
        return residual + value + local

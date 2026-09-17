import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .context_parallel import causal_conv
from .ops import RMSNorm, delta, rope, sparse_attention


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.key_width = cfg.linear_key_heads * cfg.linear_key_dim
        self.value_width = cfg.linear_value_heads * cfg.linear_value_dim
        width = 2 * self.key_width + self.value_width
        self.qkv = nn.Linear(cfg.hidden_size, width, bias=False)
        self.conv = nn.Conv1d(width, width, cfg.conv_size, groups=width, bias=False)
        self.gate = nn.Linear(cfg.hidden_size, self.value_width, bias=False)
        self.decay = nn.Linear(cfg.hidden_size, cfg.linear_value_heads, bias=False)
        self.beta = nn.Linear(cfg.hidden_size, cfg.linear_value_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(cfg.linear_value_heads).uniform_(0.01, 16).log())
        self.dt_bias = nn.Parameter(torch.ones(cfg.linear_value_heads))
        self.norm = RMSNorm(cfg.linear_value_dim, cfg.rms_eps)
        self.output = nn.Linear(self.value_width, cfg.hidden_size, bias=False)

    def forward(self, x, context=None):
        cfg = self.cfg
        content = F.silu(causal_conv(self.conv, self.qkv(x), context))
        q, k, v = content.split((self.key_width, self.key_width, self.value_width), dim=-1)
        q, k = (a.unflatten(-1, (cfg.linear_key_heads, cfg.linear_key_dim)) for a in (q, k))
        repeat = cfg.linear_value_heads // cfg.linear_key_heads
        q, k = (a.repeat_interleave(repeat, dim=2) for a in (q, k))
        v = v.unflatten(-1, (cfg.linear_value_heads, cfg.linear_value_dim))
        g = -self.A_log.float().exp() * F.softplus(self.decay(x).float() + self.dt_bias.float())
        beta = self.beta(x).sigmoid()
        y = (delta(q, k, v, g, beta, cfg.backend) if context is None
             else context.delta(q, k, v, g, beta, cfg.backend))
        y = self.norm(y).flatten(-2) * self.gate(x).sigmoid()
        return self.output(y), x.new_zeros(())


def select_blocks(scores, positions, block_size, token_budget):
    """Select complete blocks, then append the causal incomplete tail."""
    count = min(token_budget // block_size, scores.size(-1))
    ends = (torch.arange(scores.size(-1), device=scores.device) + 1) * block_size - 1
    visible = ends[None, None, :] <= positions[None, :, None]
    ranked = scores.masked_fill(~visible, -torch.inf)
    selected = ranked.topk(count, dim=-1).indices
    valid = visible.expand_as(scores).gather(-1, selected)
    slots = torch.arange(block_size, device=scores.device)
    expanded = selected[..., None] * block_size + slots
    expanded = expanded.masked_fill(~valid[..., None], -1).flatten(-2)
    tail_start = ((positions + 1) // block_size) * block_size
    tail = tail_start[:, None] + slots[:-1]
    tail = tail.masked_fill(tail > positions[:, None], -1)
    tail = tail[None].expand(scores.size(0), -1, -1)
    return torch.cat((expanded, tail), dim=-1), selected, valid


def indexer_kl(scores, selected, valid, q, k, indices, block_size, query_valid=None):
    """Sparse-stage KL: max-pool detached attention probabilities per selected block."""
    count = selected.size(-1)
    if count == 0:
        return scores.sum() * 0
    with torch.no_grad():
        rows = torch.arange(q.size(0), device=q.device)[:, None, None]
        keys = k[rows, indices.clamp_min(0)]
        grouped_q = q.unflatten(2, (k.size(2), q.size(2) // k.size(2)))
        logits = torch.einsum("bqhgd,bqkhd->bqhgk", grouped_q.float(), keys.float())
        logits = logits.flatten(2, 3) / math.sqrt(q.size(-1))
        logits = logits.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
        teacher = logits.softmax(-1).mean(2)[..., :count * block_size]
        teacher = teacher.unflatten(-1, (count, block_size)).amax(-1) * valid
        teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-20)
    chosen = scores.gather(-1, selected).masked_fill(~valid, -torch.inf)
    any_valid = valid.any(-1, keepdim=True)
    chosen = torch.where(any_valid, chosen, torch.zeros_like(chosen))
    log_prob = chosen.log_softmax(-1).masked_fill(~valid, 0)
    terms = teacher * (teacher.clamp_min(1e-20).log() - log_prob)
    if query_valid is not None:
        terms = terms * query_valid[..., None]
    return terms.sum()


class SparseAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_gate = nn.Linear(cfg.hidden_size, 2 * cfg.num_heads * cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.hidden_size, 2 * cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.index_qk = nn.Linear(cfg.hidden_size, (cfg.index_heads + 1) * cfg.index_dim, bias=False)
        self.index_q_norm = RMSNorm(cfg.index_dim, cfg.rms_eps)
        self.index_k_norm = RMSNorm(cfg.index_dim, cfg.rms_eps)
        self.output = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, context=None):
        cfg = self.cfg
        batch, length, _ = x.shape
        positions = (torch.arange(length, device=x.device) if context is None else context.positions(x))
        q, gate = self.q_gate(x).unflatten(-1, (cfg.num_heads, 2 * cfg.head_dim)).chunk(2, -1)
        k, v = self.kv(x).chunk(2, -1)
        k, v = (a.unflatten(-1, (cfg.num_kv_heads, cfg.head_dim)) for a in (k, v))
        q = rope(self.q_norm(q), positions, cfg.rotary_dim, cfg.rope_theta)
        k = rope(self.k_norm(k), positions, cfg.rotary_dim, cfg.rope_theta)
        index_q, raw_k = self.index_qk(x).split((cfg.index_heads * cfg.index_dim, cfg.index_dim), -1)
        index_q = self.index_q_norm(index_q.unflatten(-1, (cfg.index_heads, cfg.index_dim)))
        index_q = rope(index_q, positions, cfg.rotary_dim, cfg.rope_theta)
        blocks = length // cfg.block_size
        pooled = raw_k[:, :blocks * cfg.block_size].reshape(batch, blocks, cfg.block_size, cfg.index_dim)
        pooled = self.index_k_norm(pooled.float().mean(2).to(raw_k.dtype))
        pooled = rope(pooled.unsqueeze(2), positions[::cfg.block_size][:blocks],
                      cfg.rotary_dim, cfg.rope_theta).squeeze(2)
        if context is not None:
            if length % cfg.block_size:
                raise ValueError("QSA context shards must contain a multiple of block_size tokens")
            k, v, pooled = (context.gather(t) for t in (k, v, pooled))
        query_valid = context.valid_mask(x) if context is not None else torch.ones((batch, length), dtype=torch.bool, device=x.device)
        outputs, losses = [], []

        def attend(query, index_query, keys, values, block_keys, query_positions, valid_queries):
            scores = torch.einsum("bqhd,bnd->bqhn", index_query.float(), block_keys.float())
            scores = scores.relu().sum(2) / math.sqrt(cfg.index_dim)
            indices, selected, valid = select_blocks(scores, query_positions, cfg.block_size, cfg.token_budget)
            y = sparse_attention(query, keys, values, indices, cfg.backend)
            loss = (indexer_kl(scores, selected, valid, query, keys, indices, cfg.block_size, valid_queries)
                    if self.training and cfg.index_aux_coef else query.new_zeros(()))
            return y, loss

        for start in range(0, length, cfg.query_chunk_size):
            stop = min(start + cfg.query_chunk_size, length)
            args = (q[:, start:stop], index_q[:, start:stop], k, v, pooled, positions[start:stop], query_valid[:, start:stop])
            # Recompute scores in backward instead of retaining O(sequence^2 / block_size).
            y, loss = (checkpoint(attend, *args, use_reentrant=False)
                       if self.training and torch.is_grad_enabled() else attend(*args))
            outputs.append(y)
            losses.append(loss)
        output = torch.cat(outputs, dim=1) * gate.sigmoid()
        loss = torch.stack(losses).sum()
        loss = (loss / (batch * length) if context is None else context.mean_loss(loss, query_valid.sum()))
        return self.output(output.flatten(-2)), loss

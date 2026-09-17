from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ohara.modules.moe_grouped import GroupedMoE

from .attention import GatedDeltaNet, SparseAttention
from .config import Config
from .ngram import NGram
from .ops import RMSNorm


class GatedResidual(nn.Module):
    def __init__(self, cfg, write=True):
        super().__init__()
        self.branches = cfg.branches
        self.norm = RMSNorm((cfg.branches, cfg.hidden_size), cfg.rms_eps)
        self.down = nn.Linear(cfg.branches * cfg.hidden_size, cfg.residual_rank, bias=False)
        self.up = nn.Linear(cfg.residual_rank, cfg.branches * cfg.hidden_size, bias=False)
        self.write = nn.Linear(cfg.branches * cfg.hidden_size, cfg.branches, bias=False) if write else None

    def forward(self, residual):
        normalized = self.norm(residual)
        flat = normalized.flatten(-2)
        gate = self.up(F.silu(self.down(flat) / self.branches)).sigmoid().reshape_as(residual)
        read = (normalized * gate).mean(-2)
        write = 2 * (self.write(flat) / self.branches).sigmoid() if self.write is not None else None
        return read, write


class Experts(GroupedMoE):
    def __init__(self, cfg):
        super().__init__(
            cfg.hidden_size, cfg.expert_dim, cfg.num_experts, cfg.top_k,
            num_shared_experts=1, gate_fn="softmax", quantile_balancing=False,
        )
        self.backend = cfg.backend
        self.shared_output_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, x, context=None):
        flat = x.reshape(-1, self.dim)
        probabilities = self.router(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else flat.dtype
        if flat.is_cuda and self.backend != "torch":
            if dtype != torch.bfloat16 or self.dim % 8 or self.hidden_dim % 8:
                raise ValueError("grouped CUDA experts require BF16 and dimensions divisible by 8")
            out = self._dispatch_grouped(flat, indices, weights)
        else:
            out = self._dispatch_reference(flat, indices, weights)
        out = out + self._shared(flat) * self.shared_output_gate(flat).sigmoid()
        if self.training:
            if context is not None:
                auxiliary = context.router_loss(probabilities, indices, x.shape[:2])
            else:
                load = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
                load = load / indices.numel()
                auxiliary = self.num_experts * (load * probabilities.mean(0)).sum()
        else:
            auxiliary = flat.new_zeros(())
        return out.reshape_as(x), auxiliary


class Layer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.attention = (SparseAttention(cfg) if (index + 1) % cfg.attention_interval == 0
                          else GatedDeltaNet(cfg))
        self.attention_residual = GatedResidual(cfg)
        self.experts = Experts(cfg)
        self.expert_residual = GatedResidual(cfg)

    def forward(self, residual, context=None):
        x, write = self.attention_residual(residual)
        output, index_loss = self.attention(x, context)
        residual = residual + output.unsqueeze(-2) * write.unsqueeze(-1)
        x, write = self.expert_residual(residual)
        output, router_loss = self.experts(x, context)
        residual = residual + output.unsqueeze(-2) * write.unsqueeze(-1)
        return residual, index_loss, router_loss


@dataclass
class Output:
    logits: torch.Tensor | None
    loss: torch.Tensor | None
    auxiliary_loss: torch.Tensor


class MultiTokenPrediction(nn.Module):
    """One shared QSA/MoE layer unrolled over teacher-forced future tokens."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        # MTP normalizes the widened hidden vector before a per-branch projection.
        self.hidden_norm = RMSNorm(cfg.branches * cfg.hidden_size, cfg.rms_eps)
        self.embedding_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layer = Layer(cfg, cfg.attention_interval - 1)
        self.readout = GatedResidual(cfg, write=False)

    def forward(self, residual, next_embeddings, context=None):
        hidden = self.hidden_norm(residual.flatten(-2)).reshape_as(residual)
        fused = self.hidden_proj(hidden) + self.embedding_proj(self.embedding_norm(next_embeddings)).unsqueeze(-2)
        residual, index_loss, router_loss = self.layer(fused, context)
        hidden, _ = self.readout(residual)
        return residual, hidden, index_loss, router_loss


class Qwen38(nn.Module):
    """Trainable text-backbone example; see docs/qwen38.md for scope and kernels."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.ngram = NGram(cfg)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_layers))
        self.readout = GatedResidual(cfg, write=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.mtp = MultiTokenPrediction(cfg) if cfg.mtp_steps else None
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, Experts):
                for weight in (module.w_gate, module.w_up, module.w_down):
                    nn.init.normal_(weight, std=cfg.init_std)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None, *, loss_chunk_size=128, ignore_index=-1, context=None):
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("tokens must have shape (batch, nonempty sequence)")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must contain integer IDs")
        if targets is not None and (targets.shape != tokens.shape or loss_chunk_size < 1):
            raise ValueError("targets must match tokens and loss_chunk_size must be positive")
        if self.config.backend == "cuda" and not tokens.is_cuda:
            raise ValueError("backend=cuda requires CUDA tensors")
        residual = self.embedding(tokens).unsqueeze(-2).expand(-1, -1, self.config.branches, -1)
        # Launch the lookup before the first layer; the table can reside on the host.
        ngrams = self.ngram.lookup(tokens, context)
        index_losses, router_losses = [], []
        for i, layer in enumerate(self.layers):
            if i + 1 == self.config.ngram_layer:
                residual = self.ngram(residual, ngrams, context)
            residual, index_loss, router_loss = (
                checkpoint(layer, residual, context, use_reentrant=False)
                if self.config.activation_checkpointing and self.training
                else layer(residual, context)
            )
            if isinstance(layer.attention, SparseAttention):
                index_losses.append(index_loss)
            router_losses.append(router_loss)
        hidden, _ = self.readout(residual)
        auxiliary = self.config.router_aux_coef * torch.stack(router_losses).mean()
        if index_losses:
            auxiliary = auxiliary + self.config.index_aux_coef * torch.stack(index_losses).mean()
        if targets is None:
            return Output(self.lm_head(hidden), None, auxiliary)

        loss = self._loss(hidden, targets, loss_chunk_size, ignore_index, context)
        if self.mtp is not None and self.training:
            mtp_losses = []
            total_length = tokens.size(1) * (context.size if context else 1)
            for step in range(1, min(self.config.mtp_steps + 1, total_length)):
                mtp_context = context.for_mtp(step) if context else None
                next_tokens = context.shift(tokens, step, self.config.eos_token_id) if context else tokens[:, step:]
                next_targets = context.shift(targets, step, ignore_index) if context else targets[:, step:]
                args = (residual if context else residual[:, :-1], self.embedding(next_tokens), mtp_context)
                residual, draft, index_loss, router_loss = (
                    checkpoint(self.mtp, *args, use_reentrant=False)
                    if self.config.activation_checkpointing else self.mtp(*args)
                )
                mtp_losses.append(self._loss(draft, next_targets, loss_chunk_size, ignore_index, mtp_context))
                mtp_losses[-1] = mtp_losses[-1] + self.config.index_aux_coef * index_loss
                mtp_losses[-1] = mtp_losses[-1] + self.config.router_aux_coef * router_loss
            if mtp_losses:
                auxiliary = auxiliary + self.config.mtp_loss_coef * torch.stack(mtp_losses).mean()
        return Output(None, loss + auxiliary, auxiliary)

    def _loss(self, hidden, targets, loss_chunk_size, ignore_index, context=None):
        flat_hidden, flat_targets = hidden.flatten(0, 1), targets.flatten()

        def score(features, labels):
            return F.cross_entropy(self.lm_head(features).float(), labels,
                                   reduction="sum", ignore_index=ignore_index)

        losses = []
        for start in range(0, flat_targets.numel(), loss_chunk_size):
            args = (flat_hidden[start:start + loss_chunk_size], flat_targets[start:start + loss_chunk_size])
            loss = (checkpoint(score, *args, use_reentrant=False)
                    if self.training and torch.is_grad_enabled() else score(*args))
            losses.append(loss)
        count = (flat_targets != ignore_index).sum()
        total = torch.stack(losses).sum()
        return total / count.clamp_min(1) if context is None else context.mean_loss(total, count)

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("model must return logits with shape (batch, sequence, vocabulary)")
    return logits


def _score(model, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    logits = _logits(model(input_ids))
    shifted_labels = labels[:, 1:]
    count = int(shifted_labels.ne(-100).sum().item())
    if count == 0:
        return 0.0, 0
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return float(losses[shifted_labels.reshape(-1).ne(-100)].double().sum().item()), count


def _result(protocol: str, nll: float, tokens: int) -> dict[str, float | int | str]:
    if tokens < 1:
        raise ValueError("evaluation produced no predicted tokens")
    loss = nll / tokens
    return {
        "protocol": protocol,
        "loss_nats_per_token": loss,
        "token_perplexity": math.exp(loss),
        "predicted_tokens": tokens,
    }


@torch.inference_mode()
def fixed_block_perplexity(
    model,
    token_ids: torch.Tensor,
    *,
    device: str | torch.device,
    sequence_length: int = 2048,
    batch_size: int = 8,
) -> dict[str, float | int | str]:
    """GPTQ-style token perplexity over non-overlapping, full-length blocks."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if sequence_length < 2 or batch_size < 1:
        raise ValueError("sequence_length must be at least 2 and batch_size must be positive")

    usable = token_ids.numel() // sequence_length * sequence_length
    if usable == 0:
        raise ValueError("not enough tokens for one complete evaluation block")
    blocks = token_ids[:usable].view(-1, sequence_length)

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    try:
        for offset in range(0, len(blocks), batch_size):
            batch = blocks[offset : offset + batch_size].to(device)
            nll, count = _score(model, batch, batch)
            total_nll += nll
            total_tokens += count
    finally:
        model.train(was_training)
    return _result("fixed_nonoverlapping", total_nll, total_tokens)


@torch.inference_mode()
def sliding_window_perplexity(
    model,
    token_ids: torch.Tensor,
    *,
    device: str | torch.device,
    sequence_length: int = 2048,
    stride: int = 512,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float | int | str]:
    """Hugging Face-style strided token perplexity, scoring each eligible token once."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if sequence_length < 2 or not 1 <= stride < sequence_length:
        raise ValueError("stride must be between 1 and sequence_length - 1")
    if token_ids.numel() < 2:
        raise ValueError("at least two tokens are required")

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    try:
        for begin in range(0, token_ids.numel(), stride):
            end = min(begin + sequence_length, token_ids.numel())
            target_length = end - previous_end
            input_ids = token_ids[begin:end].unsqueeze(0).to(device)
            labels = input_ids.clone()
            labels[:, :-target_length] = -100
            nll, count = _score(model, input_ids, labels)
            total_nll += nll
            total_tokens += count
            previous_end = end
            if progress is not None:
                progress(end, token_ids.numel())
            if end == token_ids.numel():
                break
    finally:
        model.train(was_training)
    return _result(f"sliding_stride_{stride}", total_nll, total_tokens)

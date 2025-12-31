import torch
import torch.nn.functional as F
from torch import Tensor
import time
from torch.nn.attention import flex_attention as flex_attn_mod


def make_swa_mask(
    seq_len: int, window_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    causal = k_idx[None, :] <= q_idx[:, None]
    in_window = (q_idx[:, None] - k_idx[None, :]) < window_size
    allow = causal & in_window
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask.masked_fill_(allow, 0.0)
    return mask


def sliding_window_attention_with_mask(q: Tensor, k: Tensor, v: Tensor, window_size: int = 16):
    _, T, _ = q.shape
    mask = make_swa_mask(T, window_size, device=q.device, dtype=q.dtype)
    wei = q @ k.transpose(-1, -2)
    wei = wei + mask
    wei = F.softmax(wei, dim=-1)

    return wei @ v


def sliding_window_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int = 16,
    *,
    block_size: int = 128,
    enable_gqa: bool = False,
):

    def mask_mod(batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        in_window = (q_idx - kv_idx) < window_size
        causal = kv_idx <= q_idx
        return in_window & causal

    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeeze_heads = True
    else:
        squeeze_heads = False

    B, H, Q_LEN, _ = q.shape
    _, _, KV_LEN, _ = k.shape
    block_mask = flex_attn_mod.create_block_mask(
        mask_mod=mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=q.device,
        BLOCK_SIZE=block_size,
    )

    out = flex_attn_mod.flex_attention(
        q, k, v, block_mask=block_mask, enable_gqa=enable_gqa
    )
    return out.squeeze(1) if squeeze_heads else out


def compare_swa_mask_vs_flex(
    q: Tensor, k: Tensor, v: Tensor, *, window_size: int, block_size: int = 128
):
    mask_out = sliding_window_attention_with_mask(q, k, v, window_size=window_size)
    flex_out = sliding_window_flex_attention(
        q, k, v, window_size=window_size, block_size=block_size
    )
    max_abs = (mask_out - flex_out).abs().max().item()
    return mask_out, flex_out, max_abs


if __name__ == "__main__":
    B, T, C = 5, 1000, 8

    q = torch.rand(B, T, C)
    k = torch.rand(B, T, C)
    v = torch.rand(B, T, C)

    window_size = 3

    start = time.perf_counter()
    mask_out, flex_out, max_abs = compare_swa_mask_vs_flex(
        q, k, v, window_size=window_size
    )
    end = time.perf_counter()
    print(f"compare_time:{(end-start)}")
    print(f"max_abs:{max_abs}")

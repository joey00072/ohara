"""Indexed sparse attention with online softmax and recomputed backward.

One program handles one query/head. Only selected KV rows are read; no dense
attention matrix or gathered KV tensor is allocated. Backward accumulates shared
KV gradients in FP32 with atomics, so it is not bitwise deterministic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, I, O, L,
             T: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
             D: tl.constexpr, N: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    head = row % H
    token = (row // H) % T
    batch = row // (T * H)
    kv_head = head // (H // HK)
    d = tl.arange(0, BD)
    n = tl.arange(0, BN)
    q = tl.load(Q + row * D + d, d < D, 0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    normalizer = tl.full((), 0, tl.float32)
    output = tl.full((BD,), 0, tl.float32)
    for start in range(tl.cdiv(N, BN)):
        slot = start * BN + n
        idx = tl.load(I + (batch * T + token) * N + slot, slot < N, -1)
        valid = (slot < N) & (idx >= 0) & (idx < S)
        address = ((batch * S + idx[:, None]) * HK + kv_head) * D + d[None, :]
        key = tl.load(K + address, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        value = tl.load(V + address, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * (D ** -0.5)
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 0))
        # Initial tiles may contain only -1 slots before the causal tail.
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(score - safe_max)
        output = output * correction + tl.sum(probability[:, None] * value, 0)
        normalizer = normalizer * correction + tl.sum(probability, 0)
        maximum = next_max
    denominator = tl.maximum(normalizer, 1e-30)
    tl.store(O + row * D + d, output / denominator, d < D)
    tl.store(L + row, maximum + tl.log(denominator))


@triton.jit
def _backward(Q, K, V, I, O, L, DO, DQ, DK, DV,
              T: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
              D: tl.constexpr, N: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    head = row % H
    token = (row // H) % T
    batch = row // (T * H)
    kv_head = head // (H // HK)
    d = tl.arange(0, BD)
    n = tl.arange(0, BN)
    q = tl.load(Q + row * D + d, d < D, 0).to(tl.float32)
    out = tl.load(O + row * D + d, d < D, 0).to(tl.float32)
    grad = tl.load(DO + row * D + d, d < D, 0).to(tl.float32)
    lse = tl.load(L + row)
    lse = tl.where(lse == -float("inf"), 0., lse)
    delta = tl.sum(grad * out, 0)
    dq = tl.full((BD,), 0, tl.float32)
    for start in range(tl.cdiv(N, BN)):
        slot = start * BN + n
        idx = tl.load(I + (batch * T + token) * N + slot, slot < N, -1)
        valid = (slot < N) & (idx >= 0) & (idx < S)
        address = ((batch * S + idx[:, None]) * HK + kv_head) * D + d[None, :]
        mask = valid[:, None] & (d[None, :] < D)
        key = tl.load(K + address, mask, 0).to(tl.float32)
        value = tl.load(V + address, mask, 0).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * (D ** -0.5)
        probability = tl.where(valid, tl.exp(score - lse), 0.)
        ds = probability * (tl.sum(value * grad[None, :], 1) - delta) * (D ** -0.5)
        dq += tl.sum(ds[:, None] * key, 0)
        tl.atomic_add(DK + address, ds[:, None] * q[None, :], mask)
        tl.atomic_add(DV + address, probability[:, None] * grad[None, :], mask)
    tl.store(DQ + row * D + d, dq, d < D)


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices):
        if any(x.ndim != 4 for x in (q, k, v)) or indices.ndim != 3:
            raise ValueError("expected Q/K/V (B,T,H,D) and indices (B,T,K)")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("sparse kernel supports FP16, BF16, or FP32")
        if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
            raise ValueError("Q/K/V must share dtype and device")
        if indices.dtype not in (torch.int32, torch.int64) or indices.device != q.device:
            raise ValueError("indices must be integers on the Q/K/V device")
        q, k, v, indices = (x.contiguous() for x in (q, k, v, indices))
        batch, length, heads, dim = q.shape
        if k.shape != v.shape or k.size(0) != batch or k.size(3) != dim:
            raise ValueError("Q/K/V dimensions do not match")
        if k.size(2) < 1 or heads < 1 or heads % k.size(2) or indices.shape[:2] != (batch, length):
            raise ValueError("invalid grouped heads or selected indices")
        if not 1 <= dim <= 256 or indices.size(-1) < 1:
            raise ValueError("sparse kernel requires head_dim <= 256 and at least one index slot")
        options = dict(T=length, S=k.size(1), H=heads, HK=k.size(2), D=dim,
                       N=indices.size(-1), BD=triton.next_power_of_2(dim), BN=32)
        output = torch.empty_like(q)
        lse = torch.empty((batch, length, heads), device=q.device, dtype=torch.float32)
        _forward[(batch * length * heads,)](q, k, v, indices, output, lse, **options)
        ctx.save_for_backward(q, k, v, indices, output, lse)
        ctx.options = options
        return output

    @staticmethod
    def backward(ctx, grad):
        q, k, v, indices, output, lse = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk, dv = (torch.zeros_like(x, dtype=torch.float32) for x in (k, v))
        _backward[(q.numel() // q.size(-1),)](
            q, k, v, indices, output, lse, grad.contiguous(), dq, dk, dv, **ctx.options,
        )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


def sparse_attention(q, k, v, indices):
    return _SparseAttention.apply(q, k, v, indices)

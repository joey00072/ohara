"""Checks for the shared building blocks in ohara.modules / embeddings_pos / utils."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from ohara.adaptor.dora import DoRALinear, replace_with_dora
from ohara.adaptor.lora import LoRALinear, replace_with_lora
from ohara.embeddings_pos.alibi import get_alibi_mask
from ohara.embeddings_pos.rotary import RoPE, apply_rope, precompute_freqs_cis
from ohara.embeddings_pos.xpos import XPos
from ohara.modules.attention import Attention, CasualAttention, CausalAttention
from ohara.modules.kv_cache import KVCache, dequantize_int8, quantize_int8
from ohara.modules.mlp import MLP_MAP
from ohara.modules.moe import MoE
from ohara.modules.norm import RMSNorm
from ohara.swa import make_swa_mask, sliding_window_attention_with_mask
from ohara.utils import BetterCycle, random_name
from ohara.utils.tools import build_mask


class TinyNetwork(nn.Module):
    """Covers the three ways a Linear can be nested, for adaptor replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.seq = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    def forward(self, x):
        return self.linear(x)


# --- KV cache ---------------------------------------------------------------


def make_cache(**kwargs) -> KVCache:
    return KVCache(shape=(1, 8, 2, 4), max_seq_length=8, **kwargs)


def test_kv_cache_returns_everything_written_so_far() -> None:
    cache = make_cache(dtype=torch.float32)
    first = torch.randn(1, 3, 2, 4)
    second = torch.randn(1, 1, 2, 4)

    keys, _ = cache.forward(first, first.clone(), 0)
    assert keys.shape == (1, 3, 2, 4)

    keys, values = cache.forward(second, second.clone(), 3)
    assert keys.shape == (1, 4, 2, 4)
    torch.testing.assert_close(keys[:, :3], first)
    torch.testing.assert_close(values[:, 3:], second)
    assert cache.length == 4


def test_kv_cache_rejects_non_sequential_writes() -> None:
    cache = make_cache(dtype=torch.float32)
    entry = torch.randn(1, 2, 2, 4)
    cache.forward(entry, entry.clone(), 0)
    with pytest.raises(ValueError, match="sequential"):
        cache.forward(entry, entry.clone(), 5)


def test_kv_cache_rejects_overflow() -> None:
    cache = make_cache(dtype=torch.float32)
    entry = torch.randn(1, 9, 2, 4)
    with pytest.raises(ValueError, match="max_sequence_length"):
        cache.forward(entry, entry.clone(), 0)


def test_kv_cache_rejects_mismatched_keys_and_values() -> None:
    cache = make_cache(dtype=torch.float32)
    with pytest.raises(ValueError, match="matching shapes"):
        cache.forward(torch.randn(1, 2, 2, 4), torch.randn(1, 3, 2, 4), 0)


def test_int8_kv_cache_dequantizes_on_read() -> None:
    """The int8 cache must hand back float values, not raw int8 codes."""
    cache = make_cache(dtype=torch.float32, int8=True)
    entry = torch.randn(1, 4, 2, 4)

    keys, values = cache.forward(entry, entry.clone(), 0)

    assert keys.dtype == torch.float32
    torch.testing.assert_close(keys, entry, rtol=0.05, atol=0.05)
    torch.testing.assert_close(values, entry, rtol=0.05, atol=0.05)


def test_int8_quantize_roundtrip() -> None:
    tensor = torch.randn(4, 16)
    quantized, scale, min_val = quantize_int8(tensor)
    assert quantized.dtype == torch.int8
    torch.testing.assert_close(
        dequantize_int8(quantized, scale, min_val), tensor, rtol=0.05, atol=0.05
    )


# --- rotary embeddings ------------------------------------------------------


def test_apply_rope_preserves_shape_and_norm() -> None:
    q = torch.randn(2, 6, 4, 8)
    k = torch.randn(2, 6, 4, 8)
    cis = precompute_freqs_cis(8, 16)

    rotated_q, rotated_k = apply_rope(q, k, cis)

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    # A rotation preserves the length of each rotated pair, so the norm holds.
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_apply_rope_leaves_position_zero_untouched() -> None:
    q = torch.randn(1, 4, 2, 8)
    rotated, _ = apply_rope(q, q.clone(), precompute_freqs_cis(8, 8))
    torch.testing.assert_close(rotated[:, 0], q[:, 0], rtol=1e-6, atol=1e-6)


def test_rope_module_passes_through_unrotated_tail() -> None:
    rope = RoPE(dims=4)
    x = torch.randn(1, 3, 8)
    out = rope(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out[..., 4:], x[..., 4:])


def test_rope_rejects_odd_width() -> None:
    # An odd width would broadcast two different-sized halves together.
    with pytest.raises(ValueError, match="even"):
        RoPE(dims=3)


# --- attention / norms / mlp ------------------------------------------------


def test_causal_attention_shape_and_aliases() -> None:
    attn = CausalAttention(d_model=32, num_heads=4).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        assert attn(x).shape == x.shape
    # The historic misspelling and the generic name both point at one class.
    assert CasualAttention is CausalAttention
    assert Attention is CausalAttention


def test_causal_attention_verbose_returns_causal_matrix() -> None:
    attn = CausalAttention(d_model=16, num_heads=2, idx=3).eval()
    with torch.no_grad():
        out, info = attn(torch.randn(1, 5, 16), verbose=True)
    assert out.shape == (1, 5, 16)
    assert info["idx"] == 3
    # Nothing may attend to the future.
    upper = info["attn_mtx"].triu(diagonal=1)
    assert torch.count_nonzero(upper) == 0


def test_causal_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalAttention(d_model=30, num_heads=4)


def test_rmsnorm_normalizes_scale() -> None:
    norm = RMSNorm(8)
    out = norm(torch.randn(4, 8) * 100)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", sorted(MLP_MAP))
def test_every_mlp_variant_preserves_shape(name: str) -> None:
    block = MLP_MAP[name](dim=16, hidden_dim=32).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert block(x).shape == x.shape


def test_moe_routes_without_changing_shape() -> None:
    moe = MoE(dim=16, hidden_dim=32, num_experts=4, num_experts_per_tok=2).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# --- positional biases and masks --------------------------------------------


def test_alibi_mask_shape_and_monotonic_penalty() -> None:
    mask = get_alibi_mask(number_of_heads=4, max_seq_len=5)
    assert mask.shape == (4, 5, 5)
    # Distance penalties grow (more negative) the further back a key is.
    row = mask[0, -1]
    assert row[0] < row[-1]


def test_xpos_returns_decay_mask_per_head() -> None:
    (cos, sin), decay_mask = XPos(dim=64, num_heads=4).forward(slen=8)
    assert cos.shape == sin.shape == (8, 16)
    assert decay_mask.shape == (4, 8, 8)
    # The decay mask is causal.
    assert torch.count_nonzero(decay_mask.triu(diagonal=1)) == 0


def test_build_mask_causal_and_sliding_window() -> None:
    causal = build_mask(4)
    assert causal.shape == (1, 1, 4, 4)
    assert torch.isinf(causal[0, 0, 0, 1]) and causal[0, 0, 1, 0] == 0

    windowed = build_mask(4, sliding_window_attention=True, window_size=2)
    # Row 3 may see positions 2 and 3 only.
    assert windowed[0, 0, 3, 3] == 0
    assert windowed[0, 0, 3, 2] == 0
    assert torch.isinf(windowed[0, 0, 3, 1])


def test_swa_mask_matches_manual_attention_window() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 4) for _ in range(3))
    window = 3

    mask = make_swa_mask(6, window, device=q.device, dtype=q.dtype)
    expected = torch.softmax(q @ k.transpose(-1, -2) + mask, dim=-1) @ v

    torch.testing.assert_close(
        sliding_window_attention_with_mask(q, k, v, window_size=window), expected
    )


# --- adaptors ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("replace", "adapter_type"),
    [(replace_with_lora, LoRALinear), (replace_with_dora, DoRALinear)],
)
def test_adaptor_replaces_every_nested_linear(replace, adapter_type) -> None:
    model = replace(TinyNetwork())

    assert isinstance(model.linear, adapter_type)
    assert all(isinstance(layer, adapter_type) for layer in model.layers)
    assert all(isinstance(layer, adapter_type) for layer in model.seq)

    with torch.no_grad():
        assert model(torch.randn(1, 2)).shape == (1, 2)


# --- misc utils -------------------------------------------------------------


def test_better_cycle_repeats_and_counts_epochs() -> None:
    cycle = BetterCycle([0, 1, 2, 3])
    seen = [next(cycle) for _ in range(9)]
    assert seen == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert cycle.idx == 2


def test_better_cycle_close_drops_iterator() -> None:
    cycle = BetterCycle([1, 2])
    next(cycle)
    cycle.close()
    assert cycle._iterator is None
    assert next(cycle) == 1


def test_random_name_is_slug_plus_timestamp() -> None:
    name = random_name()
    slug, _, stamp = name.rpartition("-")
    assert slug and slug.replace("-", "").isalnum()
    assert len(stamp.split("_")) == 6
    assert math.isclose(len(stamp), 19, abs_tol=1)

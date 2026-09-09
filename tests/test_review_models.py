"""Regression coverage for model/module issues in the September code review."""

import copy
import pickle
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ohara.adaptor.dora import dora_from_linear, mark_dora_as_trainable, merge_dora
from ohara.adaptor.lora import lora_from_linear, mark_lora_as_trainable, merge_lora
from ohara.models.llama import Config, Llama
from ohara.models.mamba import Mamba, MambaConfig
from ohara.models.phi import Phi, PhiConfig
from ohara.models.qwen3 import Qwen3, Qwen3Config
from ohara.models.retnet import Config as RetNetConfig, RetNet
from ohara.models.transformer import Config as TransformerConfig, Transformer
from ohara.modules.attention import CausalAttention
from ohara.modules.linear_rnn import RG_LRU
from ohara.modules.moe import MoE
from ohara.modules.moe_grouped import GroupedMoE
from ohara.modules.pscan import pscan


def tiny_config(**kwargs):
    return Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        max_sequence_length=16,
        dropout=0,
        **kwargs,
    )


@pytest.mark.parametrize("wrap", [lora_from_linear, dora_from_linear])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("rank", [0, 2])
def test_adapter_preserves_weights_output_and_freezes_base(wrap, bias, rank):
    linear = torch.nn.Linear(5, 7, bias=bias).double()
    x = torch.randn(3, 5, dtype=torch.float64)
    adapter = wrap(linear, rank=rank)
    torch.testing.assert_close(adapter(x), linear(x))
    assert (adapter.linear.bias is not None) == bias
    assert next(adapter.parameters()).dtype == torch.float64
    pickle.loads(pickle.dumps(adapter))
    (adapter.lora_trainable_only if wrap is lora_from_linear else adapter.dora_trainable_only)()
    assert all(not p.requires_grad for p in adapter.linear.parameters())
    if rank:
        with torch.no_grad():
            adapter.lora_B.normal_()
        expected = adapter(x)
        adapter.merge()
        torch.testing.assert_close(adapter(x), expected)


@pytest.mark.parametrize(
    "wrap,mark,merge",
    [
        (lora_from_linear, mark_lora_as_trainable, merge_lora),
        (dora_from_linear, mark_dora_as_trainable, merge_dora),
    ],
)
def test_adapter_target_selection(wrap, mark, merge):
    model = torch.nn.ModuleDict(
        {n: wrap(torch.nn.Linear(5, 5), rank=2) for n in ["chosen", "other"]}
    )
    mark(model, target_layer=["chosen"])
    assert model["chosen"].lora_A.requires_grad
    assert not model["other"].lora_A.requires_grad
    merge(model, target_layer=["chosen"])
    assert model["chosen"].merged and not model["other"].merged


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 16, 17, 33])
def test_parallel_scan_matches_time_loop_and_gradients(length):
    a = torch.rand(2, length, 3, 2, dtype=torch.float64, requires_grad=True)
    x = torch.randn_like(a, requires_grad=True)
    state = torch.zeros_like(x[:, 0])
    sequential = []
    for t in range(length):
        state = a[:, t] * state + x[:, t]
        sequential.append(state)
    expected = torch.stack(sequential, 1)
    actual = pscan(a, x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (a, x), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (a, x), upstream)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)


def test_rg_lru_scans_time():
    layer = RG_LRU(5).double()
    x = torch.randn(2, 7, 5, dtype=torch.float64)
    alpha = (-layer.C * F.softplus(layer.forget_lambda) * layer.gate_proj(x).sigmoid()).exp()
    updates = (1 - alpha.square() + 1e-6).sqrt() * layer.input_proj(x).sigmoid() * x
    state = torch.zeros_like(x[:, 0])
    expected = []
    for t in range(x.size(1)):
        state = alpha[:, t] * state + updates[:, t]
        expected.append(state)
    torch.testing.assert_close(layer(x), torch.stack(expected, 1))


@pytest.mark.parametrize("gate", ["softmax", "sigmoid"])
@pytest.mark.parametrize("grouped", [True, False])
def test_top_one_router_learns_and_empty_experts_have_gradients(grouped, gate):
    cls = GroupedMoE if grouped else MoE
    model = cls(dim=16, hidden_dim=24, num_experts=4, num_experts_per_tok=1, gate_fn=gate)
    if grouped:
        with torch.no_grad():
            model.w_down.normal_(std=0.1)
    model(torch.randn(1, 1, 16)).square().sum().backward()
    router = model.router if grouped else model.gate
    assert router.weight.grad.norm() > 0
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("exclusive", [True, False])
def test_grouped_dispatch_matches_reference_outputs_and_gradients(dtype, exclusive):
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("grouped matmul unavailable in this Torch version")
    module = (
        GroupedMoE(
            16,
            24,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            shared_exclusive=exclusive,
        )
        .to(dtype)
        .eval()
    )
    with torch.no_grad():
        module.w_down.normal_(std=0.1)
        module.shared_down.weight.normal_(std=0.1)
    reference = copy.deepcopy(module)
    x = torch.randn(5, 16, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    idx, weights, _, _ = module._route(x)
    ir, wr, _, _ = reference._route(xr)
    try:
        out = module._dispatch_grouped(x, idx, weights)
    except (RuntimeError, NotImplementedError) as exc:
        if "not implemented" in str(exc) or "Could not run" in str(exc):
            pytest.skip("CPU grouped matmul unavailable in this Torch version")
        raise
    expected = reference._dispatch_reference(xr, ir, wr)
    shared, shared_ref = module._shared(x), reference._shared(xr)
    if exclusive:
        out, expected = module._reject(out, shared), reference._reject(expected, shared_ref)
    out, expected = out + shared, expected + shared_ref
    tol = dict(atol=0.01, rtol=0.06) if dtype == torch.bfloat16 else dict(atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(out, expected, **tol)
    out.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, **tol)
    for p, r in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, r.grad, **tol)


@pytest.mark.parametrize("boolean", [True, False])
def test_causal_attention_mask_matches_diagnostic_path(boolean):
    attn = CausalAttention(16, 4).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.eye(5, dtype=torch.bool)
    mask[0] = False  # SDPA defines fully masked queries to return zero.
    if not boolean:
        mask = torch.zeros(5, 5).masked_fill(~mask, float("-inf"))
    normal = attn(x, mask=mask)
    diagnostic, _ = attn(x, mask=mask, verbose=True)
    torch.testing.assert_close(normal, diagnostic, atol=1e-6, rtol=1e-5)


def test_retnet_causal_prefix_invariance():
    model = RetNet(
        RetNetConfig(vocab_size=32, seq_len=8, d_model=16, num_heads=2, num_layers=1, dropout=0)
    ).eval()
    ids = torch.randint(0, 32, (2, 8))
    changed = ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4], atol=1e-7, rtol=1e-6)
    # Position rotations must influence cross-position scores.
    before = model(ids)
    model.freq_cos.fill_(1)
    model.freq_sin.zero_()
    assert not torch.allclose(before, model(ids))


def test_transformer_activation_and_length_contract():
    cfg = TransformerConfig(32, 8, 16, 32, 4, num_layers=1, dropout=0, activation="relu")
    relu, gelu = Transformer(cfg), Transformer(replace(cfg, activation="gelu"))
    gelu.load_state_dict(relu.state_dict())
    ids = torch.randint(0, 32, (2, 8))
    assert not torch.allclose(relu(ids), gelu(ids))
    with pytest.raises(ValueError, match="seq_len"):
        relu(torch.zeros(1, 9, dtype=torch.long))


@pytest.mark.parametrize(
    "model_cls,config",
    [
        (Llama, tiny_config(weight_tying=True)),
        (
            Qwen3,
            Qwen3Config(
                vocab_size=32,
                max_sequence_length=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                weight_tying=True,
            ),
        ),
    ],
)
def test_checkpoint_meta_load_preserves_ties_dtype_and_cached_decode(tmp_path, model_cls, config):
    model = model_cls(config).eval()
    model.save_pretrained(tmp_path)
    loaded = model_cls.from_pretrained(tmp_path, dtype=torch.float64)
    assert loaded.token_emb.weight is loaded.vocab_proj.weight
    assert all(p.dtype == torch.float64 and not p.is_meta for p in loaded.parameters())
    ids = torch.randint(0, 32, (2, 7))
    torch.testing.assert_close(loaded(ids).float(), model(ids), atol=1e-5, rtol=1e-4)
    cache = loaded.build_kv_cache(2, max_sequence_length=7)
    pieces = [loaded(ids[:, i : i + 1], cache, i) for i in range(7)]
    torch.testing.assert_close(torch.cat(pieces, 1), loaded(ids))
    assert cache[0].key.size(1) == 7
    for entry in cache:
        entry.reset()
    torch.testing.assert_close(loaded(ids, cache, 0), loaded(ids))


def test_tied_checkpoint_requires_embedding_and_supports_either_key(tmp_path):
    model = Llama(tiny_config(weight_tying=True)).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    original = {k: v.clone() for k, v in load_file(path).items()}
    for retained in ["token_emb.weight", "vocab_proj.weight"]:
        state = {
            k: v
            for k, v in original.items()
            if k not in {"token_emb.weight", "vocab_proj.weight"} or k == retained
        }
        save_file(state, path)
        loaded = Llama.from_pretrained(tmp_path)
        assert loaded.token_emb.weight is loaded.vocab_proj.weight
        torch.testing.assert_close(loaded.token_emb.weight, model.token_emb.weight)
    save_file(
        {k: v for k, v in original.items() if k not in {"token_emb.weight", "vocab_proj.weight"}},
        path,
    )
    with pytest.raises(RuntimeError, match="missing keys"):
        Llama.from_pretrained(tmp_path)


def test_legacy_rope_is_rebuilt_for_target_config():
    model = Llama(tiny_config())
    target = Llama(replace(model.config, max_sequence_length=24, rope_theta=1000))
    expected = target.freq_cos.clone()
    target.load_state_dict(model.state_dict())
    torch.testing.assert_close(target.freq_cos, expected)


def test_mixed_dense_expert_widths():
    cfg = replace(
        tiny_config(),
        num_hidden_layers=2,
        moe_num_experts=4,
        moe_layer_interval=2,
        moe_expert_hidden_dim=8,
    )
    model = Llama(cfg)
    assert model.layers[0].ff.experts[0].up.out_features == 8
    assert model.layers[1].ff.up.out_features == 32


def test_mamba_pure_bfloat16_forward_and_step():
    model = Mamba(MambaConfig(d_model=8, n_layers=1, d_state=4)).to(torch.bfloat16).eval()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    actual = model(x)
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    # The recurrent block follows the same dtype boundary.
    block = model.layers[0].mixer
    cache = (None, torch.zeros(2, 16, 3, dtype=torch.bfloat16))
    out, _ = block.step(x[:, 0], cache)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_phi_cache_rotated_only_new_tokens_and_length_limit():
    model = Phi(
        PhiConfig(
            vocab_size=32,
            max_sequence_length=8,
            hidden_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            rotary_dim=0.5,
        )
    ).eval()
    ids = torch.randint(0, 32, (2, 6))
    cache = model.build_kv_cache(2)
    outputs = [model(ids[:, i : i + 1], cache, i) for i in range(6)]
    torch.testing.assert_close(torch.cat(outputs, 1), model(ids), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="max_sequence_length"):
        model(torch.zeros(1, 9, dtype=torch.long))

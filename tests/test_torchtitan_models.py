"""Regressions identified by the TorchTitan model comparison."""

import pytest
import torch
import torch.nn.functional as F

from ohara.models.qwen3 import Qwen3Config
from ohara.modules.moe import MoE
from ohara.modules.moe_grouped import GroupedMoE
from ohara.modules.router import RouterLinear


@pytest.mark.parametrize("kind", [MoE, GroupedMoE])
@pytest.mark.parametrize("quantile_balancing", [False, True])
def test_actual_moe_autocast_keeps_close_router_logits(kind, quantile_balancing):
    kwargs = {"num_shared_experts": 0} if kind is GroupedMoE else {}
    model = kind(
        dim=2, hidden_dim=4, num_experts=2, num_experts_per_tok=1,
        quantile_balancing=quantile_balancing, **kwargs,
    )
    router = model.router if kind is GroupedMoE else model.gate
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.001, 0.0], [1.002, 0.0]]))
    observed = []
    handle = router.register_forward_hook(lambda module, args, output: observed.append(output))
    x = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(x)
    handle.remove()
    assert model.expert_counts.tolist() == [0, 1]
    assert observed[0].dtype == torch.float32
    torch.testing.assert_close(observed[0], torch.tensor([[1.001, 1.002]]), rtol=0, atol=0)
    # This very input previously rounded both scores to 1.0 and selected expert 0.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rounded = F.linear(x.reshape(1, 2), router.weight).float()
    assert rounded.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_router_autocast_forward_and_gradients_match_fp32_projection(storage_dtype):
    torch.manual_seed(42)
    router = RouterLinear(13, 7, bias=True).to(storage_dtype)
    x = torch.randn(5, 13).to(storage_dtype).requires_grad_()
    ref_x = x.detach().float().requires_grad_()
    ref_weight = router.weight.detach().float().requires_grad_()
    ref_bias = router.bias.detach().float().requires_grad_()
    reference = F.linear(ref_x, ref_weight, ref_bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = router(x)
        loss = actual.square().sum()
    loss.backward()
    reference.square().sum().backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, expected in (
        (x.grad, ref_x.grad), (router.weight.grad, ref_weight.grad),
        (router.bias.grad, ref_bias.grad),
    ):
        torch.testing.assert_close(got, expected.to(storage_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("key", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("rope", [
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
    {"type": "linear", "factor": 2.0},
    {"rope_type": "default", "factor": 4.0},
    {"full_attention": {"rope_type": "yarn", "factor": 4.0}},
    "yarn",
])
def test_qwen_rejects_unsupported_rope_config(key, rope):
    payload = Qwen3Config().to_hf_config()
    payload[key] = rope
    with pytest.raises(ValueError, match="RoPE|mapping"):
        Qwen3Config.from_hf_config(payload)


@pytest.mark.parametrize("key", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("rope", [None, {}, {"rope_type": "default"}, {"type": "default"}])
def test_qwen_accepts_unscaled_rope_config(key, rope):
    expected = Qwen3Config(rope_theta=123456.0)
    payload = expected.to_hf_config()
    payload[key] = rope
    assert Qwen3Config.from_hf_config(payload) == expected


def test_qwen_default_rope_parameters_theta_survives_config_roundtrip():
    payload = Qwen3Config().to_hf_config()
    payload["rope_parameters"] = {"rope_type": "default", "rope_theta": 12345.0}
    config = Qwen3Config.from_hf_config(payload)
    assert config.rope_theta == 12345.0
    assert Qwen3Config.from_hf_config(config.to_hf_config()) == config

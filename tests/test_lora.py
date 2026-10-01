"""LoRA replacement, freezing, merging, and checkpoint round trips."""

import copy
import io
import pickle

import pytest
import torch
import torch.nn as nn

from ohara.adaptor.lora import (
    LoRALinear,
    lora_from_linear,
    mark_lora_as_trainable,
    merge_lora,
    replace_with_lora,
)


class TinyNetwork(nn.Module):
    """Covers the three ways a Linear can be nested, for adaptor replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.seq = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    def forward(self, x):
        return self.linear(x)


def test_lora_replaces_every_nested_linear() -> None:
    model = replace_with_lora(TinyNetwork())

    assert isinstance(model.linear, LoRALinear)
    assert all(isinstance(layer, LoRALinear) for layer in model.layers)
    assert all(isinstance(layer, LoRALinear) for layer in model.seq)

    with torch.no_grad():
        assert model(torch.randn(1, 2)).shape == (1, 2)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("rank", [0, 2])
def test_lora_preserves_weights_output_and_freezes_base(bias, rank):
    linear = torch.nn.Linear(5, 7, bias=bias).double()
    x = torch.randn(3, 5, dtype=torch.float64)
    adapter = lora_from_linear(linear, rank=rank)
    torch.testing.assert_close(adapter(x), linear(x))
    assert (adapter.linear.bias is not None) == bias
    assert next(adapter.parameters()).dtype == torch.float64
    pickle.loads(pickle.dumps(adapter))
    adapter.lora_trainable_only()
    assert all(not p.requires_grad for p in adapter.linear.parameters())
    if rank:
        with torch.no_grad():
            adapter.lora_B.normal_()
        expected = adapter(x)
        adapter.merge()
        torch.testing.assert_close(adapter(x), expected)


def test_lora_target_selection():
    model = torch.nn.ModuleDict(
        {n: lora_from_linear(torch.nn.Linear(5, 5), rank=2) for n in ["chosen", "other"]}
    )
    mark_lora_as_trainable(model, target_layer=["chosen"])
    assert model["chosen"].lora_A.requires_grad
    assert not model["other"].lora_A.requires_grad
    merge_lora(model, target_layer=["chosen"])
    assert model["chosen"].merged and not model["other"].merged


@pytest.mark.parametrize("merged", [False, True])
def test_lora_checkpoint_preserves_predictions(merged):
    model = torch.nn.Sequential(LoRALinear(4, 3, rank=2)).eval()
    with torch.no_grad():
        model[0].lora_B.normal_()
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    if merged:
        model[0].merge()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = torch.nn.Sequential(LoRALinear(4, 3, rank=2)).eval()
    # Loading must also reset a previously merged destination to unmerged.
    restored[0].merge()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored[0].merged is merged
    torch.testing.assert_close(restored(inputs), expected)
    restored[0].merge()
    torch.testing.assert_close(restored(inputs), expected)


def test_legacy_unmerged_lora_checkpoint_loads_strictly():
    model = LoRALinear(4, 3, rank=2).eval()
    with torch.no_grad():
        model.lora_B.normal_()
    state = copy.deepcopy(model.state_dict())
    del state["_merged"]
    restored = LoRALinear(4, 3, rank=2).eval()
    restored.merge()
    restored.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 4)
    assert not restored.merged
    torch.testing.assert_close(restored(inputs), model(inputs))

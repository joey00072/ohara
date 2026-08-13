"""Tests for the hyperspherical (constant-norm) optimizers, AdamH and MuonH.

Reference: Rethinking Language Model Scaling under Transferable Hypersphere
Optimization, https://arxiv.org/abs/2603.28743, cross-checked against Levanter's
``optim/adamh.py`` and ``optim/muonh.py``.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn.functional as F

from ohara.models.llama import Config, Llama
from ohara.optimizer import (
    MuonAdamW,
    _hypersphere_update_,
    _hypersphere_update_stacked_,
    build_adamh,
    build_muonh_adamh,
)


def small_llama(**overrides) -> Llama:
    defaults = dict(
        vocab_size=64,
        max_sequence_length=32,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        dropout=0.0,
        weight_tying=False,
    )
    torch.manual_seed(0)
    return Llama(Config(**{**defaults, **overrides}))


def train_steps(model: Llama, optimizer, steps: int = 20) -> list[float]:
    generator = torch.Generator().manual_seed(1)
    data = torch.randint(0, 64, (4, 16), generator=generator)
    target = torch.randint(0, 64, (4, 16), generator=generator)
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(data).reshape(-1, 64), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# --- the projection itself ---------------------------------------------------


def test_projection_preserves_frobenius_norm_exactly() -> None:
    torch.manual_seed(0)
    params = [torch.randn(6, 4), torch.randn(8, 8)]
    before = [p.norm().clone() for p in params]

    _hypersphere_update_(params, [torch.randn_like(p) for p in params], 0.1)

    for p, norm in zip(params, before, strict=True):
        torch.testing.assert_close(p.norm(), norm, rtol=1e-6, atol=1e-6)


def test_projection_is_invariant_to_update_scale() -> None:
    """Only the update's direction matters, so momentum conventions cannot matter."""
    torch.manual_seed(0)
    base = [torch.randn(6, 4) for _ in range(2)]
    updates = [torch.randn(6, 4) for _ in range(2)]

    unscaled = [p.clone() for p in base]
    _hypersphere_update_(unscaled, updates, 0.1)

    rescaled = [p.clone() for p in base]
    _hypersphere_update_(rescaled, [u * 137.0 for u in updates], 0.1)

    for left, right in zip(unscaled, rescaled, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)


def test_relative_step_size_tracks_learning_rate() -> None:
    torch.manual_seed(0)
    original = torch.randn(16, 16)
    for learning_rate in (0.01, 0.05, 0.2):
        param = [original.clone()]
        _hypersphere_update_(param, [torch.randn(16, 16)], learning_rate)
        relative = ((param[0] - original).norm() / original.norm()).item()
        # The projection shortens the chord slightly; it never lengthens it.
        assert 0.7 * learning_rate <= relative <= learning_rate * 1.001


def test_batched_and_foreach_projections_agree() -> None:
    torch.manual_seed(0)
    params = [torch.randn(6, 4) for _ in range(3)]
    updates = [torch.randn(6, 4) for _ in range(3)]

    foreach = [p.clone() for p in params]
    _hypersphere_update_(foreach, updates, 0.1)

    stacked = torch.stack([p.clone() for p in params])
    _hypersphere_update_stacked_(stacked, torch.stack(updates), 0.1)

    for left, right in zip(foreach, stacked.unbind(0), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)


def test_zero_update_leaves_parameters_untouched() -> None:
    params = [torch.randn(4, 4)]
    original = params[0].clone()
    _hypersphere_update_(params, [torch.zeros(4, 4)], 0.1)
    torch.testing.assert_close(params[0], original, rtol=1e-6, atol=1e-6)


# --- group validation --------------------------------------------------------


def test_hyperspherical_group_rejects_weight_decay() -> None:
    param = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="weight_decay=0"):
        MuonAdamW(
            [
                {
                    "kind": "adamw",
                    "hypersphere": True,
                    "params": [param],
                    "lr": 0.01,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                }
            ]
        )


def test_hyperspherical_group_rejects_vector_parameters() -> None:
    param = torch.nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="matrix parameters only"):
        MuonAdamW(
            [
                {
                    "kind": "adamw",
                    "hypersphere": True,
                    "params": [param],
                    "lr": 0.01,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                }
            ]
        )


# --- builders ----------------------------------------------------------------


def test_adamh_partition_covers_every_parameter_once() -> None:
    model = small_llama()
    optimizer = build_adamh(model)

    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}

    for group in optimizer.param_groups:
        assert group["weight_decay"] == 0.0
        if group.get("hypersphere", False):
            assert all(p.ndim == 2 for p in group["params"])


def test_adamh_puts_the_lm_head_on_the_sphere() -> None:
    model = small_llama()
    optimizer = build_adamh(model)
    head_id = id(model.vocab_proj.weight)
    owning = next(g for g in optimizer.param_groups if any(id(p) == head_id for p in g["params"]))
    assert owning.get("hypersphere") is True


def test_muonh_assigns_muonh_adamh_and_adamw_to_the_right_parameters() -> None:
    model = small_llama()
    optimizer = build_muonh_adamh(model)

    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}

    def group_for(param) -> dict:
        return next(g for g in optimizer.param_groups if any(p is param for p in g["params"]))

    hidden = group_for(model.layers[0].attn.query.weight)
    assert (hidden["kind"], hidden["hypersphere"]) == ("muon", True)

    # The lm_head is a matrix but is not orthogonalized.
    head = group_for(model.vocab_proj.weight)
    assert (head["kind"], head["hypersphere"]) == ("adamw", True)

    embedding = group_for(model.token_emb.weight)
    assert embedding["kind"] == "adamw"
    assert not embedding.get("hypersphere", False)

    norm_gain = group_for(model.layers[0].norm1.weight)
    assert norm_gain["kind"] == "adamw"
    assert not norm_gain.get("hypersphere", False)


def test_muonh_and_adamh_share_one_learning_rate() -> None:
    """The paper's practical claim: both normalize relative step, so one LR serves."""
    model = small_llama()
    optimizer = build_muonh_adamh(model, learning_rate=0.05)
    spherical = [g for g in optimizer.param_groups if g.get("hypersphere")]
    assert len(spherical) >= 2
    assert {g["lr"] for g in spherical} == {0.05}


@pytest.mark.parametrize("builder", [build_adamh, build_muonh_adamh])
def test_builders_reject_non_positive_learning_rates(builder) -> None:
    with pytest.raises(ValueError, match="positive"):
        builder(small_llama(), learning_rate=0.0)


@pytest.mark.parametrize("builder", [build_adamh, build_muonh_adamh])
def test_training_reduces_loss_and_holds_every_matrix_norm(builder) -> None:
    model = small_llama()
    watched = {
        name: parameter.detach().norm().clone()
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and "token_emb" not in name
    }
    optimizer = builder(model)

    losses = train_steps(model, optimizer, steps=25)
    assert losses[-1] < losses[0]

    for name, norm in watched.items():
        current = dict(model.named_parameters())[name].detach().norm()
        torch.testing.assert_close(current, norm, rtol=1e-4, atol=1e-4, msg=f"{name} drifted")


def test_learning_rate_controls_rotation_not_scale() -> None:
    """A tiny LR must leave hidden matrices where they started."""
    rotations = {}
    for learning_rate in (1e-8, 0.2):
        model = small_llama()
        before = model.layers[0].attn.query.weight.detach().clone()
        optimizer = build_muonh_adamh(model, learning_rate=learning_rate)
        train_steps(model, optimizer, steps=15)
        after = model.layers[0].attn.query.weight.detach()
        rotations[learning_rate] = (
            (after * before).sum() / (after.norm() * before.norm())
        ).item()

    assert rotations[1e-8] > 0.999  # effectively frozen
    assert rotations[0.2] < 0.9  # meaningfully rotated


def _serialized(state: dict) -> dict:
    """Round-trip through torch.save/load, the way a real checkpoint resume does.

    Loading a ``state_dict`` in-process would leave the two optimizers *sharing*
    momentum buffers: ``Optimizer.load_state_dict`` only shallow-copies, and
    ``Tensor.to(dtype)`` is a no-op returning the same object when the dtype
    already matches. Serializing forces independent tensors.
    """
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@pytest.mark.parametrize("builder", [build_adamh, build_muonh_adamh])
def test_checkpoint_resume_reproduces_the_trajectory(builder) -> None:
    model = small_llama()
    optimizer = builder(model)
    train_steps(model, optimizer, steps=3)

    resumed = small_llama()
    resumed.load_state_dict(_serialized(model.state_dict()))
    resumed_optimizer = builder(resumed)
    resumed_optimizer.load_state_dict(_serialized(optimizer.state_dict()))

    continued = train_steps(model, optimizer, steps=3)
    replayed = train_steps(resumed, resumed_optimizer, steps=3)
    assert continued == pytest.approx(replayed, rel=1e-6, abs=1e-7)

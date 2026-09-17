import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.tensor_parallel import TensorParallelExperts


def _tiny_config():
    return Config(
        vocab_size=41,
        hidden_size=32,
        num_layers=2,
        attention_interval=2,
        num_heads=4,
        num_kv_heads=1,
        head_dim=8,
        rotary_dim=8,
        linear_key_heads=2,
        linear_value_heads=4,
        linear_key_dim=8,
        linear_value_dim=8,
        branches=2,
        residual_rank=4,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        index_heads=2,
        index_dim=8,
        block_size=2,
        token_budget=8,
        query_chunk_size=3,
        ngram_layer=1,
        ngram_vocab=17,
        ngram_heads=1,
        ngram_dim=32,
        mtp_steps=1,
        backend="torch",
    )


def _replace_experts(model, mesh):
    paths = [model.layers]
    if model.mtp is not None:
        paths.append([model.mtp.layer])
    for layers in paths:
        for layer in layers:
            old = layer.experts
            new = TensorParallelExperts(
                model.config.hidden_size,
                model.config.expert_dim,
                model.config.num_experts,
                model.config.top_k,
                mesh,
                num_shared_experts=1,
                device="cpu",
                dtype=old.w_gate.dtype,
                init_std=model.config.init_std,
                init_seed=31,
            )
            state = {
                "w_gate": old.w_gate.detach(),
                "w_up": old.w_up.detach(),
                "w_down": old.w_down.detach(),
                "router.weight": old.router.weight.detach(),
                "shared_gate.weight": old.shared_gate.weight.detach(),
                "shared_up.weight": old.shared_up.weight.detach(),
                "shared_down.weight": old.shared_down.weight.detach(),
                "shared_output_gate.weight": old.shared_output_gate.weight.detach(),
            }
            new.load_full_state_dict(state)
            layer.experts = new


def _local_slice(value, rank, world, dim):
    width = value.size(dim) // world
    return value.narrow(dim, rank * width, width)


def _compare_expert_gradients(reference, parallel, rank, world):
    reference_layers = list(reference.layers) + ([reference.mtp.layer] if reference.mtp else [])
    parallel_layers = list(parallel.layers) + ([parallel.mtp.layer] if parallel.mtp else [])
    for expected_layer, actual_layer in zip(reference_layers, parallel_layers, strict=True):
        expected = expected_layer.experts
        actual = actual_layer.experts
        for name, dim in (("w_gate", 2), ("w_up", 2), ("w_down", 1)):
            reference_grad = getattr(expected, name).grad
            actual_grad = getattr(actual, name).grad.to_local()
            torch.testing.assert_close(
                actual_grad,
                _local_slice(reference_grad, rank, world, dim),
                atol=3e-6,
                rtol=3e-5,
            )
        for name in (
            "router",
            "shared_gate",
            "shared_up",
            "shared_down",
            "shared_output_gate",
        ):
            reference_grad = getattr(expected, name).weight.grad
            actual_grad = getattr(actual, name).weight.grad
            torch.testing.assert_close(actual_grad, reference_grad, atol=3e-6, rtol=3e-5)


def _compare_expert_parameters(reference, parallel, rank, world):
    for ref_layer, tp_layer in zip(
        list(reference.layers) + ([reference.mtp.layer] if reference.mtp else []),
        list(parallel.layers) + ([parallel.mtp.layer] if parallel.mtp else []),
        strict=True,
    ):
        for name, dim in (("w_gate", 2), ("w_up", 2), ("w_down", 1)):
            torch.testing.assert_close(
                getattr(tp_layer.experts, name).to_local(),
                _local_slice(getattr(ref_layer.experts, name), rank, world, dim),
                atol=3e-6,
                rtol=3e-5,
            )
        for name in (
            "router",
            "shared_gate",
            "shared_up",
            "shared_down",
            "shared_output_gate",
        ):
            torch.testing.assert_close(
                getattr(tp_layer.experts, name).weight,
                getattr(ref_layer.experts, name).weight,
                atol=3e-6,
                rtol=3e-5,
            )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        _worker_model(rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_model(rank):
    mesh = init_device_mesh("cpu", (2,))
    cfg = _tiny_config()
    torch.manual_seed(101)
    reference = Qwen38(cfg)
    fresh = TensorParallelExperts(
        cfg.hidden_size,
        cfg.expert_dim,
        cfg.num_experts,
        cfg.top_k,
        mesh,
        init_seed=31,
    )
    assert torch.isfinite(fresh.w_gate.to_local()).all()
    probes = [torch.zeros(1) for _ in range(2)]
    dist.all_gather(probes, fresh.w_gate.to_local().flatten()[:1])
    assert not torch.equal(probes[0], probes[1])
    parallel = copy.deepcopy(reference)
    _replace_experts(parallel, mesh)
    for layer in list(parallel.layers) + ([parallel.mtp.layer] if parallel.mtp else []):
        expert = layer.experts
        assert expert.w_gate.to_local().shape == (cfg.num_experts, cfg.hidden_size, cfg.expert_dim // 2)
        assert torch.isfinite(expert.w_gate.to_local()).all()

    with pytest.raises(ValueError, match="hidden_dim"):
        TensorParallelExperts(cfg.hidden_size, cfg.expert_dim + 1, cfg.num_experts, cfg.top_k, mesh)

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 0, 10, 11, 12, 13]])
    targets = tokens.roll(-1, dims=1)
    reference.train()
    parallel.train()
    expected = reference(tokens, targets)
    actual = parallel(tokens, targets)
    torch.testing.assert_close(actual.loss, expected.loss, atol=4e-6, rtol=4e-5)
    torch.testing.assert_close(
        actual.auxiliary_loss, expected.auxiliary_loss, atol=4e-6, rtol=4e-5
    )
    # The replicated loss is intentionally undivided; TP collectives use the
    # standard row/column backward convention around this same scalar.
    expected.loss.backward()
    actual.loss.backward()

    reference_parameters = dict(reference.named_parameters())
    parallel_parameters = dict(parallel.named_parameters())
    for name, reference_parameter in reference_parameters.items():
        if ".experts." in name:
            continue
        actual_parameter = parallel_parameters[name]
        if reference_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            torch.testing.assert_close(
                actual_parameter.grad,
                reference_parameter.grad,
                atol=5e-6,
                rtol=5e-5,
            )
    _compare_expert_gradients(reference, parallel, rank, 2)
    assert parallel.layers[0].experts.router.weight.grad.abs().sum() > 0
    assert parallel.layers[0].experts.shared_gate.weight.grad.abs().sum() > 0

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    parallel_optimizer = torch.optim.SGD(parallel.parameters(), lr=0.03)
    reference_optimizer.step()
    parallel_optimizer.step()
    _compare_expert_parameters(reference, parallel, rank, 2)
    for name, reference_parameter in reference_parameters.items():
        if ".experts." in name:
            continue
        torch.testing.assert_close(
            parallel_parameters[name],
            reference_parameter,
            atol=5e-6,
            rtol=5e-5,
        )
    dist.barrier()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_qwen38_expert_tensor_parallel_matches_full_model(tmp_path):
    rendezvous = tmp_path / "rendezvous"
    mp.spawn(_worker, args=(str(rendezvous),), nprocs=2, join=True)

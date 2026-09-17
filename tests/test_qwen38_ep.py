"""Small expert-parallel correctness tests.

The two-rank test deliberately gives each rank different tokens.  Matching a
dense reference after gradient reduction catches the common implementation
mistakes: routing only local tokens, dropping repeated top-k copies, reducing
the replicated router twice, or failing when one destination has no tokens.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ohara.models.qwen38 import Config
from ohara.models.qwen38.expert_parallel import (
    ExpertParallelMoE,
    _expert_partitions,
    sync_gradients,
)
from ohara.models.qwen38.model import Experts


def _config():
    return Config(
        hidden_size=8,
        expert_dim=12,
        num_experts=4,
        top_k=2,
        backend="torch",
    )


def _copy_dense_grads(dense: Experts):
    gradients = {}
    for name, parameter in dense.named_parameters():
        gradients[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return gradients


@torch.no_grad()
def _make_dense_experts_trainable(dense: Experts):
    # ``GroupedMoE`` intentionally starts as a residual no-op.  These tests
    # exercise routing and gradients, so use the nonzero initialization applied
    # by ``Qwen38`` after constructing its expert blocks.
    for parameter in (dense.w_gate, dense.w_up, dense.w_down):
        torch.nn.init.normal_(parameter, std=0.05)
    for name in ("shared_gate", "shared_up", "shared_down", "shared_output_gate"):
        torch.nn.init.normal_(getattr(dense, name).weight, std=0.05)


def test_uneven_expert_partition_is_contiguous():
    starts, counts = _expert_partitions(5, 2)
    assert starts == (0, 3)
    assert counts == (3, 2)
    starts, counts = _expert_partitions(3, 4)
    assert starts == (0, 1, 2, 3)
    assert counts == (1, 1, 1, 0)


def test_single_rank_matches_dense_experts_and_optimizer_update():
    torch.manual_seed(41)
    dense = Experts(_config()).train()
    _make_dense_experts_trainable(dense)
    ep = ExpertParallelMoE.from_dense(dense, backend="torch").train()
    x = torch.randn(2, 5, 8, requires_grad=True)
    target = torch.randn_like(x)

    dense_output, dense_aux = dense(x)
    ep_output, ep_aux = ep(x)
    torch.testing.assert_close(ep_output, dense_output)
    torch.testing.assert_close(ep_aux, dense_aux)
    dense_loss = (dense_output - target).square().mean() + 0.13 * dense_aux
    ep_loss = (ep_output - target).square().mean() + 0.13 * ep_aux
    dense_loss.backward()
    ep_loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert ep.w_gate.grad.abs().sum() > 0
    assert ep.w_up.grad.abs().sum() > 0
    assert ep.w_down.grad.abs().sum() > 0
    assert ep.router.weight.grad.abs().sum() > 0

    dense_grads = _copy_dense_grads(dense)
    for name, parameter in ep.named_parameters():
        dense_name = name
        if name in {"w_gate", "w_up", "w_down"}:
            dense_grads[dense_name] = dense_grads[dense_name]
        torch.testing.assert_close(parameter.grad, dense_grads[dense_name])

    dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=2e-3)
    ep_optimizer = torch.optim.AdamW(ep.parameters(), lr=2e-3)
    dense_optimizer.step()
    ep_optimizer.step()
    for name, parameter in ep.named_parameters():
        torch.testing.assert_close(parameter, dict(dense.named_parameters())[name])


def _distributed_ep_worker(rank: int, rendezvous: str, empty_destination: bool = False):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(99)
        first = ExpertParallelMoE(8, 12, 5, 2, backend="torch")
        second = ExpertParallelMoE(8, 12, 5, 2, backend="torch")
        for module in (first, second):
            for parameter in module.replicated_parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(parameter, reference)

        torch.manual_seed(41)
        dense = Experts(_config()).train()
        _make_dense_experts_trainable(dense)
        ep = ExpertParallelMoE.from_dense(dense, backend="torch").train()
        if empty_destination:
            # Positive first feature makes every top-k pair target experts 0/1,
            # leaving rank 1's local experts empty while rank 1 still sends
            # its own tokens to rank 0.
            with torch.no_grad():
                dense.router.weight.zero_()
                dense.router.weight[0, 0] = 10.0
                dense.router.weight[1, 0] = 9.0
                ep.router.weight.copy_(dense.router.weight)

        x = torch.randn(2, 3, 8) + rank * 0.7
        target = torch.randn_like(x)
        if empty_destination:
            with torch.no_grad():
                x[..., 0].abs_().add_(1.0)
        x.requires_grad_()

        dense_output, _ = dense(x)
        ep_output, _ = ep(x)
        torch.testing.assert_close(ep_output, dense_output, atol=1e-6, rtol=1e-5)
        loss = (ep_output - target).square().mean()
        dense_loss = (dense_output - target).square().mean()
        loss.backward()
        dense_loss.backward()
        sync_gradients(ep)

        # Every dense rank sees a different local token set.  The EP shard
        # must equal the mean gradient of the corresponding dense expert.
        dense_grads = _copy_dense_grads(dense)
        for gradient in dense_grads.values():
            if gradient is not None:
                dist.all_reduce(gradient)
                gradient.div_(2)
        for name, parameter in dense.named_parameters():
            if dense_grads[name] is not None:
                parameter.grad.copy_(dense_grads[name])
        for name, parameter in ep.named_parameters():
            expected = dense_grads[name]
            if expected is None:
                assert parameter.grad is None, name
                continue
            if name in {"w_gate", "w_up", "w_down"}:
                start = ep.expert_start
                stop = start + ep.local_num_experts
                expected = expected[start:stop]
            torch.testing.assert_close(parameter.grad, expected, atol=2e-6, rtol=2e-5)
        assert x.grad is not None and x.grad.abs().sum() > 0
        assert ep.router.weight.grad is not None and ep.router.weight.grad.abs().sum() > 0
        if not empty_destination or rank == 0:
            assert ep.w_gate.grad.abs().sum() > 0
            assert ep.w_up.grad.abs().sum() > 0
            assert ep.w_down.grad.abs().sum() > 0
        else:
            assert ep.w_gate.grad is not None
            assert ep.w_up.grad is not None
            assert ep.w_down.grad is not None

        # A shared AdamW step must remain identical on both replicated copies,
        # while each local expert shard must match its dense slice.
        dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=2e-3)
        ep_optimizer = torch.optim.AdamW(ep.parameters(), lr=2e-3)
        dense_optimizer.step()
        ep_optimizer.step()
        dense_parameters = dict(dense.named_parameters())
        for name, parameter in ep.named_parameters():
            expected = dense_parameters[name]
            if name in {"w_gate", "w_up", "w_down"}:
                expected = expected[ep.expert_start:ep.expert_start + ep.local_num_experts]
            torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-5)

        if empty_destination:
            assert ep.expert_load(reset=False).tolist() == [12, 12, 0, 0]
            if rank == 1:
                assert ep.local_num_experts == 2
                assert ep.expert_counts.sum() == 0
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_ep_matches_dense_gradients_and_updates(tmp_path):
    mp.spawn(
        _distributed_ep_worker,
        args=(f"file://{tmp_path / 'ep'}",),
        nprocs=2,
        join=True,
    )


def test_two_rank_ep_handles_empty_destination_and_repeated_token_routes(tmp_path):
    mp.spawn(
        _distributed_ep_worker,
        args=(f"file://{tmp_path / 'ep-empty'}", True),
        nprocs=2,
        join=True,
    )


def test_eval_returns_zero_auxiliary_loss():
    torch.manual_seed(12)
    module = ExpertParallelMoE.from_dense(Experts(_config()).eval(), backend="torch").eval()
    _, auxiliary = module(torch.randn(2, 3, 8))
    assert auxiliary.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_grouped_local_dispatch_matches_reference():
    torch.manual_seed(7)
    cfg = _config()
    cfg.expert_dim = 16
    dense = Experts(cfg).cuda().eval()
    with torch.no_grad():
        for weight in (dense.w_gate, dense.w_up, dense.w_down, dense.shared_down.weight):
            weight.normal_(std=0.1)
    ep = ExpertParallelMoE.from_dense(dense, backend="cuda").cuda().eval()
    values = torch.randn(2, 5, cfg.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert ep._use_grouped(values)
        actual, _ = ep(values)
    reference = ExpertParallelMoE.from_dense(dense, backend="torch").cuda().eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, _ = reference(values)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=0.03)
    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    expected_input = torch.autograd.grad(expected, values, gradient, retain_graph=True)[0]
    torch.testing.assert_close(values.grad, expected_input, atol=3e-4, rtol=0.03)
    expected.backward(gradient)
    for name, parameter in ep.named_parameters():
        reference_grad = dict(reference.named_parameters())[name].grad
        torch.testing.assert_close(parameter.grad, reference_grad, atol=3e-4, rtol=0.03, msg=name)
    assert actual.abs().sum() > 0
    assert ep.w_gate.grad.abs().sum() > 0

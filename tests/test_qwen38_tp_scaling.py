import copy
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from ohara.models.qwen38 import Qwen38
from ohara.models.qwen38.distributed import clip_tensor_parallel_grad_norm, tensor_parallel_param_groups
from ohara.models.qwen38.tensor_parallel import TensorParallelExperts, tensor_parallelize
from test_qwen38_tp import _tiny_config


def _worker(rank, rendezvous, directory, device_type):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(47)
        reference = Qwen38(_tiny_config()).to(device)
        mesh = init_device_mesh(device_type, (2,))
        model = tensor_parallelize(copy.deepcopy(reference), mesh)
        optimizer = torch.optim.AdamW(tensor_parallel_param_groups(model), lr=1e-3, eps=1e-6)
        expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-6)
        tokens = torch.tensor([[1, 2, 0, 3, 4, 5, 6, 7]], device=device)
        expected_parameters = dict(reference.named_parameters())
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            expected = reference(tokens, tokens.roll(-1, 1)).loss
            actual = model(tokens, tokens.roll(-1, 1)).loss
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            expected.backward()
            actual.backward()
            for name, parameter in model.named_parameters():
                gradient = parameter.grad.full_tensor() if isinstance(parameter, DTensor) else parameter.grad
                torch.testing.assert_close(gradient, expected_parameters[name].grad, atol=3e-6, rtol=3e-4,
                                           msg=lambda message, name=name: f"{name}: {message}")
            norm = clip_tensor_parallel_grad_norm(model, 0.3, mesh)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(norm, expected_norm, atol=3e-6, rtol=3e-5)
            optimizer.step()
            expected_optimizer.step()
            for name, parameter in model.named_parameters():
                value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
                torch.testing.assert_close(value, expected_parameters[name], atol=3e-6, rtol=3e-4)
            if step == 0:
                weights, moments = get_state_dict(model, optimizer)
                dcp.save({"model": weights, "optimizer": moments}, checkpoint_id=directory)
                model = tensor_parallelize(Qwen38(_tiny_config()).to(device), mesh)
                optimizer = torch.optim.AdamW(tensor_parallel_param_groups(model), lr=1e-3, eps=1e-6)
                weights, moments = get_state_dict(model, optimizer)
                state = {"model": weights, "optimizer": moments}
                dcp.load(state, checkpoint_id=directory)
                set_state_dict(model, optimizer, model_state_dict=state["model"],
                               optim_state_dict=state["optimizer"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_tp_full_model_adam_clipping_and_checkpoint(tmp_path, device_type):
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'tp'}", str(tmp_path / "checkpoint"), device_type), nprocs=2)


def _bf16_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=rendezvous,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(83)
        cfg = _tiny_config()
        reference = Qwen38(cfg).layers[0].experts.cuda(rank)
        with torch.no_grad():
            for weight in (reference.w_gate, reference.w_up, reference.w_down):
                weight.normal_(std=0.15)
        mesh = init_device_mesh("cuda", (2,))
        parallel = TensorParallelExperts(cfg.hidden_size, cfg.expert_dim, cfg.num_experts,
                                         cfg.top_k, mesh, device=torch.device("cuda", rank))
        parallel.load_full_state_dict(reference.state_dict())
        x = torch.randn(2, 9, cfg.hidden_size, device=torch.device("cuda", rank),
                        dtype=torch.bfloat16, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected, aux_a = reference(x)
            with patch.object(parallel, "_dispatch_grouped", wraps=parallel._dispatch_grouped) as grouped:
                actual, aux_b = parallel(other)
                assert grouped.call_count == 1
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
        (expected.float().square().sum() + 0.01 * aux_a).backward()
        (actual.float().square().sum() + 0.01 * aux_b).backward()
        _assert_bf16_gradient(other.grad, x.grad)
        parameters = dict(reference.named_parameters())
        for name, parameter in parallel.named_parameters():
            grad = parameter.grad.full_tensor() if isinstance(parameter, DTensor) else parameter.grad
            _assert_bf16_gradient(grad, parameters[name].grad, name)
        assert parallel.w_gate.grad.to_local().abs().sum() > 0
        assert parallel.router.weight.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group()


def _assert_bf16_gradient(actual, expected, name="input"):
    # TP changes BF16 accumulation order. Relative L2 error avoids division by
    # nearly cancelled individual entries; FP32 tests above use tight elementwise checks.
    error = (actual.float() - expected.float()).norm()
    bound = 0.03 * expected.float().norm() + 1e-5
    assert error <= bound, f"{name}: gradient error {error.item()} exceeds {bound.item()}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_tp_native_bf16_grouped_forward_backward(tmp_path):
    mp.spawn(_bf16_worker, args=(f"file://{tmp_path / 'bf16'}",), nprocs=2)

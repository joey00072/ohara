import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.context_parallel import ContextParallel, synchronize_gradients
from ohara.models.qwen38.ops import delta_reference


def _worker(rank, rendezvous, device_type):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(13)
        cfg = Config(hidden_size=32, head_dim=8, rotary_dim=8, linear_key_dim=8,
                     linear_value_dim=8, index_dim=8, ngram_dim=32, residual_rank=4,
                     expert_dim=16, token_budget=8, query_chunk_size=3,
                     backend="torch", mtp_steps=2)
        reference = Qwen38(cfg).to(device)
        sharded = copy.deepcopy(reference)
        context = ContextParallel(dist.group.WORLD)
        # EOS on either side of a rank boundary exercises n-gram history resets.
        tokens = torch.tensor([[1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 0, 12, 13, 14, 15, 16]], device=device)
        targets = (tokens + 1) % cfg.vocab_size
        targets[:, 2] = -1
        local = slice(rank * 8, (rank + 1) * 8)
        reference.eval()
        sharded.eval()
        expected = reference(tokens).logits[:, local]
        actual = sharded(tokens[:, local], context=context).logits
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for model in (reference, sharded):
            model.train()
        a = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        b = torch.optim.AdamW(sharded.parameters(), lr=1e-3)
        # Run both normal and activation-checkpointed passes through MTP.
        for checkpointed in (False, True):
            reference.config.activation_checkpointing = checkpointed
            sharded.config.activation_checkpointing = checkpointed
            a.zero_grad(set_to_none=True)
            b.zero_grad(set_to_none=True)
            expected = reference(tokens, targets).loss
            actual = sharded(tokens[:, local], targets[:, local], context=context).loss
            mean = actual.detach().clone()
            dist.all_reduce(mean)
            torch.testing.assert_close(mean / 2, expected, atol=2e-6, rtol=2e-5)
            expected.backward()
            actual.backward()
            synchronize_gradients(sharded, context)
            for (name, left), (_, right) in zip(reference.named_parameters(), sharded.named_parameters()):
                torch.testing.assert_close(right.grad, left.grad, atol=3e-6, rtol=3e-4, msg=lambda message, name=name: f"{name}: {message}")
            a.step()
            b.step()
            for left, right in zip(reference.parameters(), sharded.parameters()):
                torch.testing.assert_close(right, left, atol=3e-6, rtol=3e-4)
    finally:
        dist.destroy_process_group()


def test_context_parallel_model_loss_gradients_and_updates(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'cp'}", "cpu"), nprocs=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_context_parallel_cuda_collectives(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'cp-cuda'}", "cuda"), nprocs=2)


def _delta_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=rendezvous,
                            timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(19)
        device = torch.device("cuda", rank)
        q, k, v = [torch.randn(1, 128, 2, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(3)]
        # Slow decay makes the boundary state contribution large enough to test.
        g = (-0.01 * torch.rand(1, 128, 2, device=device)).requires_grad_()
        beta = torch.rand(1, 128, 2, device=device, requires_grad=True)
        inputs = (q, k, v, g, beta)
        expected = delta_reference(*inputs)[0]
        expected_grads = torch.autograd.grad(expected.float().square().sum(), inputs)
        local = [x[:, rank * 64:(rank + 1) * 64].detach().clone().requires_grad_() for x in inputs]
        actual = ContextParallel(dist.group.WORLD).delta(*local, "cuda")
        torch.testing.assert_close(actual, expected[:, rank * 64:(rank + 1) * 64], atol=0.02, rtol=0.03)
        actual_grads = torch.autograd.grad(actual.float().square().sum(), local)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad[:, rank * 64:(rank + 1) * 64], atol=0.05, rtol=0.05)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs and FLA")
def test_fla_context_state_and_backward_match_reference(tmp_path):
    mp.spawn(_delta_worker, args=(f"file://{tmp_path / 'fla-cp'}",), nprocs=2)

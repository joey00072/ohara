"""Whole-model EP parity, including explicit rank-local checkpoint restore."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.distributed import clip_expert_parallel_grad_norm
from ohara.models.qwen38.expert_parallel import (
    ExpertParallelMoE,
    expert_parallelize,
    sync_model_gradients,
)
from examples.train_qwen38 import TrainingState


def _config():
    return Config(
        hidden_size=32, head_dim=8, rotary_dim=8, linear_key_dim=8,
        linear_value_dim=8, index_dim=8, ngram_dim=32, residual_rank=4,
        expert_dim=16, num_experts=5, token_budget=8, query_chunk_size=3,
        backend="torch", mtp_steps=2, activation_checkpointing=False,
    )


def _expected(model, name, value):
    module_path, _, parameter_name = name.rpartition(".")
    owner = model.get_submodule(module_path)
    if isinstance(owner, ExpertParallelMoE) and parameter_name in {"w_gate", "w_up", "w_down"}:
        return value[owner.expert_start:owner.expert_start + owner.local_num_experts]
    return value


def _worker(rank, rendezvous, directory, device_type):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=2,
        init_method=rendezvous, timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(77)
        reference = Qwen38(_config()).to(device)
        parallel = expert_parallelize(copy.deepcopy(reference), backend="torch")
        optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-3, eps=1e-6)
        with pytest.raises(ValueError, match="rank-local checkpoints"):
            TrainingState(parallel, optimizer)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-6)
        tokens = torch.arange(1, 27, device=device).reshape(2, 13)
        targets = tokens.roll(-1, 1)
        for step in range(2):
            # The second pass exercises collectives inside recomputation.
            parallel.config.activation_checkpointing = bool(step)
            reference.config.activation_checkpointing = bool(step)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected = reference(tokens, targets)
            actual = parallel(tokens[rank:rank + 1], targets[rank:rank + 1])
            loss = actual.loss.detach().clone()
            auxiliary = actual.auxiliary_loss.detach().clone()
            dist.all_reduce(loss)
            dist.all_reduce(auxiliary)
            torch.testing.assert_close(loss / 2, expected.loss.detach(), atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(auxiliary / 2, expected.auxiliary_loss.detach(), atol=3e-6, rtol=3e-5)
            expected.loss.backward()
            actual.loss.backward()
            sync_model_gradients(parallel)
            reference_parameters = dict(reference.named_parameters())
            for name, parameter in parallel.named_parameters():
                target = _expected(parallel, name, reference_parameters[name].grad)
                torch.testing.assert_close(
                    parameter.grad, target, atol=4e-6, rtol=5e-4,
                    msg=lambda message, name=name, step=step: f"step {step}, {name}: {message}",
                )
            for name in ("embedding.weight", "layers.0.experts.router.weight",
                         "layers.0.experts.w_gate", "layers.0.experts.w_up", "layers.0.experts.w_down"):
                assert reference_parameters[name].grad.abs().sum() > 0, name
            norm = clip_expert_parallel_grad_norm(parallel, 0.3)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(norm, expected_norm, atol=3e-6, rtol=3e-5)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in parallel.named_parameters():
                torch.testing.assert_close(
                    parameter, _expected(parallel, name, reference_parameters[name]),
                    atol=4e-6, rtol=5e-4,
                    msg=lambda message, name=name, step=step: f"step {step}, {name}: {message}",
                )
            if step == 0:
                # Plain local EP tensors need rank-local files. Generic DCP
                # would interpret equal parameter names as replicated tensors.
                path = Path(directory) / f"rank-{rank}.pt"
                torch.save({"rank": rank, "world": 2, "model": parallel.state_dict(),
                            "optimizer": optimizer.state_dict()}, path)
                parallel = expert_parallelize(Qwen38(_config()).to(device), backend="torch")
                optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-3, eps=1e-6)
                saved = torch.load(path, map_location=device, weights_only=True)
                assert saved["rank"] == rank and saved["world"] == 2
                parallel.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_ep_model_gradients_optimizer_and_rank_checkpoint_match_reference(tmp_path, device_type):
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path), device_type), nprocs=2)

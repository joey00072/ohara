"""Small whole-model checks for FSDP and owner-routed n-gram composition."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor

from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.context_parallel import ContextParallel
from ohara.models.qwen38.distributed import build_sharded, clip_grad_norm


def _config():
    return Config(
        hidden_size=32, head_dim=8, rotary_dim=8, linear_key_dim=8,
        linear_value_dim=8, index_dim=8, ngram_dim=32, residual_rank=4,
        expert_dim=16, token_budget=8, query_chunk_size=3, backend="torch",
        activation_checkpointing=True, mtp_steps=2,
    )


def _worker(rank, rendezvous, directory, device_type, context_parallel):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=2,
        init_method=rendezvous, timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(31)
        mesh = init_device_mesh(device_type, (2,))
        rng = torch.get_rng_state()
        model = build_sharded(_config(), mesh, device, mixed_precision=False)
        if device_type == "cpu":
            assert torch.equal(torch.get_rng_state(), rng)
        initialized = model.embedding.weight.full_tensor()
        assert not torch.equal(initialized[:128], initialized[128:])
        torch.manual_seed(32)
        reference = Qwen38(_config()).to(device)
        reference_parameters = dict(reference.named_parameters())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(distribute_tensor(
                    reference_parameters[name].detach(), parameter.device_mesh, parameter.placements,
                ))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-6)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-6)
        # Separate examples preserve the router's per-microbatch auxiliary loss.
        context = ContextParallel(dist.group.WORLD) if context_parallel else None
        batches = (torch.arange(1, 33, device=device).reshape(2, 16) if context_parallel
                   else torch.arange(1, 53, device=device).reshape(4, 13))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            losses = []
            for accumulation in range(2):
                if context_parallel:
                    whole = batches[accumulation:accumulation + 1]
                    local = slice(rank * 8, (rank + 1) * 8)
                    row, labels = whole[:, local], whole.roll(-1, 1)[:, local]
                else:
                    row = batches[2 * accumulation + rank:2 * accumulation + rank + 1]
                    labels = row.roll(-1, 1)
                loss = model(row, labels, context=context).loss / 2
                loss.backward()
                losses.append(loss.detach())
            expected_loss = sum(
                reference(row[None], row.roll(-1)[None]).loss / batches.size(0) for row in batches
            )
            expected_loss.backward()
            actual_loss = torch.stack(losses).sum()
            dist.all_reduce(actual_loss)
            torch.testing.assert_close(actual_loss / 2, expected_loss.detach(), atol=2e-6, rtol=2e-6)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.grad.full_tensor(), reference_parameters[name].grad,
                    atol=2e-6, rtol=2e-4,
                    msg=lambda message, step=step, name=name: f"step {step}, {name}: {message}",
                )
            actual_norm = clip_grad_norm(model, 0.3)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(actual_norm, expected_norm, atol=2e-6, rtol=2e-5)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.full_tensor(), reference_parameters[name], atol=3e-6, rtol=3e-4,
                    msg=lambda message, step=step, name=name: f"step {step}, {name}: {message}",
                )
            if step == 0:
                model_state, optimizer_state = get_state_dict(model, optimizer)
                state = {"model": model_state, "optimizer": optimizer_state}
                dcp.save(state, checkpoint_id=directory)
                # Fresh state catches missing embedding shards or Adam moments.
                model = build_sharded(_config(), mesh, device, mixed_precision=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-6)
                model_state, optimizer_state = get_state_dict(model, optimizer)
                state = {"model": model_state, "optimizer": optimizer_state}
                dcp.load(state, checkpoint_id=directory)
                set_state_dict(model, optimizer, model_state_dict=state["model"],
                               optim_state_dict=state["optimizer"])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("context_parallel", [False, True], ids=["dp", "cp"])
def test_fsdp_ngram_gradients_optimizer_and_checkpoint_match_reference(tmp_path, device_type, context_parallel):
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'pg'}", str(tmp_path / "checkpoint"), device_type, context_parallel),
             nprocs=2)

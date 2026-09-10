"""CPU/gloo checks for actual distributed updates and checkpoint interoperability."""

import copy
import gc
from unittest.mock import patch
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ohara.models.llama import Config, Llama
from ohara.runtime import EngineConfig, OharaEngine, ParallelConfig, PrecisionConfig, PrecisionMode
from ohara.tokenbin import TokenBinDataset, write_token_bin
from ohara.trainer import Trainer


class ConstantLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, inputs):
        return self.logits.expand(*inputs.shape, 2)


class _BinTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    name_or_path = "distributed-test"

    def __len__(self):
        return 256

    def __call__(self, batch, add_special_tokens=False):
        return {"input_ids": [[ord(character) for character in text] for text in batch]}


def _run_distributed_case(rank, mode, directory):
    """Own case-local tensors and modules so they leave scope before PG teardown."""
    config = EngineConfig(
        precision=PrecisionConfig(mode=PrecisionMode.FP32),
        parallel=ParallelConfig(tp=2 if mode.startswith("tp") else 1),
    )
    engine = OharaEngine(config)
    engine.launch()
    assert engine.global_rank == rank  # No RANK environment variable required.
    torch.manual_seed(12)
    if mode == "tp_prebuilt_optimizer":
        model = Llama(Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            max_sequence_length=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            dropout=0,
            weight_tying=False,
        ))
        optimizer = torch.optim.AdamW(model.parameters())
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        heads = (
            model.layers[0].attn.num_attention_heads,
            model.layers[0].attn.num_key_value_heads,
        )
        with pytest.raises(ValueError, match="prepare the model first"):
            engine.prepare(model, optimizer)
        assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
        assert (
            model.layers[0].attn.num_attention_heads,
            model.layers[0].attn.num_key_value_heads,
        ) == heads
        assert not hasattr(model, "_ohara_tensor_parallel")
        return
    if mode in ("tp_data", "ddp_data"):
        dataset = TokenBinDataset(
            Path(directory) / "data.bin",
            max_length=2,
            shuffle=False,
            infinite=False,
        )
        loader = engine.prepare_dataloaders(DataLoader(dataset, batch_size=2))
        first = next(iter(loader))[0]
        gathered = [torch.empty_like(first) for _ in range(2)]
        dist.all_gather(gathered, first)
        assert (dataset.data_rank, dataset.data_world_size) == (
            engine.data_parallel_rank,
            engine.data_parallel_world_size,
        )
        if mode == "tp_data":
            torch.testing.assert_close(gathered[0], gathered[1])
        else:
            assert not torch.equal(gathered[0], gathered[1])
            combined = torch.cat(gathered)
            assert len({tuple(row.tolist()) for row in combined}) == len(combined)
        return
    if mode == "masked_ddp":
        torch.manual_seed(42 + rank)
        sharded = engine.prepare_dataloaders(DataLoader(
            TensorDataset(torch.arange(20)), batch_size=2, shuffle=True,
        ))
        seen = torch.cat([batch[0] for batch in sharded])
        partitions = engine.all_gather(seen)
        assert sorted(torch.cat(partitions).tolist()) == list(range(20))
        model = engine.prepare(ConstantLogits())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.zeros(1, 2 if rank == 0 else 3, dtype=torch.long)
        targets = torch.tensor([[0, -1]] if rank == 0 else [[1, 1, 1]])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
        trainer = Trainer(
            engine, model, optimizer, loader, loader, lambda _: 0.1,
            micro_batch=1, max_iters=1, eval_iters=0, save_ckpt_iters=1,
            checkpoint_path=Path(directory) / "ddp.pt", print_every=100,
        )
        trainer.train()
        assert trainer.train_tokens_seen == 5
        torch.testing.assert_close(model.module.logits, torch.tensor([-0.025, 0.025]))
        saved = engine.load(Path(directory) / "ddp.pt")
        assert len(saved["rng_states"]) == 2
        assert list(saved["model"]) == ["logits"]
        with torch.no_grad():
            model.module.logits.zero_()
        engine.load(Path(directory) / "ddp.pt", {"model": model})
        torch.testing.assert_close(model.module.logits, torch.tensor([-0.025, 0.025]))
        return

    if mode == "chunked_ddp":
        torch.manual_seed(211)
        model_config = Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            max_sequence_length=5,
            num_attention_heads=4,
            num_hidden_layers=1,
            dropout=0,
            weight_tying=True,
        )
        raw_model = Llama(model_config)
        reference = copy.deepcopy(raw_model)
        model = engine.prepare(raw_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        inputs = torch.arange(1, 21).reshape(4, 5)
        targets = inputs.roll(-1, dims=1)
        targets[0].fill_(-1)
        targets[1, 1:].fill_(-1)
        targets[2, 0] = -1
        local_inputs = inputs[2 * rank : 2 * rank + 2]
        local_targets = targets[2 * rank : 2 * rank + 2]
        train_loader = DataLoader(
            TensorDataset(local_inputs, local_targets), batch_size=1
        )
        eval_loader = DataLoader(
            TensorDataset(local_inputs[:1], local_targets[:1]), batch_size=1
        )
        trainer = Trainer(
            engine, model, optimizer, train_loader, eval_loader, lambda _: 0.01,
            micro_batch=2, max_iters=2, eval_iters=0, save_ckpt_iters=0,
            print_every=100, loss_chunk_size=3,
        )
        trainer.train()

        for _ in range(2):
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = F.cross_entropy(
                reference(inputs).reshape(-1, model_config.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
            reference_loss.backward()
            reference_optimizer.step()
        for actual, expected in zip(
            raw_model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-7)

        metrics = trainer.evaluate(trainer.val_dataloader, 1)
        gathered_loss = [None, None]
        dist.all_gather_object(gathered_loss, metrics["loss"])
        assert gathered_loss[0] == pytest.approx(gathered_loss[1])
        with torch.no_grad():
            expected_eval = F.cross_entropy(
                reference(inputs[2:3]).reshape(-1, model_config.vocab_size),
                targets[2:3].reshape(-1),
                ignore_index=-1,
            ).item()
        assert metrics["loss"] == pytest.approx(expected_eval, rel=1e-6)
        return

    if mode == "tp_nonfinite":
        model_config = Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            max_sequence_length=8, num_attention_heads=4,
            num_key_value_heads=2, num_hidden_layers=1, dropout=0,
            weight_tying=False,
        )
        model = engine.prepare(Llama(model_config))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[2, 3, 4]])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=1)
        trainer = Trainer(
            engine, model, optimizer, loader, loader, lambda _: 0.1,
            micro_batch=1, max_iters=1, eval_iters=0, save_ckpt_iters=0,
            grad_clip_norm=None, print_every=100,
        )

        real_cross_entropy = F.cross_entropy

        def local_nonfinite_cross_entropy(*args, **kwargs):
            loss = real_cross_entropy(*args, **kwargs)
            if rank == 0:
                loss = loss + torch.full_like(loss, float("nan")).detach()
            return loss

        with patch(
            "ohara.trainer.F.cross_entropy",
            side_effect=local_nonfinite_cross_entropy,
        ), patch.object(engine, "optimizer_step", wraps=engine.optimizer_step) as step:
            try:
                trainer.train()
            except RuntimeError as exc:
                assert "Non-finite" in str(exc)
            else:
                raise AssertionError("non-finite loss did not stop training")

        stepped = torch.tensor(step.call_count, dtype=torch.int32)
        gathered = [torch.zeros_like(stepped) for _ in range(2)]
        dist.all_gather(gathered, stepped)
        assert [value.item() for value in gathered] == [0, 0]
        return

    model_config = Config(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        max_sequence_length=8, num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=1, dropout=0, weight_tying=False,
        moe_num_experts=8 if mode == "moe_ddp" else 0,
    )
    raw_model = Llama(model_config)
    reference = copy.deepcopy(raw_model)
    model = torch.compile(raw_model, backend="eager") if mode == "tp_compiled" else raw_model
    model = engine.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
    inputs = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    for _ in range(2):
        logits = model(inputs)
        if mode.startswith("tp"):
            reference_logits = reference(inputs)
            torch.testing.assert_close(logits, reference_logits, atol=2e-6, rtol=2e-5)
            F.cross_entropy(reference_logits.flatten(0, 1), labels.flatten()).backward()
            reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            reference_optimizer.step()
            reference_optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        if mode.startswith("tp"):
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
                norm = engine.clip_gradients(model, optimizer, max_norm=0.1)
            assert reduce.call_count == 1
            torch.testing.assert_close(norm, reference_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if mode.startswith("tp"):
        from torch.distributed.tensor import DTensor, distribute_tensor, Shard
        from ohara.runtime.engine import _checkpoint_to_dtensor

        uneven = torch.arange(10).reshape(5, 2)
        target = distribute_tensor(torch.zeros_like(uneven), engine._tp_mesh, [Shard(0)])
        sharded = _checkpoint_to_dtensor(uneven, target)
        torch.testing.assert_close(sharded.full_tensor(), uneven)
        checkpoint = Path(directory) / f"{mode}.pt"
        with patch.object(DTensor, "full_tensor", side_effect=AssertionError("full GPU gather")):
            engine.save(checkpoint, {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "uneven": sharded,
            })
        payload = torch.load(checkpoint, weights_only=False)
        torch.testing.assert_close(payload["uneven"], uneven)
        restored = Llama(model_config)
        restored.load_state_dict(payload["model"])
        torch.testing.assert_close(restored(inputs), reference(inputs), atol=2e-6, rtol=2e-5)
        loaded = engine.load(checkpoint, {"model": model, "optimizer": optimizer})
        assert all(tensor.device.type == "cpu" for tensor in loaded["model"].values())
        # Resume must leave optimizer state usable with the sharded parameters.
        F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten()).backward()
        optimizer.step()
        F.cross_entropy(reference(inputs).flatten(0, 1), labels.flatten()).backward()
        reference_optimizer.step()
        torch.testing.assert_close(model(inputs), reference(inputs), atol=2e-6, rtol=2e-5)
    mean = engine.all_reduce(torch.tensor(rank + 1), "mean")
    assert mean.item() == 1.5


def _worker(rank, rendezvous, mode, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        _run_distributed_case(rank, mode, directory)
        # Release deferred DDP/autograd references while both ranks still have
        # a live process group, then synchronize successful cleanup.
        gc.collect()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "mode",
    [
        "tp",
        "tp_compiled",
        "masked_ddp",
        "chunked_ddp",
        "moe_ddp",
        "tp_nonfinite",
        "tp_prebuilt_optimizer",
        "tp_data",
        "ddp_data",
    ],
)
def test_distributed_updates_and_checkpoints(tmp_path, monkeypatch, mode):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    if mode in ("tp_data", "ddp_data"):
        write_token_bin(["abcdefghijklmnop"], _BinTokenizer(), tmp_path / "data.bin", log=False)
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), mode, str(tmp_path)), nprocs=2)


@pytest.mark.parametrize("dimension", ["pp", "cp", "ep"])
def test_unimplemented_parallelism_fails_before_launch(dimension):
    engine = OharaEngine(EngineConfig(parallel=ParallelConfig(**{dimension: 2})))
    with pytest.raises(ValueError, match="not implemented"):
        engine.launch()

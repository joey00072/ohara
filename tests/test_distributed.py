"""CPU/gloo checks for actual distributed updates and checkpoint interoperability."""

import copy
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
from ohara.trainer import Trainer


class ConstantLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, inputs):
        return self.logits.expand(*inputs.shape, 2)


def _worker(rank, rendezvous, mode, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        config = EngineConfig(
            precision=PrecisionConfig(mode=PrecisionMode.FP32),
            parallel=ParallelConfig(tp=2 if mode.startswith("tp") else 1),
        )
        engine = OharaEngine(config)
        engine.launch()
        assert engine.global_rank == rank  # No RANK environment variable required.
        torch.manual_seed(12)
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
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "mode", ["tp", "tp_compiled", "masked_ddp", "moe_ddp", "tp_nonfinite"]
)
def test_distributed_updates_and_checkpoints(tmp_path, monkeypatch, mode):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), mode, str(tmp_path)), nprocs=2)


@pytest.mark.parametrize("dimension", ["pp", "cp", "ep"])
def test_unimplemented_parallelism_fails_before_launch(dimension):
    engine = OharaEngine(EngineConfig(parallel=ParallelConfig(**{dimension: 2})))
    with pytest.raises(ValueError, match="not implemented"):
        engine.launch()

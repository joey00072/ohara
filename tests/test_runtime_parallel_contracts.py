from __future__ import annotations

import random
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from ohara.data_parallel import DataParallelIterableDataset
from ohara.dataset import PreTokenizedDataset, TinyShakespeareDataset
from ohara.models.llama import Config, Llama
from ohara.runtime import EngineConfig, OharaEngine, ParallelConfig, PrecisionConfig, PrecisionMode
from ohara.runtime.tensor_parallel import TensorParallelPlan, apply_tensor_parallel
from ohara.runtime.topology import ParallelTopology


def _llama(*, dropout: float = 0.0) -> Llama:
    return Llama(Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        max_sequence_length=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        dropout=dropout,
        weight_tying=False,
    ))


@pytest.mark.parametrize("method", ["prepare", "setup"])
def test_tp_combined_prepare_rejects_optimizer_before_launch_or_model_mutation(method):
    engine = OharaEngine(EngineConfig(
        precision=PrecisionConfig(mode=PrecisionMode.FP32),
        parallel=ParallelConfig(tp=2),
    ))
    model = _llama()
    optimizer = torch.optim.AdamW(model.parameters())
    parameters = tuple(model.parameters())
    devices = tuple(parameter.device for parameter in parameters)

    with pytest.raises(ValueError, match="prepare the model first"):
        getattr(engine, method)(model, optimizer)

    assert not engine.launched
    assert tuple(model.parameters()) == parameters
    assert tuple(parameter.device for parameter in model.parameters()) == devices
    assert not hasattr(model, "_ohara_tensor_parallel")


def test_tp_attention_dropout_rejects_before_parallelization_or_head_mutation():
    model = _llama(dropout=0.5)
    attention = model.layers[0].attn
    parameters = tuple(model.parameters())
    heads = (attention.num_attention_heads, attention.num_key_value_heads)

    with patch("ohara.runtime.tensor_parallel.parallelize_module") as parallelize:
        with pytest.raises(ValueError, match="zero attention dropout"):
            apply_tensor_parallel(model, object(), TensorParallelPlan.llama_default(2))

    parallelize.assert_not_called()
    assert tuple(model.parameters()) == parameters
    assert (attention.num_attention_heads, attention.num_key_value_heads) == heads
    assert not hasattr(model, "_ohara_tensor_parallel")


def test_engine_rejects_tp_dropout_before_true_bf16_conversion():
    engine = OharaEngine(EngineConfig(
        precision=PrecisionConfig(mode=PrecisionMode.BF16_TRUE),
        parallel=ParallelConfig(tp=2),
    ))
    engine._launched = True
    engine.topology = ParallelTopology(
        world_size=2,
        dp_replicate=1,
        dp_shard=1,
        tp=2,
        pp=1,
        cp=1,
        ep=1,
    )
    engine._tp_mesh = object()
    model = _llama(dropout=0.5).eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with pytest.raises(ValueError, match="zero attention dropout"):
        engine.prepare_module(model)

    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


class _Rows(DataParallelIterableDataset, IterableDataset):
    def __init__(self, rows: list[int], rank=None, world_size=None):
        self.rows = rows
        self.data_rank = rank
        self.data_world_size = world_size

    def __iter__(self):
        self._mark_iterator_started()
        rank = self.data_rank if self.data_rank is not None else 0
        world_size = self.data_world_size if self.data_world_size is not None else 1
        yield from self.rows[rank::world_size]


class _UnknownRows(IterableDataset):
    def __iter__(self):
        yield 1


def _launched_engine(*, rank: int, dp_world_size: int, tp: int) -> OharaEngine:
    engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
    engine._launched = True
    engine.topology = ParallelTopology(
        world_size=dp_world_size * tp,
        dp_replicate=1,
        dp_shard=dp_world_size,
        tp=tp,
        pp=1,
        cp=1,
        ep=1,
    )
    engine._dp_rank = rank
    engine._dp_world_size = dp_world_size
    return engine


def test_engine_binds_supported_iterable_to_data_parallel_topology():
    tp_batches = []
    for global_rank in (0, 1):
        dataset = _Rows([10, 20, 30, 40])
        engine = _launched_engine(rank=0, dp_world_size=1, tp=2)
        with patch.dict("os.environ", {"RANK": str(global_rank)}):
            loader = engine.prepare_dataloaders(DataLoader(dataset, batch_size=2))
            tp_batches.append(next(iter(loader)))
        assert (dataset.data_rank, dataset.data_world_size) == (0, 1)
    torch.testing.assert_close(tp_batches[0], tp_batches[1])

    dp_batches = []
    for rank in (0, 1):
        dataset = _Rows([10, 20, 30, 40])
        engine = _launched_engine(rank=rank, dp_world_size=2, tp=1)
        with patch.dict("os.environ", {"RANK": str(rank)}):
            loader = engine.prepare_dataloaders(DataLoader(dataset, batch_size=2))
            dp_batches.append(next(iter(loader)))
        assert (dataset.data_rank, dataset.data_world_size) == (rank, 2)
    assert dp_batches[0].tolist() == [10, 30]
    assert dp_batches[1].tolist() == [20, 40]


def test_engine_rejects_incompatible_started_and_unknown_iterable_topologies():
    engine = _launched_engine(rank=0, dp_world_size=1, tp=2)
    explicit = _Rows([1, 2], rank=1, world_size=2)
    with pytest.raises(ValueError, match="does not match engine"):
        engine.prepare_dataloaders(DataLoader(explicit))

    started = _Rows([1, 2])
    next(iter(started))
    with pytest.raises(RuntimeError, match="after dataset iteration has started"):
        engine.prepare_dataloaders(DataLoader(started))

    workers_started = _Rows([1, 2])
    active_loader = DataLoader(workers_started)
    active_loader._iterator = object()
    with pytest.raises(RuntimeError, match="DataLoader workers have started"):
        engine.prepare_dataloaders(active_loader)

    with pytest.raises(ValueError, match="requires iterable datasets"):
        engine.prepare_dataloaders(DataLoader(_UnknownRows()))


def test_pretokenized_dataset_uses_bound_data_coordinates_instead_of_global_rank():
    dataset = object.__new__(PreTokenizedDataset)
    dataset.ds = [
        {"input_ids": [10, 11, 12]},
        {"input_ids": [20, 21, 22]},
    ]
    dataset.PAD = 0
    dataset.max_length = 3
    dataset.data_rank = 0
    dataset.data_world_size = 1

    with (
        patch("ohara.dataset.dist.is_initialized", return_value=True),
        patch("ohara.dataset.dist.get_rank", return_value=1),
        patch("ohara.dataset.dist.get_world_size", return_value=2),
    ):
        inputs, _ = next(iter(dataset))

    assert inputs.tolist() == [10, 11]


def test_tiny_shakespeare_uses_distinct_flattened_rank_worker_random_streams():
    class _Worker:
        id = 1
        num_workers = 2

    dataset = object.__new__(TinyShakespeareDataset)
    dataset.seed = 10
    dataset.data_rank = 1
    dataset.data_world_size = 2
    dataset.length = 100
    dataset.max_length = 4
    dataset.data = torch.arange(100)
    expected = random.Random(13).randint(0, 95)

    with patch("ohara.dataset.get_worker_info", return_value=_Worker()):
        inputs, _ = next(iter(dataset))

    assert inputs.tolist() == list(range(expected, expected + 4))

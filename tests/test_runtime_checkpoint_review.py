from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.runtime.engine import _checkpoint_to_dtensor


def test_checkpoint_cpu_load_and_grad_scaler_roundtrip(tmp_path):
    config = EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP16_MIXED))
    engine = OharaEngine(config)
    # CPU GradScaler exercises real growth/backoff state without requiring CUDA.
    engine._scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=3)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine.backward(model(torch.ones(1, 2)).sum())
    engine.optimizer_step(optimizer)
    expected = engine._scaler.state_dict()
    assert expected["_growth_tracker"] == 1
    path = tmp_path / "checkpoint.pt"
    engine.save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    restored = OharaEngine(config)
    restored._scaler = torch.amp.GradScaler("cpu", init_scale=2)
    with patch.object(torch, "load", wraps=torch.load) as load:
        payload = restored.load(path, {"model": model, "optimizer": optimizer})
    assert load.call_args.kwargs["map_location"] == "cpu"
    assert all(t.device.type == "cpu" for t in payload["model"].values())
    assert restored._scaler.state_dict() == expected


def test_checkpoint_precision_mismatch_and_legacy_compatibility(tmp_path):
    path = tmp_path / "checkpoint.pt"
    engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.BF16_MIXED)))
    engine.save(path, {"step": 3})
    different = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
    # Inspection does not try to change runtime state.
    assert different.load(path)["step"] == 3
    with pytest.raises(ValueError, match="precision"):
        different.load(path, {})
    torch.save({"step": 4}, path)
    state = {"step": 0}
    different.load(path, state)
    assert state["step"] == 4


def test_checkpoint_engine_key_is_reserved(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        OharaEngine().save(tmp_path / "checkpoint.pt", {"_ohara_engine": {}})


def test_preinitialized_distributed_launch_selects_local_cuda_device(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with patch.object(torch.cuda, "set_device") as select:
        engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
        engine.launch()
    select.assert_called_once_with(torch.device("cuda:1"))


def test_checkpoint_slices_before_device_transfer():
    from torch.distributed.tensor import DTensor, Shard

    tensor = torch.arange(10).reshape(5, 2)
    target = SimpleNamespace(
        shape=tensor.shape, stride=tensor.stride, device=torch.device("cuda:1"),
        device_mesh=SimpleNamespace(get_coordinate=lambda: [1], size=lambda _: 2),
        placements=[Shard(0)],
    )
    transferred = []

    def transfer(local, device):
        assert device == target.device
        transferred.append(local.clone())
        return local

    # Inspect exactly what would be copied to CUDA without a CUDA requirement.
    with patch.object(torch.Tensor, "to", transfer), patch.object(DTensor, "from_local"):
        _checkpoint_to_dtensor(tensor, target)
    assert len(transferred) == 1
    torch.testing.assert_close(transferred[0], tensor[3:])

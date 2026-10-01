import sys
from types import SimpleNamespace

import pytest
import torch

from examples import train_qwen38 as training


def cli_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train_qwen38", *args])
    return training.parse_args()


@pytest.mark.parametrize("world,cp,accum", [(1, 1, 8), (4, 1, 2), (4, 2, 4)])
def test_token_budget_uses_data_parallel_degree(monkeypatch, world, cp, accum):
    args = cli_args(monkeypatch, "--seq-len", "16", "--micro-batch-size", "2",
                    "--cp-size", str(cp), "--tokens-per-step", "256")
    assert training.validate_training_args(args, world) == 256
    assert args.grad_accum_steps == accum


@pytest.mark.parametrize("budget", ["0", "-1", "127", "129"])
def test_invalid_token_budget(monkeypatch, budget):
    args = cli_args(monkeypatch, "--tokens-per-step", budget)
    with pytest.raises(ValueError, match="tokens-per-step"):
        training.validate_training_args(args, 1)


def test_accumulation_default_and_explicit_value(monkeypatch):
    args = cli_args(monkeypatch)
    assert training.validate_training_args(args, 1) == 128
    assert args.grad_accum_steps == 1
    args = cli_args(monkeypatch, "--grad-accum-steps", "3")
    assert training.validate_training_args(args, 1) == 384


def test_conflicting_batch_options(monkeypatch):
    with pytest.raises(SystemExit):
        cli_args(monkeypatch, "--grad-accum-steps", "2", "--tokens-per-step", "256")


@pytest.mark.parametrize("flag", ["learning-rate", "weight-decay", "grad-clip"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_optimizer_settings(monkeypatch, flag, value):
    args = cli_args(monkeypatch, f"--{flag}={value}")
    with pytest.raises(ValueError, match=flag.replace("-", "_")):
        training.validate_training_args(args, 1)


@pytest.mark.parametrize("loss_value", [float("inf"), float("nan")])
def test_nonfinite_loss_with_finite_gradient_does_not_update(monkeypatch, loss_value):
    class InvalidLossModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *args, **kwargs):
            return SimpleNamespace(loss=self.weight.square() + loss_value)

    cli_args(monkeypatch, "--synthetic", "--backend", "torch", "--steps", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "Qwen38", InvalidLossModel)
    updates = []
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: updates.append(True))
    with pytest.raises(RuntimeError, match="non-finite loss"):
        training.main()
    assert not updates

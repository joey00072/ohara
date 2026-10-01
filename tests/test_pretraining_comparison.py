import numpy as np
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from examples.compare_pretraining import batch_at, evaluate, learning_rate, losses, main
from ohara.models.llama import Config as LlamaConfig, Llama
from ohara.models.qwen38 import Config as QwenConfig, Qwen38


@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_comparison_ce_excludes_auxiliary_losses_and_uses_token_mean(kind):
    torch.manual_seed(42)
    if kind == "llama":
        model = Llama(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, dropout=0.0))
    else:
        model = Qwen38(QwenConfig(
            vocab_size=32, hidden_size=16, num_layers=2, attention_interval=2,
            num_heads=2, head_dim=8, rotary_dim=4, linear_key_heads=1,
            linear_value_heads=2, linear_key_dim=8, linear_value_dim=8,
            residual_rank=4, num_experts=2, top_k=1, expert_dim=8,
            index_heads=1, index_dim=8, ngram_vocab=17, ngram_dim=16, backend="torch",
        ))
    inputs = torch.randint(32, (2, 8))
    targets = torch.randint(32, (2, 8))
    model.train()
    objective, ce = losses(model, inputs, targets)
    if kind == "qwen":
        assert objective > ce  # Router/MTP terms remain in the training objective.
    objective.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        logits = output.logits if kind == "qwen" else output
        expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(ce, expected)


def test_packed_batches_cover_same_tokens_without_skips():
    tokens = np.arange(25, dtype=np.uint16)
    for offset in (0, 12):
        inputs, targets = batch_at(tokens, offset, 3, 4, "cpu")
        assert inputs.flatten().tolist() == list(range(offset, offset + 12))
        assert targets.flatten().tolist() == list(range(offset + 1, offset + 13))
    with pytest.raises(ValueError, match="budget"):
        batch_at(tokens, 13, 3, 4, "cpu")


def test_schedule_reaches_peak_and_common_final_rate():
    assert learning_rate(1, 512, 3e-4, 32) == pytest.approx(3e-4 / 32)
    assert learning_rate(32, 512, 3e-4, 32) == pytest.approx(3e-4)
    assert learning_rate(512, 512, 3e-4, 32) == pytest.approx(3e-5)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_validation_preserves_mode_on_success_and_failure(training, fail):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(8))

        def forward(self, inputs, *, targets, **kwargs):
            assert not self.training
            assert inputs.device == self.logits.device
            if fail:
                raise RuntimeError("injected validation failure")
            return F.cross_entropy(
                self.logits.expand(*inputs.shape, 8).flatten(0, 1),
                targets.flatten(), reduction="sum",
            )

    model = Model().train(training)
    tokens = np.arange(17, dtype=np.uint16) % 8
    args = SimpleNamespace(micro_batch=2, sequence_length=4)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            evaluate(model, tokens, args)
    else:
        assert evaluate(model, tokens, args) == pytest.approx(np.log(8))
    assert model.training is training


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_cli_rejects_invalid_learning_rates(value, monkeypatch):
    monkeypatch.setattr("sys.argv", ["compare_pretraining.py", "describe", "--lr", value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2

"""Exercise padded-vocabulary SFT and cleanup through the actual entrypoint."""
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from examples import train_sft
from ohara.chat import add_chat_tokens, IGNORE_INDEX, render_conversation, training_pair
from ohara.dpo import sequence_logps
from ohara.models.llama import Config, Llama


def chat_tokenizer():
    backend = Tokenizer(models.WordLevel(
        {"<unk>": 0, "hello": 1, "world": 2, "<bos>": 3}, unk_token="<unk>"
    ))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>", pad_token="<bos>")
    add_chat_tokens(tokenizer)
    return tokenizer


def setup_sft(tmp_path, monkeypatch):
    tokenizer = chat_tokenizer()
    config = Config(vocab_size=32, hidden_size=64, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=2, max_sequence_length=16)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model": Llama(config).state_dict(), "model_config": asdict(config)}, checkpoint)
    monkeypatch.setattr("sys.argv", ["train_sft.py", "--pretrained-checkpoint", str(checkpoint),
                                    "--checkpoint-path", str(tmp_path / "sft.pt"),
                                    "--precision", "fp32", "--optimizer", "adamw",
                                    "--batch-size", "1", "--grad-accum-steps", "1",
                                    "--max-iters", "1", "--eval-batches", "1",
                                    "--eval-every", "1", "--save-every", "1",
                                    "--buffer-size", "1", "--evaluate-bpb"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_sft, "load_chat_tokenizer", lambda **kwargs: tokenizer)
    tracker = MagicMock()
    monkeypatch.setattr(train_sft, "create_logger", lambda *args, **kwargs: tracker)
    conversations = [{"messages": [{"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "world"}]}]
    monkeypatch.setattr(train_sft, "build_mixture", lambda **kwargs: conversations)
    return tokenizer, tracker


def test_sft_entrypoint_evaluates_padded_vocabulary(tmp_path, monkeypatch):
    tokenizer, tracker = setup_sft(tmp_path, monkeypatch)
    trainers = []
    actual_trainer = train_sft.Trainer

    def capture_trainer(**kwargs):
        trainer = actual_trainer(**kwargs)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(train_sft, "Trainer", capture_trainer)
    train_sft.run()
    assert len(tokenizer) < trainers[0].token_bytes.numel() == 32
    assert torch.count_nonzero(trainers[0].token_bytes[len(tokenizer):]) == 0
    assert (tmp_path / "sft.pt").is_file()
    tracker.finish.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["mixture", "train"])
def test_sft_closes_tracker_and_engine_after_failure(tmp_path, monkeypatch, failure_stage):
    _, tracker = setup_sft(tmp_path, monkeypatch)
    events = []
    tracker.finish.side_effect = lambda: events.append("tracker")
    original_close = train_sft.OharaEngine.close

    def close(engine):
        events.append("engine")
        original_close(engine)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(train_sft.OharaEngine, "close", close)
    if failure_stage == "mixture":
        monkeypatch.setattr(train_sft, "build_mixture", fail)
    else:
        monkeypatch.setattr(train_sft.Trainer, "train", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        train_sft.run()
    assert events == ["tracker", "engine"]
    tracker.finish.assert_called_once()


def test_dpo_accepts_chat_rendered_ignore_index_and_masks_gradients():
    tokenizer = chat_tokenizer()
    rendered = render_conversation(tokenizer, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ])
    labels = torch.tensor(training_pair(*rendered)[1])[None]
    assert (labels == IGNORE_INDEX).any()
    logits = torch.randn(1, labels.shape[1], len(tokenizer), requires_grad=True)
    result = sequence_logps(logits, labels)
    valid = labels != IGNORE_INDEX
    expected = logits.log_softmax(-1)[valid].gather(-1, labels[valid].unsqueeze(-1)).sum()
    torch.testing.assert_close(result.sum(), expected)
    result.sum().backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0

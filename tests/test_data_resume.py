from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from ohara.data_resume import capture_input_state, restore_input_state
from ohara.dataset import StreamingTextDataset, PreTokenizedDataset
from ohara.tokenbin import TokenBinDataset, write_token_bin
from test_tokenbin import FakeTokenizer


ARGS = dict(gradient_accumulation_steps=2, data_rank=0, data_world_size=1)


def make_bin(tmp_path, **kwargs):
    path = tmp_path / "train.bin"
    if not path.exists():
        write_token_bin(["hello world" * 20], FakeTokenizer(), path, log=False)
    return TokenBinDataset(path, max_length=3, **kwargs)


def test_cursor_resumes_shuffled_bin_without_replaying_epochs(tmp_path):
    ds = make_bin(tmp_path, start_block=10000)
    loader = DataLoader(ds, batch_size=3)
    iterator = iter(loader)
    next(iterator)
    saved = capture_input_state(loader, **ARGS)
    expected = next(iterator)
    resumed = DataLoader(make_bin(tmp_path), batch_size=3)
    restore_input_state(resumed, saved, **ARGS)
    with patch("ohara.tokenbin.np.random.default_rng", wraps=__import__("numpy").random.default_rng) as rng:
        actual = next(iter(resumed))
    assert rng.call_count <= 2  # only current epoch, possibly next at boundary
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("changed", ["batch_size", "num_workers", "seed", "gradient_accumulation_steps", "data_world_size", "source", "training_recipe"])
def test_contract_rejects_changed_pipeline(tmp_path, changed):
    original = DataLoader(make_bin(tmp_path), batch_size=2)
    next(iter(original))
    saved = capture_input_state(original, **ARGS)
    kwargs = dict(ARGS)
    dataset = make_bin(tmp_path, seed=99) if changed == "seed" else make_bin(tmp_path)
    if changed == "training_recipe":
        dataset.training_recipe = {"max_iters": 900, "dropout": 0.9}
    if changed == "source":
        with dataset.bin_path.open("r+b") as handle:
            handle.write(b"xx")
    loader = DataLoader(dataset, batch_size=3 if changed == "batch_size" else 2,
                        num_workers=1 if changed == "num_workers" else 0)
    if changed in kwargs:
        kwargs[changed] += 1
    with pytest.raises(ValueError, match="contract mismatch"):
        restore_input_state(loader, saved, **kwargs)


def test_legacy_and_worker_checkpoint_cannot_claim_exact_resume(tmp_path):
    loader = DataLoader(make_bin(tmp_path), num_workers=2)
    saved = capture_input_state(loader, **ARGS)
    with pytest.raises(ValueError, match="num_workers=0"):
        restore_input_state(loader, saved, **ARGS)
    with pytest.raises(ValueError, match="legacy"):
        restore_input_state(loader, None, **ARGS)


def test_empty_bin_shards_fail_on_every_rank(tmp_path):
    for rank in (0, 1):
        dataset = make_bin(tmp_path, data_rank=rank, data_world_size=100)
        with pytest.raises(ValueError, match="rank/worker shards"):
            next(iter(dataset))
    with pytest.raises(ValueError, match="rank/worker shards"):
        make_bin(tmp_path).validate_capacity(100)


def test_legacy_empty_shard_fails():
    dataset = object.__new__(PreTokenizedDataset)
    dataset.ds = [{"input_ids": [1, 2, 3]}]
    with patch("ohara.dataset.dist.is_initialized", return_value=True), patch("ohara.dataset.dist.get_rank", return_value=0), patch("ohara.dataset.dist.get_world_size", return_value=2):
        with pytest.raises(ValueError, match="rank/worker shards"):
            next(iter(dataset))


class CountingTokenizer:
    bos_token_id = eos_token_id = pad_token_id = 0
    name_or_path = "counting"
    calls = 0

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        return [int(text)] * 7


def test_real_hf_cursor_preserves_shuffle_packing_and_epoch_without_tokenizing_history(tmp_path):
    (tmp_path / "train.jsonl").write_text("".join('{"text":"%s"}\n' % i for i in range(1, 12)))

    def make():
        return StreamingTextDataset(str(tmp_path), CountingTokenizer(), split="train", max_length=4,
                                    shuffle=True, shuffle_buffer_size=4)
    dataset = make()
    loader = DataLoader(dataset, batch_size=2)
    iterator = iter(loader)
    for _ in range(6):
        next(iterator)
    saved = capture_input_state(loader, **ARGS)
    expected = [next(iterator) for _ in range(18)]  # crosses source epochs
    resumed = make()
    new_loader = DataLoader(resumed, batch_size=2)
    restore_input_state(new_loader, saved, **ARGS)
    restored_iter = iter(new_loader)
    actual = next(restored_iter)
    assert resumed.tokenizer.calls <= 2
    assert all(torch.equal(a, b) for a, b in zip(actual, expected[0]))
    for batch in expected[1:]:
        assert all(torch.equal(a, b) for a, b in zip(next(restored_iter), batch))


def test_streaming_evaluation_snapshot_preserves_live_and_fresh_cursor(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"3"}\n{"text":"8"}\n')
    dataset = StreamingTextDataset(str(tmp_path), CountingTokenizer(), split="train", max_length=4)
    fresh = dataset.state_dict()
    next(iter(dataset))
    dataset.load_state_dict(fresh)
    assert dataset.state_dict() == fresh
    live = iter(dataset)
    next(live)
    saved = dataset.state_dict()
    evaluation = iter(dataset)
    next(evaluation)
    next(evaluation)
    dataset.load_state_dict(saved)
    resumed = iter(dataset)
    for _ in range(8):
        assert all(torch.equal(a, b) for a, b in zip(next(live), next(resumed)))


def test_finite_training_loader_cannot_claim_exact_cycle_resume(tmp_path):
    loader = DataLoader(make_bin(tmp_path, infinite=False))
    next(iter(loader))
    state = capture_input_state(loader, **ARGS)
    assert not state["resumable"]
    assert "finite" in state["reason"]

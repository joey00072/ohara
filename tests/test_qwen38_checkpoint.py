from dataclasses import asdict

import pytest
import torch
from torch.utils.data import DataLoader

from examples.train_qwen38 import resume_checkpoint, save_checkpoint
from ohara.models.qwen38 import Config, Qwen38
from ohara.tokenbin import TokenBinDataset, write_token_bin
from test_tokenbin import FakeTokenizer


def test_training_checkpoint_restores_input_rng_and_rejects_overwrite(tmp_path):
    path = tmp_path / "train.bin"
    write_token_bin(["hello world" * 20], FakeTokenizer(), path, log=False)
    cfg = Config(hidden_size=32, head_dim=8, rotary_dim=8, linear_key_dim=8,
                 linear_value_dim=8, index_dim=8, ngram_dim=32, residual_rank=4,
                 expert_dim=16, backend="torch")
    model = Qwen38(cfg)
    optimizer = torch.optim.AdamW(model.parameters())

    def loader():
        return DataLoader(TokenBinDataset(path, max_length=4), batch_size=1,
                          generator=torch.Generator().manual_seed(7))

    original = loader()
    iterator = iter(original)
    x, y = next(iterator)
    model(x, y).loss.backward()
    optimizer.step()
    recipe = {"config": asdict(cfg), "world_size": 1, "cp_size": 1, "grad_accum_steps": 1}
    checkpoint = tmp_path / "step"
    save_checkpoint(checkpoint, model, optimizer, original, 1, recipe, 0, False)
    expected_batch = next(iterator)
    expected_random = torch.rand(4)
    restored = loader()
    assert resume_checkpoint(checkpoint, model, optimizer, restored, recipe, 0) == 1
    actual_batch = next(iter(restored))
    for actual, expected in zip(actual_batch, expected_batch):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.rand(4), expected_random)
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, optimizer, restored, 2, recipe, 0, False)
    # Same-path corpus edits must not silently resume against a different stream.
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="input contract mismatch"):
        resume_checkpoint(checkpoint, model, optimizer, loader(), recipe, 0)

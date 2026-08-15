"""Checks for the SFT entrypoint's checkpoint-loading path.

``load_pretrained`` reconstructs a model from tensor shapes alone, so a change
to the Llama module layout would silently break resuming from a base
checkpoint. These tests pin that reconstruction against real checkpoints.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from ohara.models.llama import Config, Llama


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_sft)

load_pretrained = train_sft.load_pretrained


def write_checkpoint(directory: Path, config: Config) -> tuple[Path, Llama]:
    model = Llama(config)
    path = directory / "base.pt"
    torch.save({"model": model.state_dict(), "idx": 10}, path)
    return path, model


class LoadPretrainedTests(unittest.TestCase):
    def base_config(self, **overrides):
        options = {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_sequence_length": 96,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "dropout": 0.0,
        }
        options.update(overrides)
        return Config(**options)

    def test_recovers_the_architecture_and_weights(self):
        config = self.base_config()
        with tempfile.TemporaryDirectory() as directory:
            path, original = write_checkpoint(Path(directory), config)
            model, recovered, pretrained_vocab = load_pretrained(path, 128, None)

        self.assertEqual(pretrained_vocab, 128)
        self.assertEqual(recovered.num_hidden_layers, 3)
        self.assertEqual(recovered.hidden_size, 64)
        self.assertEqual(recovered.intermediate_size, 128)
        self.assertEqual(recovered.num_attention_heads, 2)
        self.assertEqual(recovered.max_sequence_length, 96)
        torch.testing.assert_close(
            model.token_emb.weight, original.token_emb.weight
        )
        torch.testing.assert_close(
            model.layers[2].ff.down.weight, original.layers[2].ff.down.weight
        )

    def test_grows_the_vocabulary_for_chat_tokens(self):
        config = self.base_config()
        with tempfile.TemporaryDirectory() as directory:
            path, original = write_checkpoint(Path(directory), config)
            model, _, pretrained_vocab = load_pretrained(path, 136, None)

        self.assertEqual(pretrained_vocab, 128)
        self.assertEqual(model.token_emb.weight.shape[0], 136)
        self.assertEqual(model.vocab_proj.weight.shape[0], 136)
        # Pretrained rows must survive the resize untouched.
        torch.testing.assert_close(
            model.token_emb.weight[:128], original.token_emb.weight
        )
        logits = model(torch.tensor([[135, 4, 7]]))
        self.assertEqual(logits.shape, (1, 3, 136))

    def test_preserves_weight_tying(self):
        config = self.base_config(weight_tying=True)
        with tempfile.TemporaryDirectory() as directory:
            path, _ = write_checkpoint(Path(directory), config)
            model, recovered, _ = load_pretrained(path, 128, None)

        self.assertTrue(recovered.weight_tying)
        self.assertIs(model.vocab_proj.weight, model.token_emb.weight)

    def test_recovers_grouped_query_attention(self):
        config = self.base_config(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=2
        )
        with tempfile.TemporaryDirectory() as directory:
            path, _ = write_checkpoint(Path(directory), config)
            _, recovered, _ = load_pretrained(path, 128, None)

        self.assertEqual(recovered.num_attention_heads, 4)
        self.assertEqual(recovered.num_key_value_heads, 2)

    def test_shrinking_the_context_window_is_allowed(self):
        config = self.base_config(max_sequence_length=96)
        with tempfile.TemporaryDirectory() as directory:
            path, _ = write_checkpoint(Path(directory), config)
            model, recovered, _ = load_pretrained(path, 128, 48)

        self.assertEqual(recovered.max_sequence_length, 48)
        # The rotary buffer must be rebuilt to the new length, not carried over.
        self.assertEqual(model.freq_cos.shape[0], 96)

    def test_growing_the_context_window_is_rejected(self):
        config = self.base_config(max_sequence_length=96)
        with tempfile.TemporaryDirectory() as directory:
            path, _ = write_checkpoint(Path(directory), config)
            with self.assertRaises(ValueError):
                load_pretrained(path, 128, 512)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_pretrained(Path("/nonexistent/base.pt"), 128, None)

    def test_handles_compiled_and_ddp_key_prefixes(self):
        config = self.base_config()
        with tempfile.TemporaryDirectory() as directory:
            model = Llama(config)
            prefixed = {
                f"_orig_mod.{key}": value for key, value in model.state_dict().items()
            }
            path = Path(directory) / "base.pt"
            torch.save({"model": prefixed}, path)
            loaded, recovered, _ = load_pretrained(path, 128, None)

        self.assertEqual(recovered.num_hidden_layers, 3)
        torch.testing.assert_close(loaded.token_emb.weight, model.token_emb.weight)


if __name__ == "__main__":
    unittest.main()

"""Checks for the pre-tokenized corpus path.

A bug here is expensive and quiet: the model would train on wrong ids rather
than crash. So these pin the round trip, the shard partitioning, and the guards
that stop a bin being paired with the wrong vocabulary.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ohara.tokenbin import (
    TOKEN_DTYPE,
    TokenBinDataset,
    read_token_bin_metadata,
    write_token_bin,
)


class FakeTokenizer:
    """Character-level tokenizer exposing the slice of the HF API we use."""

    def __init__(self, vocab_size=200):
        self._vocab = {chr(code): code - 31 for code in range(32, 127)}
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.name_or_path = "fake"
        self._vocab_size = vocab_size

    def __len__(self):
        return self._vocab_size

    def __call__(self, batch, add_special_tokens=False):
        return {"input_ids": [[self._vocab[c] for c in text if c in self._vocab] for text in batch]}


class WriteTokenBinTests(unittest.TestCase):
    def test_round_trips_documents(self):
        tokenizer = FakeTokenizer()
        documents = ["hello", "world!", "a"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            metadata = write_token_bin(documents, tokenizer, path, batch_size=2, log=False)

            self.assertEqual(metadata["documents"], 3)
            # One BOS per document plus its characters.
            self.assertEqual(metadata["tokens"], (1 + 5) + (1 + 6) + (1 + 1))
            self.assertEqual(metadata["dtype"], "uint16")

            tokens = np.fromfile(path, dtype=TOKEN_DTYPE)
            self.assertEqual(tokens.size, metadata["tokens"])
            self.assertEqual(int(tokens[0]), tokenizer.bos_token_id)
            expected = tokenizer([documents[0]])["input_ids"][0]
            self.assertEqual(tokens[1:6].tolist(), expected)

    def test_writes_sidecar_metadata(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            write_token_bin(["abc"], tokenizer, path, log=False)
            sidecar = json.loads((Path(directory) / "train.json").read_text())
            self.assertEqual(sidecar["vocab_size"], len(tokenizer))
            self.assertEqual(sidecar["tokenizer"], "fake")
            self.assertEqual(read_token_bin_metadata(path), sidecar)

    def test_skips_empty_and_non_string_documents(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            metadata = write_token_bin(["ok", "", None, "yes"], tokenizer, path, log=False)
            self.assertEqual(metadata["documents"], 2)

    def test_empty_stream_raises(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                write_token_bin([], tokenizer, Path(directory) / "train.bin", log=False)

    def test_rejects_vocabulary_too_large_for_dtype(self):
        tokenizer = FakeTokenizer(vocab_size=70_000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_token_bin(["a"], tokenizer, Path(directory) / "train.bin", log=False)

    def test_leaves_no_temporary_file(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            write_token_bin(["abc"], tokenizer, path, log=False)
            self.assertEqual(
                [p.name for p in Path(directory).iterdir() if p.name.startswith(".")], []
            )


def build_bin(directory: Path, tokenizer, documents, name="train.bin"):
    path = directory / name
    write_token_bin(documents, tokenizer, path, batch_size=8, log=False)
    return path


class TokenBinDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.documents = [f"document number {index} " * 4 for index in range(200)]

    def test_yields_shifted_blocks_of_the_right_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            dataset = TokenBinDataset(path, max_length=32, infinite=False, shuffle=False)
            inputs, targets = next(iter(dataset))
            self.assertEqual(inputs.shape, (32,))
            self.assertEqual(targets.shape, (32,))
            self.assertEqual(inputs.dtype, torch.long)
            # targets is inputs shifted by one within the same block.
            self.assertTrue(torch.equal(inputs[1:], targets[:-1]))

    def test_block_contents_match_the_underlying_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            raw = np.fromfile(path, dtype=TOKEN_DTYPE).astype(np.int64)
            dataset = TokenBinDataset(path, max_length=16, infinite=False, shuffle=False)
            inputs, targets = next(iter(dataset))
            self.assertEqual(inputs.tolist(), raw[:16].tolist())
            self.assertEqual(targets.tolist(), raw[1:17].tolist())

    def test_finite_epoch_covers_every_block_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            dataset = TokenBinDataset(path, max_length=32, infinite=False, shuffle=False)
            self.assertEqual(len(list(dataset)), dataset.num_blocks)

    def test_shards_partition_blocks_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            common = {"max_length": 32, "infinite": False, "shuffle": False}
            shards = [
                list(TokenBinDataset(path, data_rank=rank, data_world_size=4, **common))
                for rank in range(4)
            ]
            counts = [len(shard) for shard in shards]
            self.assertEqual(sum(counts), TokenBinDataset(path, **common).num_blocks)
            # Every block appears in exactly one shard.
            seen = [tuple(inputs.tolist()) for shard in shards for inputs, _ in shard]
            self.assertEqual(len(seen), len(set(seen)))

    def test_shuffle_changes_order_but_not_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            common = {"max_length": 32, "infinite": False}
            ordered = [
                tuple(x.tolist()) for x, _ in TokenBinDataset(path, shuffle=False, **common)
            ]
            shuffled = [
                tuple(x.tolist())
                for x, _ in TokenBinDataset(path, shuffle=True, seed=7, **common)
            ]
            self.assertNotEqual(ordered, shuffled)
            self.assertEqual(sorted(ordered), sorted(shuffled))

    def test_infinite_dataset_keeps_yielding_past_one_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            dataset = TokenBinDataset(path, max_length=32, infinite=True, shuffle=False)
            iterator = iter(dataset)
            wanted = dataset.num_blocks + 5
            self.assertEqual(len([next(iterator) for _ in range(wanted)]), wanted)

    def test_start_block_skips_consumed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            common = {"max_length": 32, "infinite": False, "shuffle": False}
            full = list(TokenBinDataset(path, **common))
            resumed = list(TokenBinDataset(path, start_block=3, **common))
            self.assertEqual(len(resumed), len(full) - 3)
            self.assertTrue(torch.equal(resumed[0][0], full[3][0]))

    def test_vocabulary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            with self.assertRaises(ValueError):
                TokenBinDataset(path, max_length=32, tokenizer=FakeTokenizer(vocab_size=999))

    def test_matching_vocabulary_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, self.documents)
            dataset = TokenBinDataset(path, max_length=32, tokenizer=self.tokenizer)
            self.assertGreater(dataset.num_blocks, 0)

    def test_missing_bin_and_sidecar_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                TokenBinDataset(Path(directory) / "absent.bin", max_length=32)
            orphan = Path(directory) / "orphan.bin"
            orphan.write_bytes(b"\x00\x00")
            with self.assertRaises(FileNotFoundError):
                TokenBinDataset(orphan, max_length=32)

    def test_corpus_smaller_than_one_block_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_bin(Path(directory), self.tokenizer, ["tiny"])
            with self.assertRaises(ValueError):
                TokenBinDataset(path, max_length=4096)


if __name__ == "__main__":
    unittest.main()

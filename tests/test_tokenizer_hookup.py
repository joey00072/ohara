import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from ohara.dataset import (
    PreTokenizedDataset,
    StreamingTextDataset,
    get_tokenizer as dataset_get_tokenizer,
)
from ohara.pretokenize import DatasetPreprocessor
from ohara.tokenizer import TokenizerLoadResult


class DummyTokenizer:
    def __init__(self, name_or_path: str = "dummy"):
        self.name_or_path = name_or_path
        self.padding_side = None
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<eos>"
        self.bos_token_id = 1

    def __len__(self):
        return 32

    @property
    def vocab_size(self):
        return 32

    def encode(self, text, add_special_tokens=True):
        return [3, 4, 5]

    def __call__(self, text, max_length=None, truncation=False):
        return {"input_ids": [1, 2, 3]}

    def apply_chat_template(self, rows):
        return [1, 2, 3]


class TokenizerHookupTests(unittest.TestCase):
    def test_dataset_get_tokenizer_uses_hf_priority_loader(self):
        dummy = DummyTokenizer("hf")
        with patch(
            "ohara.dataset.load_tokenizer",
            return_value=TokenizerLoadResult(tokenizer=dummy, source="hf", identifier="hf"),
        ) as mocked:
            tok = dataset_get_tokenizer("EleutherAI/gpt-neo-125m")

        self.assertIs(tok, dummy)
        self.assertEqual(tok.padding_side, "right")
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["hf_name"], "EleutherAI/gpt-neo-125m")
        self.assertTrue(kwargs["prefer_hf"])

    def test_pretokenized_dataset_accepts_tokenizer_string(self):
        dummy = DummyTokenizer("hf")
        rows = [{"input_ids": [6, 7, 8, 9]}]
        with patch(
            "ohara.dataset.load_tokenizer",
            return_value=TokenizerLoadResult(tokenizer=dummy, source="hf", identifier="hf"),
        ):
            with patch("ohara.dataset.load_from_disk", return_value=rows):
                ds = PreTokenizedDataset(
                    dataset_name="x/y",
                    tokenizer="EleutherAI/gpt-neo-125m",
                    split="train",
                    max_length=4,
                    hf=False,
                )
                x, y = next(iter(ds))

        self.assertEqual(x.shape[0], 4)
        self.assertEqual(y.shape[0], 4)

    def test_dataset_preprocessor_uses_new_get_tokenizer(self):
        dummy = DummyTokenizer("hf")
        with patch("ohara.pretokenize.get_tokenizer", return_value=dummy) as mocked:
            proc = DatasetPreprocessor(
                dataset_name="JeanKaddour/minipile",
                tokenizer_name="EleutherAI/gpt-neo-125m",
                splits=["train"],
            )

        self.assertIs(proc.tokenizer, dummy)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["hf_name"], "EleutherAI/gpt-neo-125m")
        self.assertTrue(kwargs["prefer_hf"])

    def test_streaming_text_dataset_packs_shifted_tokens(self):
        dummy = DummyTokenizer()
        dataset = StreamingTextDataset(
            dataset_name="x/y",
            tokenizer=dummy,
            split="train",
            max_length=3,
        )
        with patch.object(dataset, "_load_stream", return_value=[{"text": "hello"}]):
            x, y = next(iter(dataset))

        self.assertEqual(x.tolist(), [1, 3, 4])
        self.assertEqual(y.tolist(), [3, 4, 5])

    def test_streaming_text_dataset_honors_explicit_data_shard(self):
        dummy = DummyTokenizer()
        dataset = StreamingTextDataset(
            dataset_name="x/y",
            tokenizer=dummy,
            split="train",
            max_length=3,
            data_rank=1,
            data_world_size=2,
        )
        rows = [{"wrong_column": "rank zero"}, {"text": "rank one"}]
        with patch.object(dataset, "_load_stream", return_value=rows):
            x, y = next(iter(dataset))

        self.assertEqual(x.tolist(), [1, 3, 4])
        self.assertEqual(y.tolist(), [3, 4, 5])

    def test_streaming_text_dataset_skips_consumed_blocks_for_resume(self):
        dummy = DummyTokenizer()
        dataset = StreamingTextDataset(
            dataset_name="x/y",
            tokenizer=dummy,
            split="train",
            max_length=3,
            start_block=1,
        )
        rows = [{"text": "first"}, {"text": "second"}]
        with (
            patch.object(dataset, "_load_stream", return_value=rows),
            patch.object(dummy, "encode", side_effect=[[3, 4, 5], [6, 7, 8]]),
        ):
            x, y = next(iter(dataset))

        self.assertEqual(x.tolist(), [1, 6, 7])
        self.assertEqual(y.tolist(), [6, 7, 8])

    def test_streaming_text_dataset_resolves_local_split_files(self):
        dummy = DummyTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            local_file = Path(directory, "train.jsonl")
            local_file.write_text('{"text":"hello"}\n', encoding="utf-8")
            dataset = StreamingTextDataset(
                dataset_name=directory,
                tokenizer=dummy,
                split="train",
                max_length=3,
            )
            with patch(
                "ohara.dataset.load_dataset",
                return_value=[{"text": "hello"}],
            ) as mocked:
                x, y = next(iter(dataset))

        self.assertEqual(x.tolist(), [1, 3, 4])
        self.assertEqual(y.tolist(), [3, 4, 5])
        args, kwargs = mocked.call_args
        self.assertEqual(args[0], "json")
        self.assertEqual(kwargs["split"], "train")
        self.assertEqual(kwargs["data_files"]["train"], [str(local_file)])


if __name__ == "__main__":
    unittest.main()

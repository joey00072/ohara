import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "prepare_scaling_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_scaling_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_scaling_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_scaling_data)

DEFAULT_TOKENIZER = prepare_scaling_data.DEFAULT_TOKENIZER
NANOCHAT_CLIMBMIX_VALIDATION_SHARD = (
    prepare_scaling_data.NANOCHAT_CLIMBMIX_VALIDATION_SHARD
)
nanochat_climbmix_files = prepare_scaling_data.nanochat_climbmix_files
parse_args = prepare_scaling_data.parse_args
stage_split = prepare_scaling_data.stage_split


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return text.split()


class _BatchTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        assert kwargs == {
            "add_special_tokens": False,
            "return_attention_mask": False,
            "return_token_type_ids": False,
        }
        return {"input_ids": [text.split() for text in texts]}


class ScalingCorpusPreparationTests(unittest.TestCase):
    def test_defaults_select_nanochat_climbmix(self):
        with patch("sys.argv", ["prepare_scaling_data.py"]):
            args = parse_args()

        self.assertIsNone(args.dataset)
        self.assertEqual(args.climbmix_train_shards, 10)
        self.assertEqual(args.tokenizer, DEFAULT_TOKENIZER)

    def test_climbmix_reserves_nanochat_validation_shard(self):
        train, validation = nanochat_climbmix_files(3)

        self.assertEqual(len(train), 3)
        self.assertTrue(train[0].endswith("shard_00000.parquet"))
        self.assertTrue(train[-1].endswith("shard_00002.parquet"))
        self.assertTrue(
            validation.endswith(
                f"shard_{NANOCHAT_CLIMBMIX_VALIDATION_SHARD:05d}.parquet"
            )
        )
        self.assertNotIn(validation, train)

    def test_stage_split_accepts_multiple_remote_parquet_shards(self):
        rows = [{"text": "one two"}, {"text": "three"}]
        sources = [
            "https://example.test/shard_00000.parquet",
            "https://example.test/shard_00001.parquet",
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(
                prepare_scaling_data, "load_dataset", return_value=rows
            ) as load_dataset:
                result = stage_split(
                    dataset_name=None,
                    dataset_config=None,
                    source_files=sources,
                    source_split="train",
                    output_split="train",
                    text_column="text",
                    max_documents=2,
                    output_dir=output_dir,
                    force=False,
                    skip_existing=False,
                    tokenizer=_Tokenizer(),
                )

            self.assertEqual(result, (2, len("one two") + len("three"), 5))
            load_dataset.assert_called_once_with(
                path="parquet",
                data_files={"train": sources},
                split="train",
                streaming=True,
            )
            staged = [
                json.loads(line)
                for line in output_dir.joinpath("train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(staged, rows)
            stats = json.loads(output_dir.joinpath("stats.train.json").read_text())
            self.assertEqual(stats["tokens"], 5)

    def test_climbmix_shard_count_is_bounded(self):
        for count in (0, NANOCHAT_CLIMBMIX_VALIDATION_SHARD + 1):
            with self.assertRaises(ValueError):
                nanochat_climbmix_files(count)

    def test_token_counting_is_batched_and_includes_document_boundaries(self):
        rows = [{"text": "one two"}, {"text": "three"}, {"text": "four five six"}]
        tokenizer = _BatchTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(prepare_scaling_data, "load_dataset", return_value=rows):
                result = stage_split(
                    dataset_name="example",
                    dataset_config=None,
                    source_files=None,
                    source_split="train",
                    output_split="train",
                    text_column="text",
                    max_documents=3,
                    output_dir=Path(directory),
                    force=False,
                    skip_existing=False,
                    tokenizer=tokenizer,
                    tokenizer_batch_size=2,
                )

        self.assertEqual(result, (3, len("one twothreefour five six"), 9))
        self.assertEqual(tokenizer.calls, 2)


if __name__ == "__main__":
    unittest.main()

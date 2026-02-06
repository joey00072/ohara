import unittest
from unittest.mock import patch

from ohara.dataset import PreTokenizedDataset, get_tokenizer as dataset_get_tokenizer
from ohara.pretokenize import DatasetPreprocessor
from ohara.tokenizer import TokenizerLoadResult


class DummyTokenizer:
    def __init__(self, name_or_path: str = "dummy"):
        self.name_or_path = name_or_path
        self.padding_side = None
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<eos>"

    def __len__(self):
        return 32

    @property
    def vocab_size(self):
        return 32

    def encode(self, text):
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


if __name__ == "__main__":
    unittest.main()

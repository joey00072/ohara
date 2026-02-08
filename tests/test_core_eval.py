import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from ohara.core_eval import evaluate_core_from_bundle, evaluate_task


class DummyTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) + 3 for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        chars = []
        for token_id in ids:
            if token_id >= 3:
                chars.append(chr(token_id - 3))
        return "".join(chars)


class NextTokenOracle(torch.nn.Module):
    def __init__(self, vocab_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = 4096

    def forward(self, input_ids):
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        bsz, seq_len = input_ids.shape
        logits = torch.full(
            (bsz, seq_len, self.vocab_size),
            -30.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits.scatter_(2, target_ids.unsqueeze(-1), 30.0)
        return logits


class CoreEvalTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = NextTokenOracle()
        self.tokenizer = DummyTokenizer()

    def test_evaluate_task_multiple_choice(self):
        data = [
            {"query": "2+2=", "choices": ["4", "5"], "gold": 0},
            {"query": "1+1=", "choices": ["2", "3"], "gold": 0},
        ]
        task_meta = {
            "task_type": "multiple_choice",
            "num_fewshot": 0,
            "continuation_delimiter": " ",
        }
        score = evaluate_task(self.model, self.tokenizer, data, self.device, task_meta)
        self.assertGreaterEqual(score, 0.99)

    def test_evaluate_task_schema(self):
        data = [
            {"context_options": ["The sky is blue", "The sky is green"], "continuation": ".", "gold": 0},
            {"context_options": ["Water is wet", "Water is dry"], "continuation": ".", "gold": 0},
        ]
        task_meta = {
            "task_type": "schema",
            "num_fewshot": 0,
            "continuation_delimiter": " ",
        }
        score = evaluate_task(self.model, self.tokenizer, data, self.device, task_meta)
        self.assertGreaterEqual(score, 0.99)

    def test_evaluate_task_language_modeling(self):
        data = [
            {"context": "Hello", "continuation": " world"},
            {"context": "Good", "continuation": " day"},
        ]
        task_meta = {
            "task_type": "language_modeling",
            "num_fewshot": 0,
            "continuation_delimiter": " ",
        }
        score = evaluate_task(self.model, self.tokenizer, data, self.device, task_meta)
        self.assertGreaterEqual(score, 0.99)

    def test_evaluate_core_from_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp).joinpath("eval_bundle")
            data_dir = bundle.joinpath("eval_data")
            data_dir.mkdir(parents=True, exist_ok=True)

            tasks = [
                {
                    "label": "tiny_mc",
                    "icl_task_type": "multiple_choice",
                    "dataset_uri": "tiny_mc.jsonl",
                    "num_fewshot": [0],
                    "continuation_delimiter": " ",
                },
                {
                    "label": "tiny_schema",
                    "icl_task_type": "schema",
                    "dataset_uri": "tiny_schema.jsonl",
                    "num_fewshot": [0],
                    "continuation_delimiter": " ",
                },
                {
                    "label": "tiny_lm",
                    "icl_task_type": "language_modeling",
                    "dataset_uri": "tiny_lm.jsonl",
                    "num_fewshot": [0],
                    "continuation_delimiter": " ",
                },
            ]

            with open(bundle.joinpath("core.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump({"icl_tasks": tasks}, f)

            with open(bundle.joinpath("eval_meta_data.csv"), "w", encoding="utf-8") as f:
                f.write("Eval Task,Random baseline\n")
                for task in tasks:
                    f.write(f"{task['label']},50\n")

            mc_rows = [
                {"query": "2+2=", "choices": ["4", "5"], "gold": 0},
            ]
            schema_rows = [
                {"context_options": ["A", "B"], "continuation": ".", "gold": 0},
            ]
            lm_rows = [
                {"context": "Hi", "continuation": " there"},
            ]

            with open(data_dir.joinpath("tiny_mc.jsonl"), "w", encoding="utf-8") as f:
                for row in mc_rows:
                    f.write(json.dumps(row) + "\n")
            with open(data_dir.joinpath("tiny_schema.jsonl"), "w", encoding="utf-8") as f:
                for row in schema_rows:
                    f.write(json.dumps(row) + "\n")
            with open(data_dir.joinpath("tiny_lm.jsonl"), "w", encoding="utf-8") as f:
                for row in lm_rows:
                    f.write(json.dumps(row) + "\n")

            out = evaluate_core_from_bundle(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                eval_bundle_dir=bundle,
                max_per_task=-1,
            )

            self.assertIn("core_metric", out)
            self.assertEqual(set(out["results"].keys()), {"tiny_mc", "tiny_schema", "tiny_lm"})
            self.assertGreaterEqual(out["core_metric"], 0.99)


if __name__ == "__main__":
    unittest.main()

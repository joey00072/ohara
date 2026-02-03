import unittest

import torch

from ohara.inference import Inference


class DummyTokenizer:
    eos_token_id = 99

    def encode(self, _: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class DummyModel(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        bsz, seq_len = x.shape
        logits = torch.zeros(bsz, seq_len, 128)
        logits[:, :, 5] = 1.0
        return logits


class InferenceTests(unittest.TestCase):
    def test_sampler_temperature_zero_is_greedy(self):
        logits = torch.tensor([[[0.1, 0.2, 0.9]]], dtype=torch.float32)
        token = Inference.sampler(logits, temperature=0.0)
        self.assertEqual(token.item(), 2)

    def test_generate_returns_full_text_not_last_token(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        inf = Inference(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            max_new_tokens=3,
            use_kv_cache=False,
        )

        output = inf.generate("hello", stream=False, temperature=0.0)
        self.assertEqual(output, "1 2 3 5 5 5")


if __name__ == "__main__":
    unittest.main()

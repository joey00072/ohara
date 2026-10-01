import unittest

import torch

from ohara.inference import Inference
from ohara.models.gpt import Config, GPT


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


class CachedDummyModel(DummyModel):
    class Config:
        max_sequence_length = 6

    config = Config()

    def __init__(self):
        super().__init__()
        self.calls = []

    def build_kv_cache(self):
        return object()

    def forward(self, x, cache=None, position=0):
        self.calls.append((x.tolist(), cache is not None, position))
        return super().forward(x)


class InferenceTests(unittest.TestCase):
    def test_unit_temperature_samples_when_top_p_disabled(self):
        torch.manual_seed(5)
        logits = torch.tensor([[[2., 1.9, 1.8, 1.7]]]).expand(200, -1, -1)
        samples = Inference.sampler(logits, temperature=1., top_p=0.)
        self.assertEqual(set(samples.flatten().tolist()), {0, 1, 2, 3})

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

    def test_cached_generation_prefills_then_decodes_one_token_at_a_time(self):
        model = CachedDummyModel()
        inf = Inference(
            model=model,
            tokenizer=DummyTokenizer(),
            device="cpu",
            max_new_tokens=10,
            use_kv_cache=True,
        )

        output = inf.generate("hello", stream=False, temperature=0.0)

        # Context length limits the three-token prompt to three generated tokens.
        self.assertEqual(output, "1 2 3 5 5 5")
        self.assertEqual([len(call[0][0]) for call in model.calls], [3, 1, 1])
        self.assertEqual([call[2] for call in model.calls], [0, 3, 4])

    def test_generate_rejects_empty_prompt_and_negative_length(self):
        class EmptyTokenizer(DummyTokenizer):
            def encode(self, _: str) -> list[int]:
                return []

        with self.assertRaises(ValueError):
            Inference(DummyModel(), EmptyTokenizer(), device="cpu").generate(
                "", stream=False
            )
        with self.assertRaises(ValueError):
            Inference(DummyModel(), DummyTokenizer(), device="cpu").generate(
                "hello", max_new_tokens=-1, stream=False
            )


def test_default_gpt_inference_matches_explicit_uncached_generation():
    model = GPT(Config(vocab_size=32, hidden_size=16, num_attention_heads=2,
                       num_hidden_layers=1, dropout=0.0, max_sequence_length=8))
    automatic = Inference(model, DummyTokenizer(), device="cpu")
    uncached = Inference(model, DummyTokenizer(), device="cpu", use_kv_cache=False)
    options = dict(max_new_tokens=4, temperature=0.0, stream=False)
    assert automatic.generate("hi", **options) == uncached.generate("hi", **options)


def test_cache_builder_returning_none_uses_full_prefix():
    class NoCacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def build_kv_cache(self):
            return None

        def forward(self, inputs):
            self.lengths.append(inputs.size(1))
            return torch.zeros(*inputs.shape, 8)

    model = NoCacheModel()
    Inference(model, DummyTokenizer(), device="cpu").generate(
        "hi", max_new_tokens=3, temperature=0.0, stream=False,
    )
    assert model.lengths == [3, 4, 5]


if __name__ == "__main__":
    unittest.main()

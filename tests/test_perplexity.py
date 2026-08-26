import math
import unittest

import torch

from ohara.perplexity import fixed_block_perplexity, sliding_window_perplexity


class UniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, token_ids):
        return torch.zeros(*token_ids.shape, self.vocab_size)


class PerplexityTests(unittest.TestCase):
    def test_fixed_blocks_report_token_weighted_cross_entropy(self):
        model = UniformModel(vocab_size=7)
        result = fixed_block_perplexity(
            model,
            torch.arange(12) % 7,
            device="cpu",
            sequence_length=4,
            batch_size=2,
        )
        self.assertEqual(result["predicted_tokens"], 9)
        self.assertAlmostEqual(result["loss_nats_per_token"], math.log(7), places=6)
        self.assertAlmostEqual(result["token_perplexity"], 7.0, places=5)

    def test_sliding_windows_score_each_token_after_the_first_once(self):
        model = UniformModel(vocab_size=5)
        result = sliding_window_perplexity(
            model,
            torch.arange(11) % 5,
            device="cpu",
            sequence_length=6,
            stride=2,
        )
        self.assertEqual(result["predicted_tokens"], 10)
        self.assertAlmostEqual(result["token_perplexity"], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()

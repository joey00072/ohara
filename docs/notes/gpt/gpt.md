# GPT

Implementation: [gpt.py](../../../ohara/models/gpt.py).

This model uses learned position embeddings, LayerNorm, and causal attention.
`Config.mlp` selects the feed-forward block; the default MLP uses SiLU.
`forward(tokens)` accepts `(batch, sequence)` token IDs and returns vocabulary
logits. It has no KV cache; generation recomputes the full prefix.

[Original research](https://openai.com/research/language-unsupervised).

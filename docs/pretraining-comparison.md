# Small pretraining comparison

Compare a randomly initialized Llama with the Qwen Next example on the same
8,388,608 training tokens. This is an early learning-curve experiment, not a
converged benchmark or an equal-FLOP comparison.

| Model | Non-table parameters | N-gram table parameters | Architecture |
| --- | ---: | ---: | --- |
| Llama | 99,893,760 | 0 | 16 dense layers, width 512, FFN 1456, 8 query / 4 KV heads |
| Qwen Next | 100,050,976 | 50,028,544 | 8 hybrid layers, width 512, 8 experts / top-2, one MTP horizon |

Qwen's non-table count includes its 5,878,976 MTP parameters, embeddings, output
head, and n-gram projections. Its conditional expert/table parameters do not all
participate for each token. The extra memory and different active computation
are part of the requested comparison; this does not isolate the effect of n-grams.

Both use the GPT-2 tokenizer (`EleutherAI/gpt-neo-125m`), an untied 50,304-row
embedding/output vocabulary, sequence length 512, and the same packed text.
The data script streams the `sample-10BT` subset of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), removes
exact-text duplicates, and assigns whole documents to training or validation
using a text hash. It reserves 131,072 held-out tokens and records dataset and
tokenizer revisions plus checksums of the actual token arrays. Attention may
cross document boundaries in both models; EOS is inserted before each document.

Train for 512 optimizer steps, 16,384 tokens/step, using AdamW with a 3e-4 peak
learning rate, 32-step linear warmup, cosine decay to 3e-5, betas (0.9, 0.95),
weight decay 0.1, and gradient clipping at 1. Parameters and moments remain FP32;
matrix operations use BF16 autocast. These shared settings are not separately
tuned for either architecture. One seed cannot establish a robust ranking.

The plot compares **plain next-token cross-entropy in nats**, excluding Qwen's
router, indexer, and MTP auxiliary terms. Training still optimizes those terms;
the complete objective is saved separately. Validation uses the same held-out
tokens every 32 steps, including before training and at the final step.

```bash
uv run python examples/compare_pretraining.py describe
uv run python examples/compare_pretraining.py prepare
CUDA_VISIBLE_DEVICES=0 uv run python examples/compare_pretraining.py train --model llama
CUDA_VISIBLE_DEVICES=1 uv run python examples/compare_pretraining.py train --model qwen
uv run --with matplotlib python examples/compare_pretraining.py plot
```

Install Qwen's native kernel dependencies as in [its guide](qwen38.md). Each
training command uses one GPU; they can run concurrently on separate GPUs or
sequentially on one. Output directories must be empty to prevent mixing runs.
The output includes recipes, per-step JSONL metrics, final training checkpoints,
and PNG/SVG loss plots. Keep downloaded data and checkpoints outside git.

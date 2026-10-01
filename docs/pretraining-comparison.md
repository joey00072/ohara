# Llama / Qwen Next pretraining comparison

[Script](../examples/compare_pretraining.py) for training both models on the same
8,388,608 tokens and plotting next-token cross-entropy. It matches token budgets,
not FLOPs or total parameter counts.

| Model | Parameters outside n-gram tables | N-gram parameters |
| --- | ---: | ---: |
| Llama | 99,893,760 | 0 |
| Qwen Next | 100,050,976 | 50,028,544 |

Qwen's first count includes 5,878,976 MTP parameters. Run `describe` for the full
configurations and counts.

## Run

```bash
uv run python examples/compare_pretraining.py describe
uv run python examples/compare_pretraining.py prepare
uv run python examples/compare_pretraining.py train --model llama
uv run --with flash-linear-attention==0.5.2 python examples/compare_pretraining.py train --model qwen
uv run --with matplotlib python examples/compare_pretraining.py plot
```

Training requires a CUDA GPU. Commands above run sequentially; set
`CUDA_VISIBLE_DEVICES` to run them on separate GPUs. Data goes to
`data/pretraining-comparison`; results go to `runs/pretraining-comparison`.
Preparation requires an empty data directory, and training requires an empty
model output directory. There is no resume option.

## Defaults

- FineWeb-Edu `sample-10BT`, GPT-2 tokenizer, sequence length 512.
- Exact-text deduplication and document-hash train/validation assignment.
  EOS precedes each document; attention can cross document boundaries.
- 131,072 held-out target tokens; dataset/tokenizer revisions and array checksums
  are saved in `metadata.json`.
- 512 steps, 16,384 tokens per step: micro-batch 8, accumulation 4.
- AdamW, peak LR 3e-4, 32-step warmup, cosine decay to 3e-5, weight decay 0.1,
  gradient clipping at 1. FP32 parameters/moments, BF16 autocast.
- Validation before training, every 32 steps, and at the final step.

Each run writes `recipe.json`, `metrics.jsonl`, and `final.pt`. Training includes
Qwen's auxiliary losses; plots use plain next-token cross-entropy. `plot` requires
matching recipes and completed metric logs, then writes `loss.png`, `loss.svg`,
and `summary.json`. This short, single-seed run is not a converged benchmark.

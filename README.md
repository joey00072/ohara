# Ohara

PyTorch language models, training and fine-tuning scripts, and a browser chat UI.

## Install

Requires Python 3.11+ and `uv`.

```bash
git clone https://github.com/joey00072/ohara.git
cd ohara
uv sync
```

## Quick start

Try a pretrained chat model:

```bash
uv run python examples/chat_web.py \
  --checkpoint joey00072/ohara-moe-0.9B-a91M-chat-d12
```

Open <http://localhost:8080>. The model downloads from Hugging Face on first use.

To train your own small Llama on TinyStories:

```bash
uv run python examples/train_llama_engine.py --chat-tokens
```

This streams the dataset and saves checkpoints to `./ckpt/model.pt`.
Then fine-tune for chat and serve the result:

```bash
uv run python examples/train_sft.py --pretrained-checkpoint ./ckpt/model.pt
uv run python examples/chat_web.py --checkpoint ./ckpt/sft.pt
```

Training uses the GPU when available. See the [pretraining guide](./docs/pretrain.md)
for your own data, multiple GPUs, and resuming runs. Each script accepts `--help`.

## Explore

| Location | Purpose |
| --- | --- |
| [ohara/models/](./ohara/models/) | Dense, MoE, and recurrent models. |
| [ohara/modules/](./ohara/modules/) | Shared attention, feed-forward, routing, and cache layers. |
| [ohara/runtime/](./ohara/runtime/) | Device placement, precision, parallelism, and checkpointing. |
| [ohara/trainer.py](./ohara/trainer.py) | Training loop, evaluation, and metrics. |
| [ohara/adaptor/lora.py](./ohara/adaptor/lora.py) | LoRA replacement, freezing, and merging. |
| [examples/](./examples/) | Command-line entrypoints for training, evaluation, and export. |
| [tests/](./tests/) | Unit, integration, and distributed regression tests. |
| [docs/](./docs/README.md) | Usage guides, paper notes, and archived reviews. |
| [experiments/](./experiments/) | Frozen per-paper research snapshots. |

The [speedrun script](./runs/speedrun.sh) runs data preparation through chat.
See [the guides](./docs/README.md) for pretraining, Qwen3.8, and model comparisons.
`ref/` contains local read-only reference clones, not package dependencies.

## Development

```bash
uv run pytest
uv run ruff check .
```

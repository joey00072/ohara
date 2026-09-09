# Ohara

My collection of PyTorch implementations of language models and research papers.
It includes dense and MoE models, training and fine-tuning scripts, and a browser
chat UI. Use it to run experiments or copy pieces into your own projects.

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

- [Models](./ohara/models/) — Llama, Qwen3, Mamba, RetNet, and others.
- [Examples](./examples/) — training, evaluation, export, and scaling sweeps.
- [Paper notes](./docs/notes/) and [experiments](./experiments/).
- [Speedrun](./runs/speedrun.sh) — the full data → pretrain → fine-tune → chat pipeline.

## Development

```bash
uv run pytest
uv run ruff check .
```

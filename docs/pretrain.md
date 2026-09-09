# Pretraining

Two ways to feed the trainer:

- **stream** straight from Hugging Face — nothing to prepare, good for getting going.
- **pretokenize** to disk first — faster per step, worth it once you train the same corpus twice.

## Train (streaming)

`examples/train_llama_engine.py` is the current entrypoint. It tokenizes on the fly, so you can
start immediately:

```bash
uv run python examples/train_llama_engine.py
```

Everything is a flag; `--help` lists them all. A few worth knowing:

```bash
uv run python examples/train_llama_engine.py \
    --dataset roneneldan/TinyStories \
    --tokenizer EleutherAI/gpt-neo-125m \
    --hidden-size 512 --num-layers 8 --num-heads 8 \
    --seq-len 512 --batch-size 16 --grad-accum-steps 4 \
    --max-iters 20000 \
    --optimizer muon --lr-schedule wsd \
    --precision bf16_mixed
```

The script drives [`ohara.runtime.OharaEngine`](../ohara/runtime/engine.py) for device placement,
mixed precision, DDP and tensor parallel, and [`ohara.trainer.Trainer`](../ohara/trainer.py) for the
loop itself (periodic eval, checkpoints, tok/s, MFU, bits-per-byte).

Multi-GPU is torchrun plus the same script:

```bash
# data parallel across 2 GPUs
uv run torchrun --nproc-per-node 2 examples/train_llama_engine.py

# tensor parallel instead (--tp also reads the OHARA_TP env var)
uv run torchrun --nproc-per-node 2 examples/train_llama_engine.py --tp 2
```

For resumable runs, use `--num-workers 0` from the start. Remote streaming also
needs `--dataset-revision` set to the dataset's immutable 40-character commit;
local text/JSON/Parquet and token-bin inputs are fingerprinted automatically.
Then `--resume` restores model, optimizer, precision state, RNG, and the saved
input cursor without tokenizing the consumed history again:

```bash
# Example: resume a run started with these same local-corpus arguments.
uv run python examples/train_llama_engine.py --dataset ./data/my-corpus --num-workers 0 --resume
```

Resume validates batch size, sequence length, seed, data identity, DP layout,
and the training recipe. Old checkpoints without an input contract and runs
started with worker prefetch cannot provide exact resume and are rejected by
`--resume`; their model weights remain loadable. Changing worker count to zero
only after interruption does not recover the missing worker state.

`--print-every` defaults to 10. Loss/timing scalars are read at reporting,
evaluation, and checkpoint intervals; use `--print-every 1` for per-step output.
MFU accounts for every participating GPU, including tensor-parallel ranks.

To watch it in W&B, run `uv run wandb login` first; `--logger trackio` logs locally without an account. It looks something like this:

![train](./src/image.png)

## Pretokenize first (optional)

Download and tokenize a dataset into `./data`:

```bash
uv run python examples/prepare_dataset.py tinystories
```

`tinystories`, `minipile`, `fineweb-edu` and `openhermes` are wired up; `--help` shows the flags
(`--tokenizer` to override, `--push --hf-username you` to upload the result). Depending on the
dataset this takes a while.

Under the hood that is [`ohara.pretokenize.DatasetPreprocessor`](../ohara/pretokenize.py), which you
can also call directly for a corpus that is not in the list:

```python
from ohara.pretokenize import DatasetPreprocessor

DatasetPreprocessor(
    dataset_name="roneneldan/TinyStories",
    tokenizer_name="microsoft/phi-2",
    splits=["train", "validation"],
).process_and_save()
```

The result is read back by [`ohara.dataset.PreTokenizedDataset`](../ohara/dataset.py).

## Scaling sweeps

For iso-FLOP sweeps rather than a single run, `examples/scaling_laws.py` plans the grid, shells out
to the training script for each point, and fits the curves:

```bash
uv run python examples/scaling_laws.py --help
```

# Pretraining

Run commands from the repository root after `uv sync`. Each script accepts `--help`.

## Streaming data

Train a small Llama on TinyStories:

```bash
uv run python examples/train_llama_engine.py --chat-tokens
```

Defaults: sequence length 256, batch size 8 per rank, four accumulation steps,
10,000 optimizer steps, and checkpoints at `./ckpt/model.pt`. `--chat-tokens`
reserves vocabulary entries for later chat fine-tuning.

To change the model and training recipe:

```bash
uv run python examples/train_llama_engine.py \
  --dataset roneneldan/TinyStories \
  --tokenizer EleutherAI/gpt-neo-125m \
  --hidden-size 512 --num-layers 8 --num-heads 8 \
  --seq-len 512 --batch-size 16 --grad-accum-steps 4 \
  --max-iters 20000 --optimizer muon --lr-schedule wsd \
  --precision bf16_mixed --chat-tokens
```

`--dataset` also accepts local text, JSON, and Parquet inputs. Use `--text-column`,
`--train-split`, and `--validation-split` to match the source.

## Token bins

For repeated runs, stage text and tokenize it once:

```bash
uv run python examples/prepare_scaling_data.py \
  --dataset roneneldan/TinyStories --output-dir ./data/tinystories
uv run python examples/pretokenize_corpus.py \
  --corpus ./data/tinystories --chat-tokens
uv run python examples/train_llama_engine.py \
  --dataset ./data/tinystories --chat-tokens
```

Staging defaults to 100,000 training documents and 10,000 validation documents;
set `--train-documents` and `--validation-documents` to change those limits.
The tokenizer writes `train.bin`, `validation.bin`, and metadata sidecars.
Training automatically uses bins when both splits exist. Pass `--no-token-bins`
to read text instead. Tokenizer and chat-token settings must match the bins.

The older `examples/prepare_dataset.py` workflow writes datasets for
[`PreTokenizedDataset`](../ohara/dataset.py), not token bins for this entrypoint.

## Multiple GPUs

```bash
# Data parallelism on two GPUs.
uv run torchrun --nproc-per-node 2 examples/train_llama_engine.py

# Tensor parallelism on two GPUs.
uv run torchrun --nproc-per-node 2 examples/train_llama_engine.py --tp 2
```

`--tp` also reads `OHARA_TP`. Tensor-parallel attention requires zero dropout.
When using the runtime API, prepare the model before constructing the optimizer
and prepare dataloaders before creating iterators or workers. Custom iterable
datasets must expose `configure_data_parallel(rank, world_size)` and yield the
same inputs on tensor-parallel ranks.

## Resume

Use `--num-workers 0` from the start. Remote streaming also requires an immutable
40-character `--dataset-revision`; local inputs and token bins are fingerprinted.
Resume with the same data and training arguments plus `--resume`:

```bash
uv run python examples/train_llama_engine.py \
  --dataset ./data/tinystories --chat-tokens --num-workers 0 --resume
```

Resume restores model, optimizer, precision, RNG, and input cursor. It validates
the batch size, sequence length, seed, data identity, data-parallel layout, and
training recipe. Legacy checkpoints without input state and runs started with
worker prefetch cannot resume exactly; their model weights remain loadable.
Changing worker count after interruption cannot recover the missing state.

## Memory and metrics

- `--loss-chunk-size 1024` applies the vocabulary head and loss in token chunks.
  Training recomputes those chunks during backward to reduce activation memory.
  The default `0` uses full logits.
- `--print-every 10` controls progress output; use `1` for every step.
- `--evaluate-bpb` enables bits-per-byte evaluation.
- `--logger trackio` logs locally. For W&B, run `uv run wandb login` first.
- MoE quantile balancing retains FP32 token-by-expert statistics until each
  optimizer step. Memory grows with accumulated tokens and expert count;
  distributed updates also gather statistics across ranks.

For scaling sweeps, run `uv run python examples/scaling_laws.py --help`.
For fine-tuning and serving, see the [quick start](../README.md#quick-start).

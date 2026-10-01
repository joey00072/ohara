# Qwen3.8-Flash-Next

[Implementation](../ohara/models/qwen38/) of the text backbone and multi-token
prediction (MTP). Models start from random weights.

## Run

From the repository root after `uv sync`:

```bash
# Small reference model; runs on CPU or GPU.
uv run python examples/train_qwen38.py --synthetic --backend torch --steps 3 --seq-len 17

# CUDA kernels, with the optional FLA dependency.
uv run --with flash-linear-attention==0.5.2 python examples/train_qwen38.py \
  --synthetic --backend cuda --steps 3 --seq-len 65
```

For real data, replace `--synthetic` with `--train-bin path/to/train.bin` from
the [token-bin pipeline](pretrain.md#token-bins). Pass `--config settings.json`
with `vocab_size` and `eos_token_id` matching the tokenizer.

Use `--tokens-per-step` to set the global token budget per optimizer update, or
`--grad-accum-steps` to set accumulation directly (default: 1). The token budget
must divide evenly by `seq_len * micro_batch_size * (world_size / cp_size)`.

The default model has 4 layers, width 128, 8 experts with top-2 routing, and one
MTP horizon. Inspect either preset without allocating weights:

```bash
uv run python examples/train_qwen38.py --describe
uv run python examples/train_qwen38.py --preset official --describe
```

## Features

- Three Gated DeltaNet layers followed by one sparse-attention layer.
- Gated residual streams, routed experts, and one shared expert.
- Bigram/trigram embeddings at layer 2; histories reset after EOS.
- MTP shares token embeddings and the output head.
- CUDA uses BF16 autocast, FLA, grouped expert matmuls, and Triton sparse attention.
- `--compile` compiles gated residual operations. `activation_checkpointing`
  is a JSON config option; `--loss-chunk-size` controls vocabulary-loss memory.

Training uses AdamW with a constant learning rate and auxiliary losses. There is
no pretrained-weight converter, vision encoder, generation KV cache, or attention
mask separating packed documents. Full-size and multi-node training are unvalidated.

## Multiple GPUs and checkpoints

The CLI uses FSDP2 and row-sharded n-gram embeddings when launched with `torchrun`:

```bash
uv run --with flash-linear-attention==0.5.2 torchrun --standalone --nproc-per-node=2 \
  examples/train_qwen38.py --synthetic --backend cuda --steps 2 --seq-len 17 \
  --checkpoint-dir ckpt/qwen38
```

Add `--cp-size 2 --seq-len 128` for context parallelism. Sequence length must be
divisible by `cp-size * block_size`; the FLA path requires micro-batch size 1.
The distributed CLI requires CUDA.

`--checkpoint-dir` saves model, optimizer, RNG, and input state. Resume with
`--resume ckpt/qwen38/step-00000002` and the same configuration, data, and rank
count. Use a larger `--steps` value to continue training.

See [distributed APIs and checks](qwen38-scaling.md) for TP, EP, and test coverage.

## References

- [Model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Architecture report](https://arxiv.org/html/2608.30320v1)
- [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

The n-gram hash layout adapts the Transformers reference, copyright 2026 The
Qwen Team and HuggingFace Inc., under [Apache 2.0](../ohara/models/qwen38/LICENSE).

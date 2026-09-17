# Qwen3.8-Flash-Next example

This implements the language backbone and multi-token prediction (MTP), with
small defaults for architecture experiments. No pretrained weights are downloaded.

## Run a small model

```bash
uv run python examples/train_qwen38.py --synthetic --backend torch --steps 3 --seq-len 17
```

The CPU path is a differentiable correctness reference. For a CUDA GPU, install
the optional DeltaNet kernels into the existing environment:

```bash
uv pip install flash-linear-attention==0.5.2
uv run --no-sync python examples/train_qwen38.py --synthetic --backend cuda --steps 3 --seq-len 65
uv run --no-sync python examples/benchmark_qwen38.py
```

CUDA training uses BF16, FLA chunked Gated DeltaNet, grouped expert matmuls, and
an indexed Triton sparse-attention kernel with backward. The sparse kernel reads
selected KV rows directly and recomputes probabilities in backward. It uses
FP32 accumulation and atomic KV gradient updates; results are not bitwise
deterministic. `--backend torch` selects reference operations on either device.

For real data, replace `--synthetic` with `--train-bin path/to/train.bin` from
the [token-bin pipeline](pretrain.md). Set `vocab_size` and `eos_token_id` to
match the tokenizer through `--config settings.json`. Synthetic loss only checks
that training works; it says nothing about model quality.

## Architecture

| Component | Released text configuration | Small default |
| --- | --- | --- |
| Layers / width | 48 / 2560 | 4 / 128 |
| Attention pattern | 3 Gated DeltaNet, then 1 QSA | Same |
| QSA query / KV heads, dimension | 24 / 2, 256 | 4 / 1, 32 |
| DeltaNet key / value heads, dimension | 16 / 48, 128 | 2 / 4, 32 |
| Sparse indexer | 4 heads, blocks of 4, 2048-token budget | 4 heads, blocks of 4, 32-token budget |
| Gated residual branches / rank | 4 / 320 | 4 / 16 |
| Routed experts / selected / width | 512 / 10 / 640 | 8 / 2 / 64 |
| Shared experts | 1, sigmoid output gate | Same |
| N-grams | Bigram + trigram, injected at layer 2 | Same, smaller hash tables |
| MTP | One shared QSA/MoE layer | One teacher-forced horizon |

QSA selects only fully observed blocks and includes the incomplete causal tail.
Its indexer learns through KL distillation from detached sparse attention
probabilities. N-gram histories reset after EOS. Gated residuals keep four
streams throughout the backbone. MTP shares token embeddings and the output
head; `mtp_steps` controls how many future-token losses reuse its layer.

See [the model code](../ohara/models/qwen38/),
[the sparse kernel](../ohara/kernels/qwen38_sparse.py), and
[correctness tests](../tests/test_qwen38.py). Query chunking, activation
checkpointing, and chunked cross-entropy bound intermediate memory. `--compile`
optionally compiles gated residual operations.

## Scaling later

Inspect the released dimensions without allocating weights:

```bash
uv run python examples/train_qwen38.py --preset official --describe
```

The optional distributed path uses FSDP2 for layers and routes n-gram lookups to
the rank owning each embedding row, avoiding a full-table all-gather. Test it
with the small preset first:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=2 examples/train_qwen38.py \
  --synthetic --backend cuda --steps 2 --seq-len 17 --checkpoint-dir ckpt/qwen38
```

Split each sequence across the same two GPUs with `--cp-size 2 --seq-len 128`.
Context shards must have equal lengths divisible by the QSA block size; the
FLA context path currently requires micro-batch size 1. Convolution halos,
n-gram histories, recurrent state, and MTP shifts cross shard boundaries.
FSDP reduces gradients across both data and context ranks. QSA gathers compact
KV and compressed index keys, so global KV memory still grows with sequence
length even though residuals and queries are sharded.

`--checkpoint-dir` writes distributed model/optimizer shards plus rank-local RNG
and input state. Resume from a completed `step-XXXXXXXX` directory using
`--resume`, keeping the same configuration, data, and rank count. Token-bin
content is fingerprinted for input resume validation.

This is an architecture example, not the original pretraining recipe. It uses
AdamW and sparse-stage indexer supervision from the start, without the report's
dense warmup or optimizer schedule. MTP recomputes its sparse selections. There
is no vision encoder, pretrained checkpoint converter, generation KV cache,
or packed-sequence boundary mask. Full-size cluster training
and long-context performance have not been validated.

See the [scaling checks](qwen38-scaling.md) for numerical comparisons, tested
combinations, and remaining cluster work.

## Sources

- [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Released configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
- [Architecture report](https://arxiv.org/html/2608.30320v1)
- [Transformers Qwen4Exp reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
- [SGLang MTP reference](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen4_exp_mtp.py)
- [Flash Linear Attention kernels](https://github.com/fla-org/flash-linear-attention)

The n-gram hash layout adapts the Transformers reference, copyright 2026 The
Qwen Team and HuggingFace Inc., under [Apache 2.0](../ohara/models/qwen38/LICENSE).

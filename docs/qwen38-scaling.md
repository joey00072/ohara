# Qwen3.8 distributed APIs and checks

The [training CLI](qwen38.md) supports FSDP2 with data parallelism, row-sharded
n-gram embeddings, and optional context parallelism (CP). Tensor parallelism
(TP) and expert parallelism (EP) are separate Python APIs.

## Test coverage

| Path | Tests | Checkpoints |
| --- | --- | --- |
| FSDP, data parallelism, n-gram sharding | [test_qwen38_fsdp.py](../tests/test_qwen38_fsdp.py) | DCP |
| FSDP, CP, n-grams, MTP | [test_qwen38_fsdp.py](../tests/test_qwen38_fsdp.py), [test_qwen38_cp.py](../tests/test_qwen38_cp.py) | DCP |
| TP experts, replicated backbone and MTP | [test_qwen38_tp_scaling.py](../tests/test_qwen38_tp_scaling.py) | DCP |
| EP experts, replicated backbone and MTP | [test_qwen38_ep_model.py](../tests/test_qwen38_ep_model.py) | One file per rank |

These tests compare small-model gradients, optimizer updates, or checkpoint
round trips with reference execution. The distributed cases use two processes
or two GPUs. CUDA tests skip when required devices are absent. Test coverage does not
establish full-size or multi-node performance.

```bash
uv run pytest tests/test_qwen38*.py -q
```

For CUDA tests, add `--with flash-linear-attention==0.5.2` after `uv run`.

For a single-GPU sparse-attention benchmark:

```bash
uv run python examples/benchmark_qwen38.py --length 256 --budget 64
```

This reports forward/backward latency and extra peak allocation for gather+SDPA
and indexed Triton attention. It does not measure whole-model throughput.

## TP

From [tensor_parallel.py](../ohara/models/qwen38/tensor_parallel.py):

- `tensor_parallelize(model, mesh)` shards expert intermediate dimensions.
- Feed the same batch to every TP rank; do not divide the loss by TP degree.
- Build AdamW with `tensor_parallel_param_groups(model)` from
  [distributed.py](../ohara/models/qwen38/distributed.py).
- Clip gradients with `clip_tensor_parallel_grad_norm` from the same
  `distributed` module.

## EP

From [expert_parallel.py](../ohara/models/qwen38/expert_parallel.py):

- `expert_parallelize(model, process_group)` assigns experts to owner ranks.
- Feed distinct, equally sized batches to EP ranks.
- Call `sync_model_gradients` after backward, then
  `clip_expert_parallel_grad_norm` from
  [distributed.py](../ohara/models/qwen38/distributed.py) before the optimizer step.
- Save local model and optimizer state separately for each rank. The training
  CLI's DCP wrapper rejects EP models.

## Limits

TP/EP conversion starts from an unsharded model. Combining TP or EP with each
other or with FSDP/CP is unvalidated. Resume requires the original parallel
layout; EP resharding, pipeline parallelism, and host-offload prefetch are not
implemented. CP shards residuals and queries but still gathers global KV.

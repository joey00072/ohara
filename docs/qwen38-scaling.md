# Qwen3.8 scaling checks

Use small models to verify numerical correctness before renting a cluster.
The [training example](qwen38.md) defaults to four layers and eight experts.
The released dimensions are inspected on the meta device, without allocating
full-size weights.

## What each mechanism addresses

| Mechanism | Purpose | Implementation and small-scale check |
| --- | --- | --- |
| Data parallelism | More independent examples per step | Rank-separated token bins, gradient accumulation, global mean losses |
| FSDP2 | Parameter, gradient, and optimizer memory | Layer sharding; compare every gradient, clipped norm, and AdamW update with an unsharded model |
| N-gram row sharding | Large conditional-memory tables | Owner-routed IDs/rows and reverse gradient exchange; uneven rows, repeated IDs, empty requesters/owners |
| Context parallelism | Long-sequence activation memory | Local residuals/queries, convolution halos, GDN state exchange, QSA KV gathering, global loss normalization |
| Tensor parallelism | Split large matrix work within a layer | Expert intermediate dimensions; replicated inputs and reduced partial outputs |
| Expert parallelism | Keep each routed expert on an owner rank | Token all-to-all, local experts, reverse output exchange |
| Activation checkpointing | Saved activation memory | Recompute layers and sparse query chunks; compare gradients with recomputation disabled |
| Sharded checkpoints | Save and resume without gathering all weights | Fresh model/optimizer restore followed by another verified update |

| Tested combination | CPU, 2 ranks | CUDA, 2 GPUs | Checkpoint format |
| --- | --- | --- | --- |
| FSDP + data parallelism + n-gram sharding | Yes | Yes | DCP |
| FSDP + CP + n-gram sharding + MTP | Yes | Yes | DCP |
| TP expert matrices + replicated backbone/MTP | Yes | Yes | DCP |
| EP owner-routed experts + replicated backbone/MTP | Yes | Yes | One file per rank |

TP and EP are separate small-model building blocks, not additional launcher
flags. Their combination with each other or with FSDP/CP has not been validated.
The training CLI implements the first two rows.

FSDP tests cover ordinary data parallel batches and the combination of FSDP,
context parallelism, row-sharded n-grams, two MTP horizons, and activation
checkpointing. Both run on two CPU processes and two CUDA devices. The CUDA
DeltaNet context test separately compares cross-rank state and all input
gradients against the sequential reference with slow decay, so forgetting the
boundary state cannot accidentally pass.

The CPU context test compares full-model logits, loss, every parameter gradient,
and two optimizer updates against the unsplit sequence. EOS tokens occur at
shard boundaries, and MTP losses cross those boundaries. CUDA smoke runs also
exercise the combined BF16 FLA, Triton, grouped-expert, FSDP, and checkpoint path.

## Run the checks

```bash
uv run --no-sync python -m pytest tests/test_qwen38*.py -q
```

CPU checks run without GPUs; CUDA checks skip when their required devices are
absent. Install FLA as described in the [example guide](qwen38.md) before running
the CUDA tests. Multi-process tests use small tensors and process-group timeouts.

The tested server has two RTX PRO 6000 Blackwell GPUs with PyTorch 2.10.0,
Triton 3.6.0, and FLA 0.5.2. On one GPU, the sparse-attention microbenchmark at
sequence 256, budget 64, heads 8/2, dimension 64, BF16 measured:

| Forward + backward | Latency | Extra peak allocation |
| --- | --- | --- |
| Gather + SDPA reference | 2.627 ms | 130.0 MiB |
| Indexed Triton | 0.084 ms | 0.9 MiB |

These are measurements for this small kernel workload, not end-to-end or
cluster speedups. Re-run `examples/benchmark_qwen38.py` on the intended shapes.

## Before a cluster run

The [architecture report](https://arxiv.org/html/2608.30320v1) discusses TP/DP
and a distributed Muon optimizer that reconstructs whole matrices and balances
orthogonalization work. It also motivates host-resident n-gram storage and
prefetch. This example uses AdamW; it does not reproduce that optimizer system.
CP and EP here are scaling choices for this implementation, not a claimed
reproduction of the report's cluster topology.

Profile per-layer residency and communication before choosing degrees. FSDP
all-gathers each layer; EP can reduce expert residency and traffic, while TP
divides the comparatively narrow expert matrices further. CP still replicates
global compact KV; longer contexts may need ring or requested-row KV exchange.
The paper's host-offloaded prefetch path and a pipeline-parallel schedule are
not implemented. Pipeline parallelism is an optional next step if measured
layer residency or communication makes it useful; its shared embeddings/MTP
head and microbatch schedule would need separate tests.

Small-scale correctness does not establish multi-node throughput, network
overlap, fault recovery, or a safe full-size batch. Fixed-topology resume is
tested; changing the data/parallel layout is not an exact-resume promise.

## TP and EP APIs

`tensor_parallelize(model, mesh)` from `ohara.models.qwen38.tensor_parallel`
replaces backbone and MTP experts with DTensor shards. Every TP rank consumes
the same batch and backpropagates its ordinary loss, without dividing by TP
degree. Use `tensor_parallel_param_groups(model)` from the `distributed` module
when constructing AdamW: CUDA foreach optimizers cannot mix ordinary tensors
and DTensors in one parameter group. `clip_tensor_parallel_grad_norm` counts
replicated parameters once. The CUDA BF16 path uses grouped GEMMs when local
dimensions are divisible by eight; other shapes use the reference loop.

`expert_parallelize(model, process_group)` from `ohara.models.qwen38.expert_parallel`
replaces experts with owner shards. EP ranks consume distinct, equally sized
batches. Call `sync_model_gradients` after backward, then
`clip_expert_parallel_grad_norm` before the optimizer step. Router supervision
uses expert loads across the EP group. The conversion helpers start from a
small unsharded model; they are not full-size checkpoint loaders.

EP weights are ordinary local tensors. Save the complete local model and
optimizer state in a separate file for each rank, with rank/world-size metadata,
as in `tests/test_qwen38_ep_model.py`. The example's DCP wrapper explicitly
rejects EP models to prevent treating different expert shards as replicas.
Changing EP degree requires a resharding loader that is not implemented here.

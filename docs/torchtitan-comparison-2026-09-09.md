# TorchTitan comparison — 2026-09-09

Compared Ohara `014f74b` with the local official `pytorch/torchtitan` checkout
`c4bef0bb7adf8ad2ffb2d4df62780a1e742a6b35`. Three subagents reviewed models,
distributed runtime, and training/data; the primary reviewer checked findings
and the hot path. Excluded `experiments/`. Reference code was read, never run or
modified. The findings below describe the reviewed commit; implementation status
is recorded separately here so the original evidence remains available.

## Implementation status

All ten numbered findings are addressed in the working tree:

| Findings | Change |
|---|---|
| 1 | Portable checkpoints load on CPU; model/optimizer tensors are sliced into local TP shards before device transfer. The training entrypoint releases the loaded payload after restoring metadata. Saving gathers CPU shards through Gloo only onto rank zero, without materializing full GPU tensors. |
| 2, 10 | Checkpoints contain rank-local input contracts and actual iterator state. Token bins seek directly to an epoch/offset. Streaming restores source position, its owned shuffle buffer, packing leftovers, and shuffle RNG. Batch/worker/corpus/tokenizer/DP/recipe changes fail validation. Evaluation preserves input state and process RNG. |
| 3 | Token-bin and legacy datasets reject insufficient rank/worker shard capacity; the pretraining CLI checks token-bin capacity before starting workers. |
| 4 | Both MoEs use a shared precision-controlled router linear: autocast is disabled before promoting the projection operands to FP32. Parameter names remain compatible. |
| 5 | Qwen rejects unsupported scaled/per-layer RoPE in both HF config formats; default nested theta is preserved. YaRN itself is not implemented. |
| 6 | Engine checkpoints include versioned precision and GradScaler state; runtime restoration validates compatibility. |
| 7 | TP gradient clipping sums local contributions with replica accounting and performs one collective, retaining the clip coefficient on-device. |
| 8 | Finite checks remain on-device until the optimizer boundary, with asynchronous CUDA assertions. Failure status is coordinated across all ranks, including pure TP when clipping is disabled. Loss scalars and timing are read at observation intervals. CUDA events replace device-wide per-step waits; pretraining/SFT expose `--print-every`, default 10. Logit references are released before backward. |
| 9 | MFU uses the full participating device count, GPU-name matching favors specific names, and token counters sum actual global input counts across variable-size batches. |

Exact resume intentionally rejects legacy checkpoints without an input contract,
worker-prefetched inputs, and finite iterable cycle state. Remote streaming
requires an immutable 40-character dataset revision; local inputs are hashed.
Ordinary training and loading legacy model weights remain supported. TP portable
checkpoint saves require Gloo and enough host RAM on rank zero for the full
checkpoint; loads still read the portable checkpoint into host memory per rank.
This is not a distributed-checkpoint format migration.

Validation includes a real dropout-Llama/AdamW run resumed from an intermediate
checkpoint, reproducing final model and optimizer tensors bit-for-bit, plus
two-process Gloo TP/DDP tests. Tests prohibit full-tensor materialization during
TP save, check uneven shard restoration and one-collective clipping, preserve
streaming shuffle/packing across epochs, and reproduce the BF16 router case.
The final integrated suite passed **414 tests and 2 subtests**, with **1 CUDA-only
skip**, in 100.21 seconds. Two existing SWIG deprecation warnings remain.
`ruff check .`, `uv lock --check`, and `git diff --check` passed. Test command:

```bash
PYTHONPATH=/tmp/ohara-testdeps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --active --no-sync python -m pytest -q -o faulthandler_timeout=120
```

A subsequent Luna review caught missing cross-rank finite-status coordination
in pure TP without clipping. The fix uses a world-group reduction before the
optimizer. Its new two-rank Gloo regression passed, and all 17 trainer tests
passed; the regression verifies a rank-local NaN stops both optimizers. The
414-test full-suite count above predates this additional test.

Vocabulary-parallel output/loss, FSDP2, and other capacity features mentioned as
future opportunities remain outside these fixes. NCCL/GPU execution, throughput,
and the locked Torch 2.10 environment remain unverified; tests use the available
Torch 2.13 CPU environment.

These are source-level findings with small CPU reproductions where noted.
No GPU throughput or large-model memory benchmark was performed. Previously
passing CPU tests do not establish these properties.

## Correctness and reliability

1. **High — TP resume defeats model sharding.**
   `ohara/runtime/engine.py:529` loads the complete checkpoint onto each device
   before distributing tensors. The original payload is returned at line 579
   and retained by `examples/train_llama_engine.py:498` during training, so full
   model tensors remain resident in addition to the sharded model. A model that
   fits for training can therefore fail to resume. Saving also assembles a full
   host copy on every TP rank (`engine.py:503`). TorchTitan loads distributed
   target state via `torchtitan/components/checkpointer/dcp.py:354` and `:396`.
   Use sharded training checkpoints; CPU loading, staged transfer, and releasing
   weight payloads are an interim mitigation.

2. **High — resume does not preserve the input-pipeline contract.**
   `examples/train_llama_engine.py:518` reconstructs the cursor as saved batch
   count times the *current* batch size. `ohara/trainer.py:591` does not save the
   original batch size, worker count, data identity, seed, or DP layout. Saving
   after 100 batches of eight and resuming with four starts at block 400 instead
   of 800. A workers=2 run resumed with workers=0 is accepted despite different
   batching order; unchanged total world size does not imply unchanged DP layout.
   TorchTitan persists rank-local iterator state and validates DP degree in
   `torchtitan/components/data/loader.py:149`. Persist actual iterator state and
   the input contract; reject incompatible resumes.

3. **High — empty input shards can spin forever.**
   `ohara/tokenbin.py:270` repeats an empty loop when a rank/worker shard has no
   blocks. Its constructor checks only that the entire dataset has a block.
   One block with shard 1 of 2 and `infinite=True` was reproduced on CPU: `next()`
   neither yielded nor raised before an alarm interrupted it. The legacy loader
   in `ohara/dataset.py:103` has the same structure. Validate capacity against
   ranks times workers and establish coordinated exhaustion behavior. TorchTitan
   explicitly guards distributed exhaustion in
   `torchtitan/components/data/loader.py:87`; that is a design comparison, not
   proof that it covers every possible empty dataset.

4. **Medium — MoE router logits are rounded before their FP32 cast.**
   `ohara/modules/moe.py:80` and `ohara/modules/moe_grouped.py:137` perform the
   linear projection under autocast, then call `.float()`. Widening cannot recover
   the lost precision that the quantile-balancing comments promise. Actual
   `GroupedMoE._route` reproduced a route change on CPU: FP32 logits
   `[1.001, 1.002]` selected expert 1; BF16 autocast produced `[1, 1]` and selected
   expert 0. TorchTitan wires `RouterGateLinear` through
   `torchtitan/models/common/moe.py:198` and `:313`; its implementation in
   `torchtitan/models/common/linear.py:49` preserves FP32 output/backward compute.
   Implement an explicitly controlled router projection and add autocast tests.

5. **Medium — Qwen HF loading silently ignores scaled RoPE.**
   `ohara/models/qwen3.py:47` discards `rope_scaling`, while line 230 constructs
   plain theta-based frequencies. A CPU config reproduction accepted a YaRN
   configuration and produced the same config as the unscaled input. Scaled
   long-context checkpoints can consequently run with different position
   rotations. TorchTitan explicitly handles YaRN in
   `torchtitan/models/common/rope.py:300`. Support and round-trip scaling or
   reject unsupported configurations. This finding concerns the Qwen HF loader,
   not Ohara's explicitly native-only Llama loader.

6. **Medium — FP16 resume drops GradScaler state.**
   `ohara/trainer.py:591` saves model/optimizer/RNG but no dynamic scale or growth
   tracker; engine launch creates a fresh scaler. Resumed training can skip
   different updates from uninterrupted training. Persist and restore scaler
   state, with a nondefault-state regression test. This follows the principle of
   complete checkpoint state; it does not imply TorchTitan uses FP16 GradScaler.

## Performance and measurement

7. **TP clipping performs a collective per sharded parameter.**
   `ohara/runtime/engine.py:427` materializes each parameter's norm separately,
   then synchronizes through Python `float`. TorchTitan combines norms before
   materializing the total at `torchtitan/distributed/utils.py:647`. Aggregate
   local squared norms with correct treatment of replicated/sharded tensors and
   reduce once. Keep the existing numerical-equivalence tests. Speedup needs GPU
   measurement; the excessive collective count follows directly from the code.

8. **The training hot path forces frequent host/device synchronization.**
   `ohara/trainer.py:434` and `:540` synchronize every step; line 470 calls
   `.item()`, line 497 branches on a device finite check per microbatch, and line
   538 converts loss to a Python float even when not printing. TorchTitan keeps
   normalization and finite checks on-device and gates metric work on logging
   (`torchtitan/trainer.py:883`, `:949`, `:997`, `:1005`). Use asynchronous checks,
   logging-interval scalar reads, and measured timing windows without losing
   failure detection. Also profile full-vocabulary logits: Ohara's default TP
   plan leaves embedding/output projection replicated, whereas TorchTitan has
   vocabulary-parallel loss (`torchtitan/components/loss.py:32`). This is a
   capacity/throughput opportunity, not evidence of incorrect cross-entropy.

9. **MFU is inflated by two accounting bugs.**
   `ohara/trainer.py:210` divides by DP world size, although pure TP uses multiple
   devices for one data replica. With full-model FLOPs, TP=2 overstates MFU by 2x.
   The device lookup at line 192 matches `L40` before `L40S`, selecting half the
   peak declared for L40S and doubling its reported MFU. Both were reproduced
   with CPU-only synthetic inputs/mocks. Use total participating device capacity
   and unambiguous GPU-name matching before trusting optimization comparisons.

10. **Streaming resume reprocesses the consumed history.**
    `ohara/dataset.py:262` tokenizes documents before applying the saved skip at
    line 269. Recovery cost grows with consumed history; token-bin recovery also
    regenerates/shuffles preceding epochs. TorchTitan restores iterator state
    directly (`torchtitan/components/data/loader.py:172`). Store a resumable
    source/packing cursor rather than replaying preprocessing from the beginning.

## What to take from TorchTitan

Fix checkpoint capacity/continuity, empty shards, and precision/config handling
first. Then remove unnecessary collectives and synchronization, repair MFU, and
benchmark a representative GPU workload. FSDP2, activation checkpointing,
sequence/vocabulary parallelism, and composing TP with DP are useful later
capacity features, not automatic correctness requirements for a small trainer.
Our explicit rejection of unsupported parallel modes is preferable to claiming
they work. Copying TorchTitan's entire architecture would not substitute for
tests of these specific contracts.

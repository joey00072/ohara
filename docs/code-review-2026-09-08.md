# Ohara code review — fix report (2026-09-08)

Scope: everything under `ohara/`, `examples/`, `tests/`, packaging and docs.
`experiments/` is excluded from this report.

Method: eight parallel read-throughs (models, MoE/core modules, training stack,
data pipeline, scaling + distillation, runtime/inference/chat/webui,
eval/utils/adaptors/packaging), each of which reproduced its suspected bugs on
CPU where possible. The top five were re-reproduced independently.

Verification tags used below:

- **[repro]** reproduced twice (reviewer plus an independent script).
- **[verified]** reproduced once with a script or an exact code-path trace.
- **[plausible]** consistent with the code but not executed (usually needs a GPU
  or multi-process launch).

Test status at time of review: `273 passed, 1 skipped`, `ruff check .` clean.
The suite was run with the system torch 2.13 and a `/tmp`-staged pytest because
`uv sync` cannot complete (see §0). Passing tests do not cover any of the
critical findings.

Independent Codex review added on the same date: three subagents plus a primary
reviewer examined the current working tree, excluding `experiments/` from these
additions. New findings are T12, T13, M16, P16, S5, and X11; overlapping findings
are strengthened in place. This separate validation used
`uv run --active --no-sync python` with Torch `2.13.0+cu130`. Its unittest run
reported 141 passed, one CUDA-only skip, and seven module-import errors from
missing pytest/Lightning. Locked-environment pytest and Ruff could not run
because dependency setup exhausted disk space. These results are separate from
the original review's test run above; they do not establish compatibility with
locked Torch 2.10.0.

---

## Implementation status — 2026-09-09

A subsequent [TorchTitan comparison](torchtitan-comparison-2026-09-09.md)
identified and fixed additional checkpoint, input-state, router precision, and
training-performance issues. That follow-up passed 414 tests and 2 subtests,
with one CUDA-only skip; its report records the newer compatibility limits.

The fixes below are applied to the working tree. The original findings are kept
as an audit trail; their old line numbers and pre-fix observations are historical.
No files under `experiments/` or `ref/` were changed. Existing uncommitted feature
work was preserved.

| Area / finding IDs | Disposition |
|---|---|
| T1 | Zero-norm hyperspherical matrices are rejected with an actionable error. Dense MuonH sweeps select standard initialization. Grouped zero-init output matrices require an additive optimizer or explicit nonzero initialization. |
| T2–T4, T6–T13 | Fixed: scaler overflows skip updates and discard poisoned routing scratch; SFT pads byte lookups; DPO accepts the common mask; dashboard parsing, attention scaling, global training metrics, relative resume timing, per-parameter AdamH counters, cleanup, rank RNG/sampler handling, global token weighting, and inactive Muon matrices have regression coverage. |
| T5 | Not treated as a correctness defect: tied weights intentionally belong to the embedding group. README now states the LR behavior and recommends untied weights for the nanochat comparison recipe. No automatic hyperparameter change or rejection. |
| M1–M12, M14–M16 | Fixed, including temporal scan, learnable top-1 routing, supported grouped-kernel fallback, empty expert graph participation, DDP buffer handling, full config recovery, separate expert width in benchmarks, real equivalence tests, configurable activation, collective participation and explicit masks. |
| M13 | Optional performance proposal deferred: caching differentiable casts across forwards requires optimizer/autograd invalidation. The current casts are correct; no GPU benchmark establishes that the extra machinery is worthwhile. |
| D1–D7, D9–D15, D17–D20 | Fixed or explicitly scoped: tied-load validation, model config fields, causal/position-aware RetNet, BF16 Mamba, incremental Phi rotation, length validation, SDPA/GQA, meta-device loading, normalization options, cache sizing/reset/int8, and inference cache fallback. Gemma remains explicitly a research model without a compatible HF loader. |
| D8, D16 | Derived rotary tables regenerate on load, so stale buffers do not override configured angles. Llama retains its legacy persistent 2x table shape because old shape-only checkpoints depend on it; other applicable models use nonpersistent, bounded tables. Removing Llama's legacy shape is intentionally deferred to a checkpoint-format migration. |
| D21 | Broad API consolidation is optional architecture cleanup, not a remaining correctness failure. Existing public model interfaces remain compatible. |
| A1–A7 | Fixed: base weights/bias and DoRA magnitude preserved, adapters freeze the base, dropout is serializable, merge is safe, target filtering honored, and dead initialization removed. |
| R1–R4, R7–R9 | Fixed: local TP heads, compiled-name matching, dense/MoE DDP, portable full TP checkpoints and resume, distributed rank discovery, integer mean reduction, shared sampler seed, and TP-aware gradient clipping. |
| R5, R6 | Unsupported TP+DP and pipeline/context/expert parallelism now fail explicitly instead of producing invalid runs. TP uses AdamW with FP32/BF16; Muon/hyperspherical TP and FP16 TP scaling are explicitly rejected until their mixed Tensor/DTensor operations are supported. |
| C1–C15 | Fixed: UTF-8/reply budgeting/system retention, input validation/origin checks, consistent sampling, streaming-reset races, cancellation cleanup, bounded caches, special-token quoting, timeouts, offline styling/defaults, fenced code, and thinking display. A Node VM harness exercises the frontend race and formatting regressions. |
| P1–P16 | Fixed or documented: metadata discovery/validation/publication/reuse, true corpus budgets, rank checks, chat-template output, sensible filtering, pad labels, raw UTF-8 bytes, HF-owned worker sharding, explicit endian, deterministic sampling, CPU-count fallback, and BOS-only semantics. HF may still share parsing when there are too few source shards; this is a documented source-layout limitation. |
| S1–S5 | Fixed fitting/fallback/duplicate-grid and path behavior; README explicitly states the limits of OLS fitting and lack of confidence intervals or joint parametric fits. |
| X1–X11 | Fixed cache/student compatibility, token-weighted accumulation, shared A/B data, masks, rebuild wording, chunked teacher conversion, centered cached log probabilities, cache identity, portable checkpoints, cautious observed-result wording, and stable saturated residual gradients. |
| E1–E10 | Fixed oracle/span/boundary tests, few-shot limits, strict loading, complete sliding-window scoring, bundle handling, model dispatch, export dtype/config, and FP32 scoring. Ohara-native export format is explicitly documented. |
| U1–U7 | Fixed lazy optional Lightning summaries, direct NumPy dependency, unused dependencies, entrypoint/docs links, UV commands, cache resolution, MPS detection, iterator closing, independent run-name RNG, naming and ignores. The Torch 2.10 CUDA-12 pin is deliberately retained and documented; it is not changed to match the review machine. |

Environment: removed only the documented abandoned `.tmpBVIajT` extraction
(1.4 GB), after checking that no process had an open file inside it. Other
projects/worktrees and shared environments were not deleted. Locked Torch 2.10
installation remains unverified because available space is insufficient; tests
use the existing Torch 2.13 environment and staged test-only dependencies. The
lockfile was regenerated and `uv lock --check` passes.

Integrated verification (Torch 2.13.0+cu130, CPU): **364 passed, 1 CUDA-only
skip, and 2 subtests passed** in 120.59 seconds. This includes real two-process
Gloo TP/DDP runs, compiled TP, TP clipping and checkpoint/optimizer resume,
unequal SFT masks, sparse MoE, actual HF streaming workers, saturated distillation
gradients, padded-vocabulary SFT, and the Node frontend regression harness.
Two existing SWIG deprecation warnings remain. The command was:

```bash
PYTHONPATH=/tmp/ohara-testdeps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --active --no-sync python -m pytest -q -o faulthandler_timeout=120
```

`ruff check .`, `uv lock --check`, `git diff --check`, JavaScript syntax checks,
and `bash -n runs/speedrun.sh` pass. Ruff/lock commands used
`UV_CACHE_DIR=/tmp/ohara-uv-cache` to avoid the constrained home-cache volume.
A prior distributed run hung while changes were still being integrated; its
stale workers were stopped. Its cause was not established. The final integrated
run and isolated MoE DDP reproduction completed successfully. CUDA execution
and the locked Torch 2.10 environment remain unverified; no GPU result is claimed.

A final focused run covering hyperspherical optimizers, trainer behavior,
training-entrypoint cleanup, lazy imports, and browser behavior passed **41 tests**
in 27.48 seconds after the last optimizer/cleanup changes.

---

## 0. Environment blockers (fix before anything else)

| Item | Detail | Action |
|---|---|---|
| Home volume full | `/home/joey` → `/mnt/HC_Volume_106662354` is at 100 % (51 G). Largest: `~/worktrees/wavelet-*` 18 G, `~/venvs/wavelet` 8.3 G, `~/.cache/uv` 9.5 G. | Free ≥ 6 G, then `uv sync`. |
| Project venv empty | `.venv/` was created empty by a failed `uv run`. | Delete and `uv sync` once space is available. |
| Stale uv temp dir | `~/.cache/uv/.tmpBVIajT` (1.4 G) is an orphaned partial torch extraction. | `rm -rf` it or `uv cache clean`. |
| torch pin mismatch | `pyproject.toml` constrains `torch==2.10.0`; the box's global venv is on 2.13+cu130. | See U2. |

---

## 1. Priority order

Fix in this order. Several findings silently corrupt results; others prevent
the affected workflow from running at all.

| # | Area | Finding | Sev |
|---|---|---|---|
| 1 | Optimizer | Hyperspherical optimizers pin zero-initialized weights forever (T1) | Critical |
| 2 | Modules | `RG_LRU` scans over channels, not time (M1) | Critical |
| 3 | Adaptors | `lora_from_linear` / `dora_from_linear` discard pretrained weights (A1, A2) | Critical |
| 4 | Runtime | Tensor parallelism crashes on first forward for `tp > 1` (R1) | Critical |
| 5 | Data | Token-bin sidecar `.json` is loaded as the training corpus (P1) | High |
| 6 | Models | Tied checkpoint missing both embedding tensors loads silently (D1) | High |
| 7 | Utils | `ohara.utils` imports Lightning eagerly; trainer and examples fail without it (U1) | High |
| 8 | Runtime | DDP crashes on loop-MoE at step 2 (R2); compile-before-TP never matches (R3) | High |
| 9 | MoE | Top-1 routing gives the router zero gradient (M2); no 16-byte alignment check (M3) | High |
| 10 | Chat | UTF-8 streaming corruption (C1); trimming drops the system turn (C2) | High |
| 11 | Data | Reader trusts sidecar token count over file size (P2) | High |
| 12 | Scaling | Fallback grid optimums fed into the power-law fit (S1) | High |
| 13 | Distill | Cache vocab never checked against student head (X1) | High |
| 14 | Eval | CORE test oracle cannot fail (E1) | High |
| 15 | Models | `GemmaConfig.intermediate_size` is not a field (D2); RetNet deviates from reference (D3) | High |
| 16 | Distill | Saturated top-k probability loses its corrective gradient (X11) | High |
| 17 | Training | DDP averages rank means instead of the global token mean (T12); Muon rejects inactive experts (T13) | Medium |
| 18 | Data/modules | Streaming worker sharding is applied twice (P16); SDPA ignores caller masks (M16) | Medium |
| — | Everything marked Medium/Low below | | |

---

## 2. Findings by area

Format: **ID — severity — tag** · location · problem · impact · fix · test.

### 2.1 Optimizer and training stack

**T1 — CRITICAL — [repro]** · `ohara/optimizer.py:81-89` (`_hypersphere_update_`) and `:105-111` (`_hypersphere_update_stacked_`)
- Problem: the step is `p -= lr · u · ‖p‖/‖u‖`, then `p *= ‖p‖/‖p'‖`. When `‖p‖ = 0` the step is exactly 0 and the rescale multiplies by `0/eps`. The parameter can never leave zero.
- Trigger: `init_style="nanochat"` zeros `attn.proj.weight` (`ohara/models/llama.py:460`), every `ff.down.weight` (`:470`), and `w_down` / `shared_down` in `ohara/modules/moe_grouped.py:305,309`.
- Impact: `examples/scaling_laws.py` passes `--init-style nanochat` unconditionally (`:307-309`) and offers `--optimizer muonh` (`:105-107`). Every MuonH sweep run trained a model whose attention and MLP outputs are permanently zero. Loss still decreases via embedding → norm → lm_head, so the iso-FLOP curves look plausible but are meaningless. Same for `train_llama_engine.py --optimizer adamh|muonh --init-style nanochat`.
- Reproduced: after 5 AdamH steps on a nanochat-init Llama, `attn.proj.weight` has 0/1024 nonzero entries with 1024 nonzero gradient entries. Muon+AdamW on the same model moves all 1024.
- Fix (pick one, first is simplest):
  1. In `build_adamh` / `build_muonh_adamh`, raise `ValueError` if any hyperspherical parameter has zero Frobenius norm, naming the parameter and suggesting `init_style="standard"`.
  2. Or in `_hypersphere_update_`, fall back to an additive step `p -= lr · u / ‖u‖ · target_norm` when `‖p‖ < eps`, where `target_norm` is a configured init norm.
  3. Or have `Llama._init_weights` use a tiny nonzero scale for the zeroed projections when a hyperspherical optimizer is requested.
- Test: parametrize `tests/test_optimizers_hypersphere.py` with `init_style="nanochat"` and assert `attn.proj.weight` is nonzero after one step (or that the builder raises).

**T2 — MEDIUM — [verified]** · `ohara/trainer.py:472-476, 480-489`; `ohara/runtime/engine.py:182-184, 387-392`
- Problem: after `engine.clip_gradients` (which calls `scaler.unscale_`), a non-finite grad norm raises `RuntimeError`; a non-finite loss also raises. With `PrecisionMode.FP16_MIXED` the GradScaler is enabled and its contract is that inf grads occur and the step is skipped.
- Impact: FP16 mixed precision dies on the first overflow. Default `--grad-clip-norm 1.0` guarantees the clip path runs.
- Fix: when `engine._scaler.is_enabled()`, do not raise; let `scaler.step` skip the update (or check `found_inf` and `continue`). Keep the raise for bf16/fp32.

**T3 — MEDIUM — [verified]** · `examples/train_sft.py:207-211, 377-381`; `ohara/trainer.py:262-263`
- Problem: SFT keeps the padded (wider) model vocabulary but builds `token_bytes` with `len(tokenizer)` entries. `Trainer.evaluate` raises "token_bytes is smaller than the model vocabulary" at the first eval.
- Impact: `--evaluate-bpb` crashes on any base trained with `--pad-vocab-to`.
- Fix: replicate `F.pad(token_bytes, (0, model_vocab - token_bytes.numel()))` from `examples/train_llama_engine.py:452-458`. Better: move that padding into `Trainer` or a shared helper so both scripts use it.

**T4 — MEDIUM — [verified]** · `ohara/dpo.py:137` vs `ohara/chat.py:58`
- Problem: `dpo.py` defines `IGNORE_INDEX = -100`; the rest of the repo (chat rendering, `ConversationDataset`, both trainers) uses `-1`.
- Impact: `sequence_logps` on chat-rendered labels raises `index -1 is out of bounds` on CPU, a device-side assert on CUDA.
- Fix: `from ohara.chat import IGNORE_INDEX` in `dpo.py`. Also delete or correct the wrong derivation comment at `dpo.py:230-239` (the code at `:158-166` is right; the comment is not).

**T5 — MEDIUM — [plausible]** · `ohara/optimizer.py:429-433, 559-565`; `examples/train_llama_engine.py:138-141`
- Problem: with `weight_tying=True` the `unembedding` group is empty and the shared matrix lands in the embedding group at lr ≈ 0.3·√(768/d) (≈ 0.52 at hidden 256). nanochat never ties, so this combination is outside the validated recipe. `--weight-tying` defaults to True and `--optimizer muon` is allowed with it.
- Fix: raise or warn in `build_muon_adamw` / `build_adamh` when `partition.unembedding` is empty and `partition.embedding` is non-empty.

**T6 — LOW — [verified]** · `examples/train_status.py:103-111`
- Problem: `MAX_ITERS_RE` matches the header line (which also contains `params=`) and `continue`s before `PARAMS_RE` runs, so `params` is always `None`. Also affects `examples/ab_status.py`.
- Fix: drop the `continue`, or test `PARAMS_RE` first.

**T7 — LOW — [verified by trace]** · `ohara/swa.py:75` vs `:117-119`
- Problem: the reference attention omits `1/√d`, while the flex path uses its default scale. `compare_swa_mask_vs_flex` therefore reports a large gap for any `head_dim > 1`. `tests/test_modules.py:304-314` re-implements the same unscaled formula so cannot catch it.
- Fix: multiply by `q.size(-1) ** -0.5` in the reference (or pass `scale=1.0` to flex). Make the test compare against `F.scaled_dot_product_attention` with an explicit window mask instead.

**T8 — LOW — [verified by trace]** · `ohara/trainer.py`
- `:499-504, 527` — `training_step_loss` / `train_iter_loss` are rank-0 local losses, not all-reduced (eval metrics are). All-reduce or label them "rank0".
- `:573-576` — `push_to_hub` runs on every rank and would `AttributeError` on a DDP-wrapped module. Guard with `engine.is_global_zero` and unwrap.
- `:194` — timing warmup compares against absolute `idx`, so after `--resume` no steps are excluded from the ETA average. Compare against `idx - start_idx`.

**T9 — LOW — [verified by trace]** · `ohara/optimizer.py:261`
- `_adamh_step` uses `states[0]["step"]` for every parameter's bias correction. A parameter with `grad is None` on some step falls behind and is later corrected with the wrong step. Use per-parameter step counters.

**T10 — LOW** · `examples/train_llama_engine.py:616-618`, `examples/train_sft.py:454-456`
- `tracker.finish()` is only called on the happy path. Wrap the run in `try/finally`. wandb's atexit hook covers most cases; trackio's may not.

**T11 — LOW — [plausible]** · `examples/train_llama_engine.py:230-233`, `ohara/runtime/engine.py:315-337`
- Seeds are identical across ranks (dropout masks correlate when `dropout > 0`); `_replace_sampler` discards the loader's `generator`, and `DistributedSampler` shuffles with seed 0 regardless of `--seed`. Offset the seed by rank and pass `seed=args.seed` to the sampler.

**T12 — MEDIUM — [verified by trace]** · `ohara/trainer.py:441-444, 478`
- Problem: the accumulated valid-token count is local to each rank. Backward divides by that local count, then DDP averages rank gradients equally. Packed SFT batches have unequal supervised-token counts, so this optimizes a mean of rank means rather than the global token mean.
- Example: at equal logits, rank 0 supervises one class-0 token and rank 1 supervises three class-1 tokens. Averaging the two local mean gradients cancels them; the correct global-token gradient favors class 1.
- Fix: all-reduce valid-token counts over the data-parallel group and multiply the local summed loss by the DP world size before dividing by the global count, compensating for DDP gradient averaging. Handle zero-token ranks consistently across the collective.
- Test: two CPU/gloo ranks with unequal masks must match a single-process update on the concatenated batch. The independent review traced the denominator and reduction; an end-to-end Trainer reproduction was blocked by missing Lightning.

**T13 — MEDIUM — [verified]** · `ohara/optimizer.py:288-292`; `ohara/modules/moe.py:123-126`
- Problem: loop-dispatched MoE skips unused experts, leaving their gradients `None`. Muon buckets their matrices with active same-shaped matrices, then raises if only part of a bucket has gradients. MuonH has the same restriction.
- Reproduced: a tiny Llama with four experts and top-2 routing, one input token, and `build_muon_adamw` fails on the first optimizer step with `a Muon group has missing gradients; all same-shaped matrix parameters must participate in each optimization step`.
- Impact: valid sparse routing can abort even single-process training. This is separate from the DDP reduction failure in M5/R2.
- Fix: support inactive matrices explicitly, defining whether their optimizer state advances, or ensure all expert parameters participate in the graph. Test an unused expert through an optimizer step, not just through forward/backward.

Verified correct in this slice (no action): AdamW matches `torch.optim.AdamW` bit-for-bit; Muon reproduces nanochat's kernel (Nesterov lerp, Polar Express, NorMuon variance reduction, cautious WD, `sqrt(max(1, rows/cols))` lr); Trainer gradient accumulation is token-weighted within each rank and clipped once (global DDP weighting remains incorrect: T12); LR schedules agree on 1-based `idx` and hit `min_lr` exactly; SFT shift/mask is correct; eval reductions are correct under TP duplication; checkpoint resume is consistent for `num_workers=0`.

### 2.2 Core modules: RNN, MoE, MLP

**M1 — CRITICAL — [repro]** · `ohara/modules/linear_rnn.py:40`
- Problem: `scan(alpha.mT.contiguous(), xbeta.mT.contiguous()).mT` hands pscan a `(B, D, L)` tensor. `pscan` scans dim 1 of a `(B, L, D, N)` input, so the recurrence runs along `D`. Every `h_t` depends only on `x_t`.
- Reproduced: output matches a recurrence over channels to 5.6e-17 and misses the correct time recurrence by 0.565 on O(1) values.
- Fix: `h = scan(alpha, xbeta)` (drop all three `.mT` and the `.contiguous()` calls). Plain `scan(A, X)` on `(B, L, D)` is already correct.
- Test: add `tests/test_linear_rnn.py` comparing `RG_LRU` against a Python loop over time, and `tests/test_pscan.py` comparing `pscan` forward/backward to a sequential scan for L in {1, 2, 3, 7, 8, 9, 16, 17, 33} in float64. Neither module has any test today.

**M2 — HIGH — [verified]** · `ohara/modules/moe_grouped.py:150-156`; `ohara/modules/moe.py:87-91`
- Problem: with `num_experts_per_tok=1`, softmax over one logit is identically 1, and sigmoid with `normalize_weights=True` divides the weight by itself. Router gradient is exactly 0 (softmax) or 1e-7 roundoff (sigmoid+normalize).
- Impact: a top-1 config trains a random-but-balanced router; only quantile balancing moves routing.
- Fix: raise in `__init__` when `k == 1 and (gate_fn == "softmax" or normalize_weights)`. Alternatively for k = 1 use the Switch-style unnormalized probability (softmax over all experts, gather at the winner).
- Test: assert router grad norm > 0 for k = 1 in the allowed configuration, and that the disallowed one raises.

**M3 — HIGH — [verified]** · `ohara/modules/moe_grouped.py:105-107, 166-195`
- Problem: `torch._grouped_mm` requires 16-byte-aligned row strides, so `dim` and `hidden_dim` must be multiples of 8 in bf16 (4 in fp32). Nothing validates this. `GroupedMoE(dim=64, hidden_dim=28)` in bf16 fails inside `_dispatch_grouped` with "strides should be multiple of 16 bytes". `tests/test_moe_grouped.py:376` uses `hidden_dim=28`; `examples/bench_moe.py:51` derives `intermediate // experts_per_tok`, which can land on a non-multiple of 8.
- Fix: in `__init__`, `if dim % 8 or hidden_dim % 8: raise ValueError(...)`, or round `hidden_dim` up to a multiple of 8 and document it. Fix the test fixture and `bench_moe` accordingly.

**M4 — HIGH — [plausible]** · `ohara/modules/moe_grouped.py:242-244`
- Problem: `use_grouped = flat_x.is_cuda and _grouped_mm_available()` ignores dtype; without autocast `compute_dtype = flat_x.dtype`. Torch's meta registration for `_grouped_mm` asserts bf16 inputs ("matching `_grouped_mm_cuda`"). Any fp32 CUDA forward (the existing CUDA test, fp32 eval, compile tracing) likely raises.
- Fix: `use_grouped = flat_x.is_cuda and compute_dtype is torch.bfloat16`, falling back to `_dispatch_reference` otherwise. Verify on a GPU.

**M5 — HIGH — [verified]** · `ohara/modules/moe.py:119-121` with `ohara/runtime/strategy/ddp.py`
- Problem: the loop MoE `continue`s past experts that received zero tokens, so their parameters are absent from the graph. DDP is built without `find_unused_parameters`, and fails at step 2 with "Expected to have finished reduction in the prior iteration". Reproduced with gloo, world 2, 8 experts top-2, 4 tokens.
- Fix: run zero-count experts on the empty slice (drop the `continue`) so every parameter participates, or set `find_unused_parameters=True` / `static_graph=True` when the model contains `MoE`.

**M6 — MEDIUM — [plausible]** · `ohara/runtime/strategy/ddp.py` (DDP construction) with the QB buffers in `moe.py` / `moe_grouped.py`
- Problem: DDP's default `broadcast_buffers=True` overwrites per-rank `qb_beta_sum`, `qb_beta_count`, and `expert_counts` with rank 0's values at every forward. With grad accumulation G > 1 the `all_reduce(AVG)` in `apply_qb_update` mostly averages rank-0 statistics, and `expert_load()` on ranks > 0 is wrong. Also re-broadcasts Llama's persistent `(1,1,S,S)` fp32 causal mask and RoPE buffers every step (~16 MB at S = 2048, see D9).
- Fix: `broadcast_buffers=False` in `DDPStrategy` (router bias is already synchronised by the all-reduce), or hold QB scratch as plain tensor attributes rather than buffers.

**M7 — MEDIUM — [verified]** · `ohara/chat_engine.py:132-137, 190-197`
- Problem: `config_from_state_dict` has no `moe_shared_exclusive` kwarg. A raw training checkpoint (the `torch.load` path at `:438-445`) trained with rejection is rebuilt with plain addition, producing wrong outputs with no error. `save_pretrained` / `config.json` round-trips it correctly.
- Fix: add the kwarg, thread it through both `Config(...)` constructions, and add a test mirroring `test_non_shape_routing_options_round_trip_when_supplied`. **This is part of the uncommitted work and should land with it.**

**M8 — MEDIUM — [verified]** · `examples/bench_moe.py:51-53`
- Problem: `build()` shrinks `intermediate_size` for the whole model, so dense layers inside the MoE model are also k× narrower. FLOPs ratio moe/dense measured: 1.001 at interval 1, 0.797 at interval 2, 0.695 at interval 4. The "matched FLOPs" claim is false for `--moe-layer-interval > 1`.
- Fix: keep `Config.intermediate_size` at the dense width and add a separate expert width (e.g. `moe_expert_hidden_dim`) consumed only by MoE layers.

**M9 — MEDIUM — [verified]** · `tests/test_moe_grouped.py:475-476`
- Problem: `SharedExclusiveTests` is defined after `if __name__ == "__main__": unittest.main()`, so running the file directly never collects the eight new tests. Pytest does collect them.
- Fix: move the `__main__` block to the end. **Part of the uncommitted work.**

**M10 — MEDIUM — [verified]** · `tests/test_moe_grouped.py:33-43`
- Problem: the load-bearing grouped-vs-reference equivalence test is CUDA-only, fp32-only, forward-only, and skipped on every CPU run. Torch 2.13's CPU `_grouped_mm` works: grouped == reference bit-exactly in bf16 on CPU, and all four parameter gradients match.
- Fix: call `_dispatch_grouped` directly so the test runs on CPU; compare gradients too; add a bf16 case. Note the fp32 cast on CUDA may itself fail (M4).

**M11 — LOW — [verified]** · `ohara/modules/mlp.py:94, 100`; `ohara/models/transformer.py:41, 128-133`
- Problem: `GLU.forward` hardcodes `F.silu` and ignores `self.activation`; `Transformer.Block` never passes `activation_fn=config.activation`. `activation="silu"|"relu"|"gelu"` produce bit-identical outputs.
- Fix: `self.down(self.activation(gate) * up)`; pass `activation_fn=config.activation`. Add a test that two activations differ.

**M12 — LOW — [verified by trace]** · `ohara/modules/moe.py:140-154`; `ohara/modules/moe_grouped.py:274-287`
- `apply_qb_update` returns early when `qb_beta_count == 0` before the collective; a rank that skipped the MoE forward would deadlock the others. Always participate (all-reduce a zero beta with weight 0, or all-reduce the count too).

**M13 — LOW (perf) — [plausible]** · `ohara/modules/moe_grouped.py:185-189`
- `self.w_gate.to(compute_dtype)` etc. re-cast all stacked expert weights every forward under autocast; autocast's weight cache does not cover manual casts. Cast once per step or keep bf16 copies.

**M14 — LOW** · `ohara/modules/moe.py` vs `moe_grouped.py` API drift
- `assert` vs `ValueError` validation; sigmoid never normalises in `MoE` but does by default in `GroupedMoE` (and `Llama` rejects `moe_normalize_weights=False` for the loop MoE at `llama.py:249`); `gate` vs `router`; `MoE.__init__` never calls `reset_parameters()`. Harmonise or document.

**M15 — LOW (tests)** · `tests/test_moe_grouped.py`
- `:104` `test_each_token_selects_k_distinct_experts` is tautological (`topk` cannot return duplicates); `:123` `test_router_bias_changes_selection_but_not_weights` makes its main assertion conditional on `unchanged.any()`; `:69` only checks non-zero output; `:498` triggers a requires_grad-to-scalar warning (use `.detach()`). No test exercises `shared_exclusive` through grouped dispatch, autocast/bf16, or `expert_counts` semantics.

**M16 — MEDIUM — [verified]** · `ohara/modules/attention.py:84-91`
- Problem: `CausalAttention.forward(mask=...)` passes `attn_mask=None` to SDPA, while the slow/verbose path uses the supplied mask. Caller-provided padding, sliding-window, or positional-bias masks disappear in normal execution.
- Reproduced: with dropout disabled and a diagonal-only additive mask, normal and verbose execution differed by a maximum of 7.16541. Changing verbosity must not change attention semantics.
- Fix: pass the requested mask to SDPA and make its interaction with causality consistent with the reference path. Test the two paths against the same explicitly masked attention computation.

Verified correct: pscan forward and both gradients match a sequential scan to ~1e-15 in float64 across power-of-two and padded lengths; grouped dispatch (sort/bincount/cumsum offsets, `index_add_` un-sort, weight gather) agrees with the reference loop forward and backward in bf16 and fp32 with empty experts; the quantile-balancing closed form and mean-centering are correct and converge (MaxVio 5.6 → 0.031 in 50 steps); `_reject` orthogonality and gradient flow are correct.

### 2.3 Models

**D1 — HIGH — [verified]** · `ohara/models/llama.py:592-596`
- Problem: with `weight_tying=True`, `allowed_missing` excuses both `token_emb.weight` and `vocab_proj.weight`, so a checkpoint missing both passes validation. A tied model with both tensors deleted from `model.safetensors` loaded without error and produced logits differing by 0.546.
- Fix: require at least one of the two keys to be present. Qwen3's loader does this correctly (only excuses `vocab_proj.weight`).
- Test: delete both tensors from a saved tied checkpoint and assert `from_pretrained` raises.

**D2 — HIGH — [verified]** · `ohara/models/gemma.py:20`; `ohara/models/mamba.py:71`
- Problem: `intermediate_size = 16 * 2048` has no annotation, so it is a class attribute, not a dataclass field. `GemmaConfig(intermediate_size=...)` raises `TypeError` and every model gets 32768 (also not Gemma's ratio; Gemma-2B is 2048 → 16384). The test config with `hidden_size=32` silently builds a 6.3 M-param GEGLU 32 → 32768. Same latent bug for `mamba.py:71` `dt_init_floor = 1e-4`.
- Fix: `intermediate_size: int = 16384`; `dt_init_floor: float = 1e-4`.

**D3 — HIGH — [verified by reference comparison]** · `ohara/models/retnet.py:66-69, 114-116`
- Problem (a): ohara normalises by row abs-sum then multiplies by the decay mask; torchscale's `parallel_forward` does `qk_mat * mask` first, then normalises. The comment "Normalize before applying decay, as in the reference" is backwards, and normalising over unmasked scores includes future positions.
- Problem (b): the XPos rotation (`theta_shift`) is never applied to q/k in the parallel form; the docstring at `:114` claims it is only needed by the recurrent form. This RetNet has no relative positional signal beyond decay.
- Independent reproduction of (a): for a one-layer, two-head RetNet with `d_model=16`, sequence length 8, and dropout disabled, changing only tokens 4–7 changed logits at tokens 0–3 by up to 0.00030756. Detaching the denominator does not remove its forward dependence on future tokens; subsequent normalization does not cancel it exactly because of epsilon.
- Fix: apply `theta_shift(q, sin, cos)` / `theta_shift(k, sin, cos)` (adjacent-pair rotation), then `* decay_mask`, then normalise. Fix the comments.

**D4 — MEDIUM — [verified]** · `ohara/models/mamba.py:236-238, 377-381`
- Problem: `A = -exp(A_log.float())` and `D.float()` promote the SSM path to fp32; `y * z` is fp32 and `self.out_proj` (bf16 weight) raises `expected m1 and m2 to have the same dtype` under pure bf16.
- Fix: `y = y.to(x.dtype)` before `y * z`. Keep the scan in fp32.

**D5 — MEDIUM — [verified]** · `ohara/models/phi.py:20` → `ohara/utils/__init__.py:2` → `ohara/utils/info.py:2`
- Problem: `phi.py` imports `ohara.utils.load` for a one-line `snapshot_download` wrapper, which drags in Lightning. See U1 for the root fix; locally, import `snapshot_download` directly.

**D6 — MEDIUM (perf) — [verified]** · `ohara/models/phi.py:92-101`
- Problem: the KV cache stores un-rotated keys and re-applies RoPE to the entire cache (upcast to fp32) on every decode step: O(T) redundant work and a ~42 MB fp32 transient per layer per token for phi-2.
- Fix: rotate `k` for the new positions with `offset=start_pos` before writing to the cache, as llama/qwen3 do.

**D7 — MEDIUM — [verified]** · `ohara/models/transformer.py:174-177`; `ohara/models/phi.py:169-185`
- Problem: no sequence-length validation. `Transformer` at `seq_len=16` runs silently at 20 (RoPE table is 2× oversized) and hits an `AssertionError` from `reshape_for_broadcast` at 33. Phi raises an opaque broadcast `RuntimeError`. `Inference.generate` clamps using `config.max_sequence_length`, which `Transformer` / `RetNet` / `RoFormer` configs do not have.
- Fix: raise `ValueError` when `T > max length` in both, matching gpt/roformer.

**D8 — MEDIUM — [plausible]** · `ohara/models/llama.py:271-272`; `ohara/models/transformer.py:161-162`
- Problem: RoPE tables are registered as persistent buffers (Qwen3/Gemma/RoFormer use `persistent=False`). Trainer checkpoints `model.state_dict()` (`trainer.py:558`) and `Engine.load` calls `load_state_dict` strictly (`engine.py:473`). Resuming after changing `max_sequence_length` or `rope_theta` fails or silently reuses stale angles; `save_pretrained` special-cases `freq_*` by hand to compensate.
- Fix: `persistent=False`; drop the special-casing at `llama.py:495, 592`.

**D9 — MEDIUM — [plausible]** · `ohara/models/llama.py:277-280`
- The `(1,1,S,S)` fp32 `-inf` causal mask is a persistent buffer that DDP re-broadcasts every step (see M6). The slow-attention path guarded by `hasattr(F, "scaled_dot_product_attention")` is unreachable on torch ≥ 2.0 in `llama.py:274-283`, `gpt.py:122-129`, `roformer.py:139-144`, `transformer.py:164-170`. Delete the mask buffers and the dead branches.

**D10 — MEDIUM — [plausible]** · `ohara/models/llama.py:558`; `ohara/models/qwen3.py:401`
- Problem: `from_pretrained` runs `cls(config)` at fp32 on CPU (random init of every tensor) before `.to(dtype)`. Qwen3-8B in bf16 needs ~32 GB host RAM and seconds of pointless RNG.
- Fix: construct under `torch.device("meta")` and `load_state_dict(assign=True)`, or `torch.nn.utils.skip_init`.

**D11 — MEDIUM (perf) — [verified by trace]** · `ohara/models/qwen3.py:169-177`; `repeat_interleave` at `llama.py:110-111`, `gemma.py:86-87`, `transformer.py:90-91`
- Problem: Qwen3 attention always materialises the `T×T` fp32 score matrix (≈ 4 GB transient at T = 8192, batch 1) and never uses SDPA; four models copy K/V `num_queries_per_kv`× with `repeat_interleave`.
- Fix: `F.scaled_dot_product_attention(..., enable_gqa=True)` with the boolean position mask pattern from `llama.py:124-131`.

**D12 — MEDIUM — [verified by trace]** · `ohara/models/gemma.py`
- Not Gemma-shaped: final norm is `nn.LayerNorm` (Gemma uses RMSNorm with `1+w`), no `√hidden_size` embedding scale, `config.bias` ignored (qkv has bias, GEGLU does not, regardless), `dropout` / `multiple_of` / `rotary_dim` unused, no `_init_weights`, no KV cache, no HF loader. Either finish it or mark it experimental in the docstring and README.

**D13 — LOW — [verified]** · `ohara/embeddings_pos/alibi.py:20-24`
- Diagonal bias is `-m` rather than 0 (distance off by one). Harmless under a causal softmax, wrong standalone. Fix: `rows = torch.arange(n)` (drop `+1`). Unused by any model.

**D14 — LOW** · `ohara/modules/attention.py:112-130`
- `reset_parameters` uses `std = head_dim**-0.5` (0.125 at d = 512) where the rest of the repo uses 0.02 / `d_model**-0.5`; `proj` uses `std/factor` with `±3·init_std` truncation bounds. Module is unused by `ohara/`.

**D15 — LOW** · `ohara/models/llama.py:197-198, 260`
- RMSNorm eps hardcoded to 1e-5 with no Config field; Qwen3 exposes `rms_norm_eps`. Add `rms_norm_eps` to `Config`.

**D16 — LOW** · `llama.py:266-268`, `transformer.py:160`, `gemma.py:156-159`, `roformer.py:135`
- RoPE tables precomputed for `2 × max_sequence_length` while `forward` rejects anything beyond `max_sequence_length`; half the buffer is unreachable.

**D17 — LOW — [verified by trace]** · `ohara/models/qwen3.py:48-68`
- `from_hf_config` silently ignores `use_sliding_window` / `sliding_window`. Raise if set.

**D18 — LOW** · `ohara/models/mamba.py:412`
- `return y, h.squeeze(1)` is a no-op with a "todo : pq ??" comment. Remove.

**D19 — LOW** · `ohara/modules/kv_cache.py`
- `int8=True` is unreachable from any model's `build_kv_cache`; no `reset()`, so `Inference.generate` re-allocates a full-length cache per call. Expose or delete int8; add `reset()`.

**D20 — LOW — [verified by trace]** · `ohara/inference.py:96-100`
- `Inference.generate` calls the model with three positional args when `use_kv_cache=True` even if the model has no `build_kv_cache`, which would `TypeError` for GPT / Transformer / RoFormer / RetNet / Gemma. Check `hasattr(model, "build_kv_cache")` and fall back.

**D21 — LOW** · API and duplication
- Field names differ (`max_sequence_length/hidden_size/num_attention_heads` vs `seq_len/d_model/num_heads`); only llama/qwen3/transformer have `_init_weights` (others use PyTorch defaults, embeddings N(0,1)).
- Duplicated: sharded safetensors loader (`llama.py:530-603` ≈ `qwen3.py:376-444`), position-based causal mask (three copies), GQA repeat block (four copies), causal `Attention` class (four near-copies). Extract shared helpers into `ohara/modules/`.

Verified correct: Qwen3 vs `transformers.Qwen3ForCausalLM` max |Δlogits| 9e-8 (untied) / 1.2e-7 (tied), 7-step cached decode matches full forward to 1e-7; Llama GQA batch-2 7-step cached decode matches to 4e-8; Phi 7-step to 6e-7; bf16 and autocast paths with KV cache work for Llama; the uncommitted `moe_shared_exclusive` plumbing on the model side is correct and round-trips through `save_pretrained`.

### 2.4 Adaptors (LoRA / DoRA)

**A1 — CRITICAL — [repro]** · `ohara/adaptor/lora.py:73`; `ohara/adaptor/dora.py:98`
- Problem: `lora.load_state_dict(linear.state_dict(), strict=False)` feeds keys `weight` / `bias` into a module whose keys are `linear.weight` / `linear.bias`. Every key is unexpected or missing and nothing loads. Reproduced: `weight preserved=False`, `outputs match=False` for both LoRA and DoRA.
- Impact: `replace_with_lora(model)` on any pretrained model replaces every Linear with kaiming-random weights. Fine-tuning then trains from scratch while the user believes the base is frozen. DoRA computes `magnitude` from the random weights.
- Fix: `lora.linear.load_state_dict(linear.state_dict())` (strict), or `lora.linear = linear` directly.
- Test: assert `torch.allclose(lora(x), linear(x))` immediately after wrapping, for bias and no-bias inputs.

**A2 — CRITICAL — [repro]** · `ohara/adaptor/lora.py:65-71`; `ohara/adaptor/dora.py:90-96`
- Problem: `LoRALinear(in, out, ...)` always constructs `nn.Linear(..., bias=True)`. A `bias=False` source (every projection in `ohara.models.llama`) gets a random bias that is never overwritten. Even after A1, outputs differ.
- Fix: pass `bias=linear.bias is not None` through to `nn.Linear`.

**A3 — MEDIUM — [verified]** · `ohara/adaptor/lora.py:25`; `ohara/adaptor/dora.py:26`
- `lora_dropout = lambda x: x` makes the module unpicklable (`torch.save(module)`, deepcopy via pickle, DataLoader worker pickling). Fix: `nn.Identity()`.

**A4 — MEDIUM — [verified]** · `ohara/adaptor/lora.py:44-47`; `ohara/adaptor/dora.py:53-57`
- `lora_trainable_only()` calls `self.linear.train(False)` (eval mode, meaningless for Linear) and never sets `requires_grad=False`. It only works because `mark_lora_as_trainable` freezes everything first. Fix: `for p in self.linear.parameters(): p.requires_grad_(False)`.

**A5 — LOW** · `ohara/adaptor/dora.py:73-77, 59`
- Gradient flows through `weight_norm`; PEFT and the DoRA paper (§4.3) detach the LoRA term inside the norm. `merge()` runs under autograd. Detach to match reference behaviour; wrap `merge` in `torch.no_grad()`.

**A6 — LOW — [verified]** · `ohara/adaptor/lora.py:87-92`; `ohara/adaptor/dora.py:114-119`
- A Linear not in `target_layer` still constructs a full adapter that is discarded (wasted allocation on large layers). `mark_*_as_trainable` and `merge_*` ignore `target_layer` entirely. Check `target_layer` before constructing; honour it in the other two functions.

**A7 — LOW** · `ohara/adaptor/lora.py:39-42`
- `kaiming_normal_` on `linear.weight` in `__init__` only when `rank > 0`; `lora_only` parameter unused. Remove the re-init (harmless after A1) and the dead parameter.

Test gap: `tests/test_modules.py:324-333` checks type and output shape only and cannot detect A1 or A2.

### 2.5 Runtime (distributed)

**R1 — CRITICAL — [verified]** · `ohara/runtime/tensor_parallel.py:26-30` (`validate`), `:59-61` (Colwise/Rowwise plan); `ohara/models/llama.py:95-97`
- Problem: `ColwiseParallel` (default `use_local_output=True`) gives each rank a `(B, T, hidden/tp)` q/k/v, but attention still does `.view(B, T, self.num_attention_heads, head_dim)` with the global head count. Reproduced with gloo (world 2, tp 2; world 4, tp 2): first forward raises `shape '[1, 8, 4, 8]' is invalid for input of size 128`. `validate` only checks `hidden_size % tp`, not head divisibility.
- Fix: after `parallelize_module`, divide `attn.num_attention_heads` and `num_key_value_heads` by `tp` on each rank (torchtitan pattern) and validate head divisibility in `engine._resolve_tensor_parallel_plan`.
- Test: a 2-process gloo test (`torch.multiprocessing.spawn`) that runs one forward/backward under `tp=2` and compares logits against the single-process model.

**R2 — HIGH — [verified]** · see M5 (DDP + loop MoE).

**R3 — HIGH — [verified]** · `ohara/runtime/engine.py:208`; `examples/train_llama_engine.py:283-288`
- Problem: the model is compiled before `engine.prepare`, and `apply_tensor_parallel` matches rules against `named_modules()` of the `OptimizedModule`, whose names are `_orig_mod.layers.0.attn.query`. `layers.*.attn.query` never matches (0 matches compiled, 7 raw) → `ValueError: No modules matched tensor parallel rules`.
- Fix: match on `strip_wrapper_prefixes(fqn)` or unwrap `_orig_mod` before parallelising; apply TP before compile.

**R4 — HIGH — [plausible]** · `ohara/runtime/engine.py:446-456`
- Problem: `engine.save` `torch.save`s the raw `state_dict()`, whose values are `DTensor`s under TP. Only rank 0 saves, so the checkpoint holds a single shard plus mesh metadata that `chat_engine.from_checkpoint` cannot load.
- Fix: `torch.distributed.checkpoint.state_dict.get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))`.

**R5 — HIGH — [plausible]** · `ohara/runtime/engine.py:176-177, 209-211`
- DDP over DTensor parameters (TP + DP) is unsupported by PyTorch without `torch.distributed.tensor.parallel.ddp._pre_dp_module_transform`. Unreachable today because R1 fails first. Use that transform, or FSDP2 `fully_shard` on the DP mesh dim.

**R6 — MEDIUM — [verified]** · `ohara/runtime/topology.py:359-364`; `ohara/runtime/engine.py:269-273`; `ohara/runtime/pipeline.py`
- Problem: `pp`, `cp`, `ep > 1` are accepted (`ParallelTopology(world_size=4, dp_shard=2, pp=2)` is constructed) but `pipeline.py` is never used by the engine. Each "stage" runs the full model, and separate DP groups per stage mean `pp` independent trainings on half the data that never synchronise.
- Fix: raise in `launch()` when any of `pp`, `cp`, `ep` != 1 until implemented.

**R7 — MEDIUM — [verified by trace]** · `ohara/runtime/topology.py:404-409`; `ohara/runtime/engine.py:149-151`
- Rank is derived only from env vars; world size from `dist` but not rank. A process group initialised without torchrun env makes every rank "rank 0": all ranks save and log, and `data_parallel_rank == 0` everywhere (identical shards). Fix: prefer `dist.get_rank()` / `dist.get_world_size()` when initialised.

**R8 — LOW — [verified]** · `ohara/runtime/engine.py:433-435`
- `all_reduce(int_tensor, "mean")` does in-place `/=` on a Long tensor → `result type Float can't be cast to Long`. Use `tensor.div_` on a float copy or return `tensor / world`.

**R9 — LOW** · see M6 (`broadcast_buffers`), T11 (seeds / sampler).

Test gap: there is no multi-process test of any kind, which is why R1–R3 went unnoticed. Add a gloo-based `tests/test_distributed.py` with 2-process DDP (dense and MoE) and TP smoke tests.

### 2.6 Inference, chat, web UI

**C1 — HIGH — [verified]** · `ohara/chat_engine.py:553-558, 560-576`
- Problem: the streaming diff only emits when `len(text) > len(emitted)`. A partial UTF-8 token decodes to `�`, which has the same length as the completed character, so the placeholder is emitted and never corrected. With the gpt-neo tokenizer `"日本語"` streams as `"���"`, `"hi 😀 there"` as `"hi � there"`. `generate()` is `"".join(generate_stream())`, so non-streaming output is wrong too.
- Fix: hold back output while `text.endswith("�")` (HF `TextStreamer` approach) and flush the remainder after the loop.
- Test: stream a fixed token sequence containing multi-byte characters and assert the joined result equals `tokenizer.decode(tokens)`. Replace the tautological `test_streaming_and_batch_generation_agree` at `tests/test_chat.py:451-457`.

**C2 — HIGH — [verified]** · `ohara/chat_engine.py:509-510`
- Problem: context trimming drops `turns[:2]`, i.e. `[system, user]`, leaving `[assistant, user, ...]`. A system prompt plus six pairs in a 64-token window raises `message 0 has role 'assistant', expected 'user'`. In the web UI every message errors until "New chat". For the Qwen template path it renders an assistant-first prompt.
- Fix: keep a leading system message and drop `turns[1:3]`.

**C3 — HIGH — [verified]** · `ohara/webui/server.py:133-140, 144`
- Problem: only `(ValueError, JSONDecodeError)` are caught. `{"top_k": null}` or `{"temperature": [1]}` raise `TypeError` → traceback on stderr and `RemoteDisconnected` for the client. `{"temperature": NaN}` (accepted by Python's `json`) passes `SamplingConfig` validation and fails after the 200 is sent.
- Fix: catch `TypeError`; validate with `isinstance(v, (int, float)) and math.isfinite(v)`; reject NaN/inf explicitly.

**C4 — MEDIUM — [verified]** · `ohara/chat_engine.py:507, 525-528`
- Problem: `render_prompt` accepts any prompt with `len(ids) < max_sequence_length`, then `max_new_tokens` is clamped to the remainder. A 60-token prompt in a 64 window with `max_new_tokens=100` yields a 4-character reply with no trimming and no error.
- Fix: pass the reply budget into `render_prompt` (`budget = max_seq - min(config.max_new_tokens, reserve)`) so trimming happens before generation is starved.

**C5 — MEDIUM — [verified]** · `ohara/inference.py:43-45`
- Problem: the sampler is greedy at exactly `(temperature=1.0, top_p=0.0)` (always token 0) but samples at `(0.999, 0.0)`. The constructor default therefore means "greedy", a discontinuity nobody expects.
- Fix: remove the special case; use `temperature <= 0` for greedy.

**C6 — MEDIUM — [verified]** · `ohara/webui/server.py:116-131`
- Problem: no Content-Type or Origin check, so a `text/plain` POST is accepted and any web page the user visits can POST to `127.0.0.1:8080` without a CORS preflight and burn GPU under the generation lock (worse with `--host 0.0.0.0`).
- Fix: require `Content-Type: application/json`; reject requests whose `Origin` host differs from `Host`.

**C7 — MEDIUM — [verified]** · `tests/test_chat.py:360-373`
- Two sampling tests are vacuous: 2000 unrestricted samples from `[10, 9, -50, -60]` still give `{0, 1}` and from `[20, 0, 0, 0]` give `{0}`; the excluded tokens are 20–60 nats down and cannot be sampled. Use logits like `[2.0, 1.9, 1.8, 1.7]` (verified to sample all four unrestricted) or assert on the masked probability tensor.

**C8 — MEDIUM — [verified by trace]** · `ohara/webui/static/app.js:230-235`
- "New chat" during streaming calls `abort()` then `conversation = []`; the fetch rejection is a microtask, so `:186-188` pushes the partial assistant reply into the new array. The next send goes out as `[assistant, user]` → server `ValueError` on every message until reload. Capture the conversation array (or a request id) inside `send()` and only mutate if still current.

**C9 — MEDIUM — [verified by trace]** · `ohara/webui/static/app.js:186-195`
- Error/abort paths pop the user message from `conversation` but leave its bubble in the DOM; the model never sees that turn again. Remove the bubble too, or keep the turn and offer retry.

**C10 — MEDIUM — [verified by trace]** · `ohara/chat_engine.py:534`; `ohara/models/qwen3.py:29-31`
- Per-request KV cache is sized to the full context: Qwen3-0.6B at 40960 = 28 × 2 × 40960 × 8 × 128 × 2 B ≈ 4.7 GB bf16 (9.4 GB fp32 on CPU) allocated and freed on every message. Size to `len(prompt) + max_new_tokens`, or keep one cache and reset its length.

**C11 — LOW — [verified]** · `ohara/webui/static/index.html:355` vs `ohara/chat_engine.py:75`
- `top_p = 0` means "disabled" but the slider's minimum is 0; dragging from 0.01 (near-greedy) to 0 jumps to fully random. Set `min="0.01"` or treat 0 as greedy.

**C12 — LOW — [verified by trace]** · `ohara/chat.py:313-314`
- User text is encoded with default special-token matching, so typing `<|user_end|><|assistant_start|>` breaks turn structure. Use `split_special_tokens=True`.

**C13 — LOW** · `ohara/webui/server.py:69-74, 157-164`
- No socket timeout (a short body parks a thread forever); the generation lock is held while writing to a possibly stalled client; a browser "stop" is only noticed on the next failed write. Set `timeout` on the socket; release the lock between token generation and write where possible.

**C14 — LOW** · `ohara/webui/static/app.js:49, 280-282`; `index.html:294, 308-310`
- Fenced-code renderer drops the first line even when code starts on the ``` line; if `/api/info` fails, midpoint slider values (temp 0.75, top-k 100, 1032 tokens) are sent; layout depends on Tailwind CDN + Google Fonts so the page is unstyled offline. Vendor the CSS.

**C15 — LOW** · `ohara/chat_engine.py:555`
- Qwen `<think>` / `</think>` are non-special added tokens, so `skip_special_tokens=True` leaves them as literal text and the UI has no thinking separation. Strip or render them.

Verified correct: static path traversal is blocked, model output is escaped before `innerHTML`, SSE framing is correct on both sides.

### 2.7 Data pipeline

**P1 — HIGH — [verified]** · `ohara/dataset.py:167-187` (glob loop at `:183`); `ohara/tokenbin.py:50-51`
- Problem: `_load_stream` globs `{split}*{suffix}` with `.json` tried before `.jsonl`. `write_token_bin` names its sidecar `Path(bin).with_suffix(".json")` → `train.json`, next to `train.jsonl`. In a directory containing all three, both splits resolve to `builder=json files=['train.json']`. `prepare_scaling_data.py:133-137` explicitly dodged this hazard for its stats file; `tokenbin.py` reintroduced it.
- Impact: `train_llama_engine.py --no-token-bins`, or any run where only one of `train.bin` / `validation.bin` exists (`use_token_bins` false at `:357`), loads the sidecar as the dataset and dies with `KeyError: dataset row does not contain text column 'text'`.
- Fix: rename the sidecar to `train.bin.json` (or `train.tokens.json`), and/or make the glob prefer `.jsonl` over `.json` and skip files whose stem matches a `.bin` sibling.
- Test: extend `test_streaming_text_dataset_resolves_local_split_files` with a `train.json` sibling.

**P2 — HIGH — [verified]** · `ohara/tokenbin.py:213, 225, 254`
- Problem: `num_blocks = metadata["tokens"] // block_size` from the sidecar; the memmap is opened with no shape. numpy memmap slicing past EOF returns a short array. A bin shorter than the sidecar claims (truncated copy, interrupted rsync, P3) yields short blocks → collate crash deep into an epoch (or, at `batch_size=1`, silent training on a short sequence). A bin longer than claimed never trains on its tail.
- Fix: in `__init__`, assert `bin_path.stat().st_size == tokens * itemsize` and open the memmap with `shape=(tokens,)`.

**P3 — MEDIUM — [verified by trace]** · `ohara/tokenbin.py:135-147`
- The bin is renamed into place at `:135`; the sidecar is written at `:147`. A crash between them leaves a new bin with the old sidecar (wrong `tokens`, `vocab_size`, `dtype`). Write the sidecar to a temp file and rename both, or write the sidecar first and rely on P2's size check.

**P4 — MEDIUM — [verified by trace]** · `examples/pretokenize_corpus.py:92-100`
- The reuse check compares only `vocab_size`. Llama-2 and Mistral-v0.1 both have 32,000 tokens; switching `--tokenizer` silently reuses the old bin. Rerunning after `--max-documents 1000` with the full corpus prints `reuse ...` and keeps the 1000-doc bin. Compare `existing["tokenizer"]` to `tokenizer.name_or_path` and record/compare the document limit.

**P5 — MEDIUM — [verified by trace]** · `examples/pretokenize_corpus.py:40-48`; `examples/scaling_laws.py:455-460`
- `_corpus_token_budget` reads `stats.train.json` (full staged corpus), but a bin built with `--max-documents` has far fewer tokens, so `max_epochs` is underestimated and the sweep silently re-epochs a small bin — the memorisation failure the guard exists to prevent. When `train.bin` exists, read `tokens` from its sidecar.

**P6 — MEDIUM — [verified]** · `ohara/tokenbin.py:187-188` vs `ohara/dataset.py:139-142`
- `TokenBinDataset` does not validate `0 <= data_rank < data_world_size`; rank 5 of world 4 yields rank 1's blocks (duplicated data, no error). Copy the two range checks from `StreamingTextDataset`.

**P7 — HIGH — [plausible]** · `ohara/pretokenize.py:131`
- `uv.lock` pins transformers 5.14.1. In v5, `apply_chat_template(..., tokenize=True)` defaults to `return_dict=True`, so the stored value would be a dict, `filter_fn` would see `len == 2` and drop every row (or Arrow type inference fails). `tests/test_tokenizer_hookup.py` stubs this method and cannot catch it. Pass `return_dict=False` or take `["input_ids"]`.

**P8 — HIGH — [plausible]** · `examples/prepare_dataset.py:34-37`; `ohara/pretokenize.py:19, 77`
- The `tinystories` recipe filters on `min_length=512` tokens; TinyStories documents are typically 150–400 tokens, so most of the dataset would be discarded. Make `min_length` a `Recipe` field and set it low for TinyStories; check row counts after a run.

**P9 — MEDIUM — [verified by trace]** · `ohara/dataset.py:108-109`
- `PreTokenizedDataset` pads with PAD (= EOS) and puts pads in the targets. With `ignore_index=-1`, every pad position is trained as "predict EOS after EOS"; for short documents padded to 2049 that is most of the loss mass. Mask target pad positions to `ignore_index`, or pack documents.

**P10 — MEDIUM — [verified by trace]** · `ohara/tokenizer.py:176-180`
- `get_token_bytes` decodes byte-level BPE tokens through `str`, so a token covering a partial multibyte sequence decodes to U+FFFD (3 bytes) and inflates its byte count from 1–2 to 3, biasing BPB downward for GPT-NeoX/Qwen-style tokenizers. For byte-level tokenizers, map `convert_ids_to_tokens(id)` through the inverse `bytes_to_unicode` table.
- Independent executable reproduction: a local ByteLevel BPE encodes `é` as two byte tokens; the lookup returns `[3, 3]`, counting six bytes instead of the actual two. Add this regression case; ASCII-only fake tokenizers cannot expose it.

**P11 — LOW — [verified by trace]** · `ohara/dataset.py:237-251, 257-258`
- Every (rank, worker) shard reads and JSON-parses the entire stream (8 ranks × 4 workers = 32×); resume via `start_block` re-tokenises every skipped document on every rank. Use `datasets.distributed.split_dataset_by_node` or shard the file list; warn when `start_block` is large.

**P12 — LOW** · `ohara/tokenbin.py:13, 111`
- Docstring says "little-endian" but `tofile` / `memmap` use native order. Use `np.dtype("<u2")` / `"<u4"` or fix the docstring.

**P13 — LOW** · `ohara/dataset.py:295, 304-306`
- `TinyShakespeareDataset` uses unseeded `random`; `torch.Tensor(ids).long()` goes through float32 (exact only for ids < 2^24); `except Exception` swallows tokenizer errors and triggers a re-download. Add a `seed`, use `torch.tensor(ids, dtype=torch.long)`, narrow the except.

**P14 — LOW** · `ohara/pretokenize.py:45, 47`
- `os.cpu_count() - 3` raises `TypeError` when `cpu_count()` returns `None`; `self.length = tokenizer.vocab_size` is misnamed and unused.

**P15 — LOW (design note)** · `ohara/tokenbin.py:97-99`
- Documents are prefixed with BOS only. For tokenizers with distinct BOS/EOS the model never sees EOS and cannot learn to stop. nanochat makes the same choice; document it.

**P16 — MEDIUM — [verified]** · `ohara/dataset.py:238, 264-266`
- Problem: Hugging Face's streaming `IterableDataset` already partitions source shards by PyTorch worker inside `_iter_pytorch`. `StreamingTextDataset` then applies another rank/worker modulo filter. Worker sharding is applied twice, dropping additional documents; an empty HF worker is treated as an invalid dataset.
- Reproduced: a single local JSONL, `StreamingTextDataset(max_length=2)`, and `DataLoader(num_workers=2)` produced one batch, then raised `RuntimeError: no usable documents found ... split 'train'` in worker 1 after HF reported that the dataset has only one source shard.
- Fix: let one layer own worker sharding, distinguish worker-local exhaustion from a globally empty corpus, and preserve disjoint rank sharding. P11's assumption that every worker reads the whole source is not valid for HF iterables that already shard workers.
- Test: real HF streaming data with two workers and one source shard must terminate cleanly; multiple source shards must cover all documents without duplication or loss. Plain-list mocks do not exercise HF's worker behavior.

Test gaps: `FakeTokenizer` in `tests/test_tokenbin.py:84-100` only emits ids 1–95 even with `vocab_size=151_936`, so the uint32 test never writes an id > 65535; no test for size/sidecar disagreement (P2), non-atomic sidecar (P3), or `data_rank` range (P6); sharding tested only with `shuffle=False`, no `num_workers > 0`, no shuffle+shard disjointness, no seed determinism, no `start_block` with sharding or epoch wrap; `tests/test_prepare_scaling_data.py` does not cover `skip_existing`, `force`, empty-stream `RuntimeError`, or temp-file cleanup.

Verified correct: the uncommitted dtype-widening change in `tokenbin.py` is correct and tested for the happy path; the atomic temp-file bin write and vocab-size guard work. Explicit `(rank, worker)` striding is correct for the token-bin reader, but not when layered on HF streaming worker sharding (P16).

### 2.8 Scaling laws (uncommitted)

**S1 — HIGH — [verified]** · `ohara/scaling.py:517-531, 584-606`
- Problem: `interior_optimum` was redefined as "measured minimum is not at the grid edge", so when the new gap-ratio guard rejects the quadratic, the raw grid point enters `fit_power_law`. On the grid encoded in the new test (gap ratio ≈ 10, every budget falls back), two budgets whose best depth coincides give `optimal_params_power_law = {exponent: 0.000, r_squared: 1.0}` — a "perfect" fit that is pure grid quantisation.
- Fix: fit the power law only on rows with `quadratic_interpolation == 1.0`, or expose `num_quadratic_optimums` and emit `power_law_warning` when any fallback row is included.
- Test: a grid where fallback rows would produce a spurious exponent; assert they are excluded or flagged.

**S2 — MEDIUM — [verified]** · `ohara/scaling.py:509-512`
- `_quadratic_fit` runs before the zero-gap guard. Two rows with identical `params_effective` adjacent to the minimum raise `ValueError: cannot fit a quadratic to degenerate points` and abort `analyze`. Aggregate duplicate-params rows (mean loss) when grouping, or compute `gap_ratio` first and skip the fit when any bracket gap is 0.

**S3 — LOW — [verified by trace]** · `ohara/scaling.py:737-738`
- For fallback optimums the `continue` skips the star marker as well as the curve, so panel 1 shows no optimum for those budgets while panels 2–3 do. Guard only the polyline.

**S4 — note** · Only IsoFLOP fitting with OLS in log-log exists; there is no Chinchilla approach-3 parametric fit (`E + A/N^α + B/D^β`, Huber in log space) and no bootstrap/CI, so exponents carry no uncertainty. Not a bug; worth stating in the README.

**S5 — MEDIUM — [verified by trace]** · `examples/scaling_laws.py:540-547`
- Problem: `subprocess.run(..., cwd=PROJECT_ROOT)` changes the interpretation of relative dataset, result, tokenizer, and cache paths, while the parent keeps resolving paths against its own working directory.
- Trigger: launch the sweep script from outside the repository with relative paths. The child can read a different corpus; even with an absolute dataset path, it can write results under the repository while the parent opens the corresponding path under the caller's directory and fails.
- Fix: resolve filesystem arguments before building the subprocess command, or keep parent and child working directories consistent. Preserve Hugging Face identifiers as identifiers rather than converting them to local paths.
- Test: invoke the sweep from a temporary working directory with a stub training subprocess and verify that the parent reads the exact result file the child writes. No expensive training run is needed.

Test gap: `test_isoflop_fit_uses_local_bracket_on_uneven_model_grid` exercises only the fallback branch. Nothing tests the motivating case (≥ 4 points where global quadratic and 3-point bracket disagree), nothing asserts `val_bpb` / `quadratic_*` after the coefficient shift-back, and S1/S2 are uncovered.

Verified correct: `active_matmul_parameters` for `GroupedMoE` matches a hand count; `estimate_flops == 6·(active + lm_head) + 12·L·H·hd·T`; dense plans give `capacity == active`; all six MoE flags passed by `_training_command` exist in `train_llama_engine.py`.

### 2.9 Distillation (uncommitted, new files)

**X1 — HIGH — [verified]** · `ohara/distill.py:140`; `examples/train_distill.py:173`
- Problem: `TeacherCache.vocab_size` is loaded and never compared to the student's `lm_head`. Default `vocab_size = len(tokenizer)` = 151,669 for Qwen3, while the cache indexes the teacher's 151,936-wide output. Any cached top-k index in `[151669, 151936)` makes `torch.gather` fail — CPU repro `index 7 is out of bounds for dimension 2 with size 4`; on CUDA a device-side assert mid-run.
- Fix: raise in `DistillTokenBinDataset.__init__` (pass student vocab) or in `train_distill.run` when `cache.vocab_size > vocab_size`; or fold out-of-range indices into the "other" bucket in `distillation_loss`.

**X2 — MEDIUM — [verified by trace]** · `examples/train_distill.py:293-302`
- Each micro-batch loss is divided by its own token count and summed with no `/accum`; `ohara/trainer.py:478` divides by the step-wide valid-token count. At defaults (accum 32, `--grad-clip-norm 1.0`) the gradient norm is ~32× the recipe's, so clipping engages every step and the borrowed nanochat LRs lose their meaning. Both A/B arms are affected equally, but wall-clock comparisons against `train_llama_engine` runs are invalid.
- Fix: `loss = loss / args.grad_accum_steps` (or normalise by step-wide tokens like `Trainer`), and default clip to 0 as `scaling_laws.py` does. Better: reuse `Trainer` with a custom loss hook instead of a second training loop.

**X3 — MEDIUM — [verified]** · `examples/train_distill.py:3-4, 212-225`
- The baseline arm only sees the cache's blocks if `--teacher-cache` is also passed to it; the docstring's baseline example omits it. On a toy corpus the baseline iterates 10 blocks and the distill arm 6 — different data populations, defeating the "only `--distill-alpha` differs" claim. (With the flag, block order is byte-identical between the two datasets.) Require `--teacher-cache` (or `--blocks`) for every arm and print `blocks=` in both.

**X4 — LOW — [verified]** · `ohara/distill.py:120, 146`
- `ignore_index` is accepted and never used (loss identical for `-1` vs `12345`), and the caller passes no `valid_mask`. Derive `valid_mask = targets != ignore_index` inside, or delete the parameter.

**X5 — LOW — [verified]** · `ohara/distill.py:241-242`
- Docstring says the cache "can be extended later by appending", but files open `"wb"`; rebuilding 6 → 8 blocks replaced the cache. Reword, or implement resume (`"ab"`, start at `metadata["blocks"]`).

**X6 — LOW — [plausible]** · `ohara/distill.py:281`
- `.float()` on full-vocab logits: 4 × 2048 × 151,936 fp32 = 5 GB plus 2.5 GB bf16 per batch. Do `topk` / `logsumexp` on bf16 (or chunk) before casting.

**X7 — LOW — [plausible]** · fp16 storage of raw logits has spacing 2^-6 at |logit| ≈ 30 (~1.5 % relative prob error); top-k sums can exceed 1 (clamped). Store `logit - logsumexp` (≤ 0) to use fp16's range better.

**X8 — LOW — [plausible]** · `DistillTokenBinDataset` never checks `cache.metadata["token_bin"]` / `tokens_covered` against the bin it is paired with; a cache built on another corpus would silently supply wrong teacher targets. Compare and raise.

**X9 — LOW — [plausible]** · `examples/train_distill.py:343` saves `model.state_dict()` of the DDP/compiled wrapper (`module.` / `_orig_mod.` prefixes). Unwrap before saving, or confirm downstream loaders strip prefixes.

**X10 — LOW** · `examples/ab_status.py`
- The headline verdict is one eval of ≈ 197k tokens; typical A/B deltas (~0.005 bpb) are within that noise and no interval is shown. Per-run cards show each run's latest bpb while the headline uses the last shared iter — easy to misread. Half-written `loss:` lines are recorded as truncated values for one tick and self-correct (not worth fixing).

**X11 — HIGH — [verified]** · `ohara/distill.py:139-144`
- Problem: `student_other = (1 - exp(student_logprobs).sum()).clamp_min(1e-9).log()` suffers FP32 cancellation when the student assigns almost all probability to the teacher's top-k. The clamp then removes the gradient needed to correct that excessive confidence.
- Reproduced: student logits `[20, 0]`, teacher probabilities `[0.5, 0.5]`, and top-k 1 produce loss 10.3616 instead of approximately 10, with gradient `[0, 1.03e-9]` instead of `[0.5, -0.5]`. The loss remains finite but its corrective gradient is effectively gone.
- Fix: compute the excluded-vocabulary logsumexp directly from the full student logits, subtract the full-vocabulary logsumexp, and handle the empty residual bucket when k equals the vocabulary size. Avoid subtraction of nearly equal probabilities.
- Test: compare both loss and gradient against explicit teacher cross-entropy for confident students (including `[20, 0]`), as well as ordinary logits, masking, and k = V. Finiteness alone is insufficient.

Test gap: `ohara/distill.py` has zero tests. Each of these is a ~15-line unit test: k = V loss equals full CE(teacher‖student); k < V equals the explicit (k+1)-way CE; cache block/position alignment matches the dataset; A/B block parity; vocab-mismatch rejection.

Verified in the original tested cases: `distillation_loss` equals full CE when k = V, equals the explicit (k+1)-way CE for k < V with ordinary logits, and has ~0 gradient at student == teacher. Saturated top-k cases remain finite but can have incorrect loss and gradients (X11); they are not verified correct. `build_teacher_cache` block/position alignment matches the dataset exactly; `train_distill`'s per-group LR loop is right.

### 2.10 Evaluation

**E1 — HIGH — [verified]** · `tests/test_core_eval.py:35-51`
- Problem: `NextTokenOracle` emits the input's own next token as argmax, so every choice row scores loss ≈ 0, all rows tie, and `mean_losses.index(min(...))` returns 0. With `gold=1` the identical setup scores 0.0. The MC/schema tests pass for any answer-span extraction, any sign error, or a swapped `gold`, as long as gold is 0 — which every fixture uses.
- Fix: a model whose logits depend only on the prefix (lookup table keyed on the last input token, or a tiny seeded `Llama`) and fixtures with `gold != 0`.

**E2 — MEDIUM — [verified]** · `ohara/core_eval.py:215-217`
- `rng.sample(available_indices, num_fewshot)` raises `Sample larger than population` when `len(data) - 1 < num_fewshot`. 12 of 22 CORE tasks are 10-shot, so `examples/core_eval.py --max-per-task 5` dies on the second task (jeopardy). Use `min(num_fewshot, len(available))` or raise a clear error early.

**E3 — MEDIUM — [verified by trace]** · `examples/core_eval.py:107`
- `model.load_state_dict(state, strict=False)` swallows missing/unexpected keys (wrong `--moe-*` flags, vocab mismatch, another architecture) and evaluates random weights in those tensors, producing a plausible CORE number. Use `strict=True`, or inspect `_IncompatibleKeys` and abort on anything other than `freq_cos` / `freq_sin`.

**E4 — LOW — [verified]** · `ohara/core_eval.py:238`
- Truncation guard `start_idx - trim < 0` allows `start == 0`, but scoring reads `losses[row, si-1:ei-1]` → `[-1:…]`, an empty slice → NaN → `index(min([nan, …]))` returns 0. Only when the continuation alone fills the window; nanochat asserts here. Use `< 1`.
- Independent reproduction also confirms a hard failure for language-modeling tasks: a two-token continuation filling a two-token context window compares an empty prediction slice with two targets and raises `RuntimeError: The size of tensor a (0) must match the size of tensor b (2)`. Add both MC and language-modeling boundary tests.

**E5 — LOW — [verified]** · `ohara/perplexity.py:89`
- Docstring says "scoring every token once", which is false when `stride == sequence_length` (window-first tokens are never predicted: N = 16, L = 4, stride 4 → 12 predicted, not 15). Same as HF's reference; fix the docstring.

**E6 — LOW — [verified by trace]** · `ohara/core_eval.py:339-343`
- Writes `tmpzip` to disk and then extracts from `io.BytesIO(payload)` anyway (dead write). `./eval_bundle` (26 MB default download) is not in `.gitignore`. Remove the write; add `eval_bundle/` to `.gitignore`.

**E7 — LOW** · `examples/core_eval.py:119` vs `examples/evaluate_perplexity.py:42`
- One uses deprecated `torch_dtype=`, the other `dtype=`; transformers 5.x treats `torch_dtype` as BC-only. Unify on `dtype=`.

**E8 — LOW** · `examples/evaluate_perplexity.py:38-39`
- `--backend ohara` only loads `Qwen3`; an ohara `Llama` exported by `export_safetensors.py` cannot be evaluated. Dispatch on `config.json`'s `model_type`.

**E9 — LOW** · `examples/export_safetensors.py:103-106`; `ohara/models/llama.py:500-505`
- `architectures: ["Llama"]`, `model_type: "ohara_llama"` and ohara-native key names are not HF-loadable; README's "standard config.json + model.safetensors" means layout-standard only. Tied embeddings are exported as two full copies; no dtype option so fp32 master weights are published at 2× size. No test covers this script. Document the limitation or add an HF key-mapping export; add `--dtype`.

**E10 — LOW — [plausible]** · `ohara/core_eval.py:190-203`
- `forward_model` computes CE on whatever dtype the model emits; a bf16 model without autocast would tie choices more often. Add `.float()` before CE.

Verified correct: the CORE prompt fix in `7f8bbc4` is complete (1200/1200 prefix checks across four tasks with the gpt-neo BPE); few-shot RNG, shuffle seed, BOS-as-pad, `si-1:ei-1` scoring and centering match nanochat; all 22 `core.yaml` labels exist in `eval_meta_data.csv`. Perplexity uses `exp(Σ nll / count)`, correct shift, no padding, fp32 logits, `inference_mode`, train-mode restored.

### 2.11 Utils, packaging, docs

**U1 — HIGH — [verified]** · `ohara/utils/__init__.py:2` → `ohara/utils/info.py:2, 4`
- Problem: `ohara.utils` eagerly imports `info.py`, which imports `lightning`. `ohara/trainer.py:16` (`BetterCycle`), `ohara/models/phi.py:20`, `examples/core_eval.py:15`, `examples/evaluate_perplexity.py:13`, and `examples/phi_inference.py` all fail with `ModuleNotFoundError: lightning` in any environment without it, even though none of them use Lightning. This blocked collection of five test modules in the first test run of this review. `import ohara` itself is fine (0.05 s).
- Fix: make `model_summary` a lazy import inside the function, or remove `info` from `ohara/utils/__init__.py`. Consider making `lightning` an optional extra.

**U2 — MEDIUM — [verified]** · `pyproject.toml:11-24, 49-52`
- `rich`, `tqdm`, `tensorboard` are declared but imported nowhere in `ohara/` or `examples/` (`tensorboard` drags in protobuf/grpcio/werkzeug); `numpy` is imported (`tokenbin.py:26`, `distill.py:39`) but undeclared. `constraint-dependencies = ["torch==2.10.0"]` is a machine-specific pin ("the shared CUDA 12 environment") committed to the project; this box's global venv is already on torch 2.13+cu130.
- Fix: drop the three unused deps; add `numpy`; move the torch constraint to an untracked `uv.toml` or document it as a deliberate project-wide pin.

**U3 — MEDIUM — [verified]** · `README.md`; `runs/speedrun.sh`
- `README.md:146` links `scaling_results/climbmix_full/`, which is gitignored and absent (dead link for cloners). `README.md:22` omits `qwen3` / `transformer` from the model list. `README.md:58-68` and `runs/speedrun.sh:82, 145, 177, 194` run bare `python` / `torchrun`, contradicting AGENTS.md. The CORE / perplexity / export entrypoints are undocumented.
- Fix: update the README; prefix commands with `uv run`.

**U4 — LOW** · `ohara/utils/load.py:7`
- Hardcodes `~/.cache/huggingface/hub/`, ignoring `HF_HOME` / `HF_HUB_CACHE`; `get_model_path` picks `revisions[0]` arbitrarily and is unused. Use `huggingface_hub.constants.HF_HUB_CACHE` or delete.

**U5 — LOW** · `ohara/utils/tools.py:32, 83`
- `torch.backends.mps.is_built()` should be `is_available()`; docstring claims an `AssertionError` the code never raises. `BetterCycle.close()` only drops a reference and relies on refcounting for worker shutdown.

**U6 — LOW** · `ohara/utils/svd.py:21`; `ohara/utils/rand.py:22`
- `svd` names the third output `V` though it is `Vh` (math correct; unused); `rand.py` uses the global `random`, so a seeded script gets the same run name every time. Use a private `random.Random()`.

**U7 — LOW** · `.gitignore`
- Add `eval_bundle/` (see E6). `ohara.egg-info` and `runs/` outputs are correctly untracked.

---

## 3. Tests to add (consolidated)

| Test | Catches |
|---|---|
| `tests/test_pscan.py`: pscan fwd/bwd vs sequential scan, L ∈ {1,2,3,7,8,9,16,17,33}, float64 | M1 regression |
| `tests/test_linear_rnn.py`: `RG_LRU` vs Python time loop | M1 |
| `test_optimizers_hypersphere.py` parametrised with `init_style="nanochat"` | T1 |
| `test_modules.py`: LoRA/DoRA wrap preserves outputs (bias and no-bias) | A1, A2 |
| `tests/test_distributed.py` (gloo, 2 procs): DDP dense + MoE two steps; TP=2 forward matches single process; `engine.save` under TP reloads | R1, R2, R3, R4, M5 |
| `test_moe_grouped.py`: CPU grouped-vs-reference incl. gradients and bf16; k=1 raises; alignment raises; `shared_exclusive` through grouped dispatch; move `__main__` to end | M2, M3, M9, M10 |
| `test_chat.py`: multi-byte UTF-8 streaming equals `decode()`; system-turn trimming keeps role alternation; non-extreme sampling logits; server `TypeError` / NaN handling | C1, C2, C3, C7 |
| `test_tokenbin.py`: size/sidecar mismatch raises; `data_rank` range; real id > 65535 round-trip; shuffle+shard disjointness; seed determinism | P2, P6 |
| `test_dataset`: `train.json` sibling not selected as corpus | P1 |
| `test_models.py`: tied checkpoint missing both embeddings raises; GQA cached decode multi-step batch > 1; Qwen3 vs HF equivalence; `Transformer` activation differs; Gemma `intermediate_size` configurable | D1, D2, M11 |
| `test_core_eval.py`: prefix-dependent oracle with `gold != 0`; `--max-per-task` smaller than shots | E1, E2 |
| `tests/test_distill.py`: k=V equals CE; (k+1)-way equivalence; cache alignment; vocab mismatch raises; A/B block parity | X1, X3 |
| `test_scaling.py`: fallback rows excluded from power law; degenerate duplicate-params grid does not abort | S1, S2 |
| `test_train_sft.py`: `run()` with padded vocab and `--evaluate-bpb` | T3 |
| `test_trainer.py`: `--resume` reproduces uninterrupted trajectory (RNG, `start_block`, optimizer) | resume correctness |
| Two-rank masked SFT update equals a single-process concatenated-batch update | T12 |
| Muon/MuonH optimizer step with an unused loop-MoE expert | T13 |
| `CausalAttention` SDPA and reference paths honor the same explicit mask | M16 |
| Real HF streaming dataset with one/multiple source shards and two DataLoader workers | P16 |
| ByteLevel token-byte lookup for a split UTF-8 character | P10 |
| Sweep launched outside the repository resolves child and parent paths identically | S5 |
| Distillation loss and gradient match explicit CE for saturated student logits | X11 |
| CORE continuation exactly fills the context window, for MC and language modeling | E4 |

---

## 4. Uncommitted work: commit readiness

Working tree at review time: modified `examples/pretokenize_corpus.py`, `examples/scaling_laws.py`, `examples/train_llama_engine.py`, `ohara/models/llama.py`, `ohara/modules/moe_grouped.py`, `ohara/scaling.py`, `ohara/tokenbin.py`, and three tests; new `ohara/distill.py`, `examples/train_distill.py`, `examples/distill_cache.py`, `examples/ab_status.py`.

| Piece | Status | Blockers before commit |
|---|---|---|
| `moe_shared_exclusive` (llama.py, moe_grouped.py, train_llama_engine.py, test) | Model side correct | M7 (`config_from_state_dict` kwarg + test), M9 (test class after `__main__`) |
| `tokenbin.py` dtype widening + tests | Correct for happy path | None required; P1/P2 are pre-existing but this is the natural place to fix them |
| `pretokenize_corpus.py` | Works | P4 (reuse check), P5 (budget desync) recommended |
| `scaling.py` / `scaling_laws.py` / `test_scaling.py` | Close | S1, S2 (they change what `analyze` reports) |
| `distill.py` / `train_distill.py` | Not ready | X1, X2, X3, X11, plus loss/gradient-equivalence and cache-alignment tests |
| `distill_cache.py`, `ab_status.py` | Fine as-is | — |

Suggested split into commits: (1) shared-exclusive plumbing with M7/M9; (2) tokenbin widening with P1/P2/P6; (3) scaling fixes with S1/S2/S5; (4) distillation once X1–X3, X11, and tests land.

---

## 5. Adversarial verification of the Codex additions (2026-09-08, later)

The six findings added by the independent Codex pass (T12, T13, M16, P16, S5,
X11) were re-checked against the working tree. No source changes were present
at the time of this check: the `git diff --stat` of the working tree was
identical to the one the original review started from, so none of the findings
in this document have been fixed yet.

| ID | Codex claim | Verdict | Severity adjustment |
|---|---|---|---|
| T12 | DDP averages rank means instead of the global token mean | **Confirmed by trace.** `trainer.py:441-446` divides by the local valid-token count, then DDP averages. Exact for token-bin pretraining (every rank has the same count); biased only for packed SFT with unequal masks. | Keep MEDIUM for SFT; note it is a no-op for pretraining. Same behaviour as nanochat and pre-2024 HF Trainer. |
| T13 | Muon raises when a loop-MoE expert receives no tokens | **Reproduced.** Tiny Llama, 4 experts top-2, `moe_grouped=False`, one token, `build_muon_adamw`: first `opt.step()` raises "a Muon group has missing gradients". The grouped path (`moe_grouped=True`) steps fine because empty experts get zero gradients inside the stacked tensor. | Keep MEDIUM. The cleanest fix is the same as M5/R2: run empty experts on the empty slice so every parameter joins the graph, which also fixes the DDP failure. |
| M16 | `CausalAttention` SDPA path ignores the caller's mask | **Confirmed by reading** `attention.py:84-91` (`attn_mask=None`). But `ohara/modules/attention.py` is not used by any model, example, or script in the library; only `tests/test_modules.py` imports it. | Downgrade to LOW. Fix or delete the module (see D14). |
| P16 | Worker sharding applied twice on top of HF streaming | **Reproduced, and worse than stated.** One JSONL, `num_workers=2`: worker 1 receives no HF shard and raises `no usable documents found`. Two JSONL files, `num_workers=2`: only 20 of 40 documents are ever yielded across 400 blocks (multiple epochs), because each worker keeps only `1/num_workers` of the shard HF already gave it. Half the corpus is silently never trained on. Any `--num-workers > 1` run on the streaming path is affected. | **Upgrade to HIGH.** Rank sharding via modulo is fine (HF does not know about ranks unless `split_dataset_by_node` is used); only the worker factor must be removed from `shard_id` / `num_shards`, and worker-local exhaustion must not be treated as an empty corpus. The P11 statement that every worker reads the whole stream is wrong for HF iterables; P11 remains true for the rank dimension only. |
| S5 | `subprocess.run(cwd=PROJECT_ROOT)` changes relative-path meaning | **Confirmed by reading.** `_training_command` passes `args.dataset`, `--result-json`, and `--token-bytes-cache` unresolved; the parent then reads `result_json` relative to its own cwd. Only bites when the sweep is launched from outside the repository with relative paths, which the README does not do. | Downgrade to LOW. Fix is one line: `Path(...).resolve()` on the three path arguments before building the command. |
| X11 | Distill loss loses its corrective gradient when the student saturates the top-k | **Reproduced.** Student `[20, 0]`, teacher `[0.5, 0.5]`, k=1: loss 10.3616 vs exact 10.0, gradient `[0, 1e-9]` vs exact `[0.5, -0.5]`. Degradation starts earlier than saturation: at `[12, 0]` the gradient is already `[0.5, -0.537]`. | Keep HIGH for correctness, but note the practical impact is bounded: the lost term is weighted by the teacher's residual mass, so it matters exactly when the student is overconfident and the teacher is not. Fix: `student_other = logsumexp(student_logits.scatter(-1, teacher_index, -inf)) - student_logsumexp`, with a guard for `k == V` where the residual bucket is empty (`other_prob == 0` must not multiply `-inf`). |

Net effect on the priority table in §1: P16 moves up into the top ten (it is a
silent data-loss bug on a default-ish code path); M16 and S5 move to the Low
tier; the other three stay where Codex placed them.

Reproduction commands used here ran with the system torch 2.13 and datasets
4.6.1, not the locked torch 2.10, for the reasons given in §0.

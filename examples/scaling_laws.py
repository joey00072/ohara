from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from ohara.scaling import (
    ScalingPlan,
    analyze_scaling_results,
    append_result_csv,
    load_result_csv,
    plan_scaling_run,
    write_scaling_svg,
)
from ohara.tokenizer import get_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT.joinpath("examples", "train_llama_engine.py")
SCALING_RECIPE_VERSION = 3


def _int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depths", type=_int_list, default=_int_list("10,12,14,16,18,20"))
    parser.add_argument(
        "--flops-budgets",
        type=_float_list,
        default=_float_list("1e18,2.15e18,4.64e18,1e19"),
    )
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--seq-len", type=int, default=2_048)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn-multiplier", type=float, default=8 / 3)
    parser.add_argument("--ffn-multiple-of", type=int, default=256)
    parser.add_argument("--reference-depth", type=int, default=12)
    parser.add_argument("--reference-batch-size", type=int, default=2**19)
    parser.add_argument("--reference-tokens-per-param", type=float, default=12.0)
    parser.add_argument("--batch-exponent", type=float, default=0.383)
    parser.add_argument(
        "--base-matrix-learning-rate",
        "--base-learning-rate",
        dest="base_learning_rate",
        type=float,
        default=0.02,
    )
    parser.add_argument("--base-embedding-learning-rate", type=float, default=0.3)
    parser.add_argument("--base-unembedding-learning-rate", type=float, default=0.008)
    parser.add_argument("--base-scalar-learning-rate", type=float, default=0.5)
    parser.add_argument("--base-weight-decay", type=float, default=0.28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, run, and analyze nanochat-style iso-FLOP Llama sweeps"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print run configurations without training")
    _add_plan_arguments(plan_parser)
    plan_parser.add_argument("--output-json", default=None)

    run_parser = subparsers.add_parser("run", help="execute the fixed-FLOP sweep")
    _add_plan_arguments(run_parser)
    run_parser.add_argument(
        "--dataset",
        default="./data/scaling_corpus",
        help="staged corpus directory (prepare_scaling_data.py defaults to NanoChat ClimbMix)",
    )
    run_parser.add_argument("--dataset-config", default=None)
    run_parser.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125m")
    run_parser.add_argument("--tokenizer-local-files-only", action="store_true")
    run_parser.add_argument("--text-column", default="text")
    run_parser.add_argument("--train-split", default="train")
    run_parser.add_argument("--validation-split", default="validation")
    run_parser.add_argument(
        "--optimizer",
        choices=("muon", "muonh"),
        default="muon",
        help=(
            "muon: nanochat-style Muon+AdamW. muonh: the hyperspherical (constant-norm) "
            "MuonH+AdamH recipe, whose single relative learning rate the trainer derives "
            "as sqrt(matrix_lr * weight_decay) from the same planned nanochat recipe"
        ),
    )
    run_parser.add_argument(
        "--hypersphere-learning-rate",
        type=float,
        default=None,
        help=(
            "fixed relative step size for --optimizer muonh, held constant across the "
            "whole depth grid. Omit to let the trainer derive sqrt(matrix_lr * weight_decay) "
            "per run instead, which ports an additive recipe but does not hold the "
            "hyperspherical rate constant as the paper's transferability claim expects"
        ),
    )
    run_parser.add_argument("--eval-batches", type=int, default=20)
    run_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=40,
        help="ceiling on LR warmup iterations; scaled down for short runs (see --warmup-fraction)",
    )
    run_parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="warmup/momentum-warmup iters are min(ceiling, warmup_fraction * num_iterations)",
    )
    run_parser.add_argument(
        "--muon-momentum-warmup-iters",
        type=int,
        default=400,
        help="ceiling on Muon momentum warmup iterations; scaled down for short runs",
    )
    run_parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    run_parser.add_argument("--final-lr-fraction", type=float, default=0.05)
    run_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16_mixed", "bf16_mixed", "bf16_true"),
        default="bf16_mixed",
    )
    run_parser.add_argument("--num-workers", type=int, default=0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--results-dir", default="./scaling_results")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--max-corpus-epochs", type=float, default=3.0)
    run_parser.add_argument("--allow-repeated-epochs", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="fit iso-FLOP curves")
    analyze_parser.add_argument("--results-file", default="./scaling_results/results.csv")
    analyze_parser.add_argument("--metric", default="val_bpb")
    analyze_parser.add_argument("--output-json", default=None)
    analyze_parser.add_argument("--output-svg", default=None)
    return parser.parse_args()


def build_plans(args: argparse.Namespace) -> list[ScalingPlan]:
    plans = []
    for budget in args.flops_budgets:
        for depth in args.depths:
            plans.append(
                plan_scaling_run(
                    depth,
                    vocab_size=args.vocab_size,
                    flops_budget=budget,
                    sequence_length=args.seq_len,
                    device_batch_size=args.device_batch_size,
                    world_size=args.nproc_per_node,
                    total_batch_size=args.total_batch_size,
                    aspect_ratio=args.aspect_ratio,
                    head_dim=args.head_dim,
                    ffn_multiplier=args.ffn_multiplier,
                    ffn_multiple_of=args.ffn_multiple_of,
                    reference_depth=args.reference_depth,
                    reference_batch_size=args.reference_batch_size,
                    reference_tokens_per_param=args.reference_tokens_per_param,
                    batch_exponent=args.batch_exponent,
                    base_learning_rate=args.base_learning_rate,
                    base_embedding_learning_rate=args.base_embedding_learning_rate,
                    base_unembedding_learning_rate=args.base_unembedding_learning_rate,
                    base_scalar_learning_rate=args.base_scalar_learning_rate,
                    base_weight_decay=args.base_weight_decay,
                )
            )
    return plans


def print_plans(plans: list[ScalingPlan]) -> None:
    print(
        "budget,depth,dim,params,effective_params,batch,accum,iters,tokens,"
        "actual_flops,matrix_lr,embedding_lr,unembedding_lr,scalar_lr,wd"
    )
    for plan in plans:
        print(
            f"{plan.flops_budget:.6g},{plan.depth},{plan.hidden_size},"
            f"{plan.params_total},{plan.params_effective},{plan.total_batch_size},"
            f"{plan.grad_accum_steps},{plan.num_iterations},{plan.tokens_trained},"
            f"{plan.actual_training_flops:.6g},{plan.matrix_learning_rate:.6g},"
            f"{plan.embedding_learning_rate:.6g},"
            f"{plan.unembedding_learning_rate:.6g},"
            f"{plan.scalar_learning_rate:.6g},"
            f"{plan.weight_decay:.6g}"
        )


def _scaled_warmup_iters(ceiling: int, num_iterations: int, fraction: float) -> int:
    """Cap a warmup length so it never dominates the shortest runs in a sweep.

    A sweep's iso-FLOP grid spans depths whose num_iterations can differ by
    more than 100x. A single fixed warmup/momentum-warmup step count is
    either negligible on the longest runs or a large fraction of the
    shortest ones, so points on the same iso-FLOP curve end up trained under
    different effective schedules.
    """
    return max(1, min(ceiling, round(fraction * num_iterations)))


def _training_command(
    args: argparse.Namespace,
    plan: ScalingPlan,
    result_json: Path,
) -> list[str]:
    warmup_iters = _scaled_warmup_iters(
        args.warmup_steps, plan.num_iterations, args.warmup_fraction
    )
    momentum_warmup_iters = _scaled_warmup_iters(
        args.muon_momentum_warmup_iters, plan.num_iterations, args.warmup_fraction
    )
    if args.nproc_per_node > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(TRAIN_SCRIPT),
        ]
    else:
        command = [sys.executable, str(TRAIN_SCRIPT)]

    command.extend(
        [
            "--dataset",
            args.dataset,
            "--tokenizer",
            args.tokenizer,
            "--text-column",
            args.text_column,
            "--train-split",
            args.train_split,
            "--validation-split",
            args.validation_split,
            "--seq-len",
            str(plan.sequence_length),
            "--batch-size",
            str(plan.device_batch_size),
            "--grad-accum-steps",
            str(plan.grad_accum_steps),
            "--max-iters",
            str(plan.num_iterations),
            "--eval-every",
            str(plan.num_iterations),
            "--eval-batches",
            str(args.eval_batches),
            "--save-every",
            "0",
            "--optimizer",
            args.optimizer,
            "--matrix-learning-rate",
            str(plan.matrix_learning_rate),
            "--embedding-learning-rate",
            str(plan.embedding_learning_rate),
            "--unembedding-learning-rate",
            str(plan.unembedding_learning_rate),
            "--scalar-learning-rate",
            str(plan.scalar_learning_rate),
            "--weight-decay",
            str(plan.weight_decay),
            "--warmup-iters",
            str(warmup_iters),
            "--muon-momentum-warmup-iters",
            str(momentum_warmup_iters),
            "--hidden-size",
            str(plan.hidden_size),
            "--intermediate-size",
            str(plan.intermediate_size),
            "--num-layers",
            str(plan.depth),
            "--num-heads",
            str(plan.num_heads),
            "--no-weight-tying",
            "--init-style",
            "nanochat",
            "--lr-schedule",
            "wsd",
            "--warmdown-ratio",
            str(args.warmdown_ratio),
            "--final-lr-fraction",
            str(args.final_lr_fraction),
            "--grad-clip-norm",
            "0",
            "--evaluate-bpb",
            "--token-bytes-cache",
            str(Path(args.results_dir).joinpath("token_bytes.pt")),
            "--precision",
            args.precision,
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--tp",
            "1",
            "--result-json",
            str(result_json),
            "--scaling-depth",
            str(plan.depth),
            "--flops-budget",
            str(plan.flops_budget),
        ]
    )
    if args.dataset_config is not None:
        command.extend(["--dataset-config", args.dataset_config])
    if args.tokenizer_local_files_only:
        command.append("--tokenizer-local-files-only")
    if args.optimizer == "muonh" and args.hypersphere_learning_rate is not None:
        command.extend(
            ["--hypersphere-learning-rate", str(args.hypersphere_learning_rate)]
        )
    return command


def _experiment_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Return fields that must remain fixed within one fitted scaling sweep."""
    return {
        "recipe_version": SCALING_RECIPE_VERSION,
        "optimizer": "muon_adamw" if args.optimizer == "muon" else "muonh_adamh",
        "hypersphere_learning_rate": args.hypersphere_learning_rate,
        "initialization": "nanochat",
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "tokenizer": args.tokenizer,
        "tokenizer_local_files_only": args.tokenizer_local_files_only,
        "text_column": args.text_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "sequence_length": args.seq_len,
        "vocab_size": args.vocab_size,
        "device_batch_size": args.device_batch_size,
        "world_size": args.nproc_per_node,
        "total_batch_size": args.total_batch_size,
        "aspect_ratio": args.aspect_ratio,
        "head_dim": args.head_dim,
        "ffn_multiplier": args.ffn_multiplier,
        "ffn_multiple_of": args.ffn_multiple_of,
        "reference_depth": args.reference_depth,
        "reference_batch_size": args.reference_batch_size,
        "reference_tokens_per_param": args.reference_tokens_per_param,
        "batch_exponent": args.batch_exponent,
        "base_matrix_learning_rate": args.base_learning_rate,
        "base_embedding_learning_rate": args.base_embedding_learning_rate,
        "base_unembedding_learning_rate": args.base_unembedding_learning_rate,
        "base_scalar_learning_rate": args.base_scalar_learning_rate,
        "base_weight_decay": args.base_weight_decay,
        "warmup_steps_ceiling": args.warmup_steps,
        "muon_momentum_warmup_iters_ceiling": args.muon_momentum_warmup_iters,
        "warmup_fraction": args.warmup_fraction,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_fraction": args.final_lr_fraction,
        "eval_batches": args.eval_batches,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }


def _prepare_results_manifest(args: argparse.Namespace, results_dir: Path) -> None:
    manifest_path = results_dir.joinpath("experiment.json")
    expected = _experiment_manifest(args)
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != expected:
            differing = sorted(
                key
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"{manifest_path} does not match this run ({', '.join(differing)}); "
                "use a fresh --results-dir"
            )
        return
    if results_dir.joinpath("results.csv").exists():
        raise ValueError(
            f"{results_dir} has results but no experiment manifest; use a fresh "
            "--results-dir so incompatible runs cannot be mixed"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus_token_budget(dataset: str) -> int | None:
    """Read a token count staged by prepare_scaling_data.py's --tokenizer pass.

    Returns None when the dataset is not a locally staged corpus, or was
    staged without --tokenizer (no stats file, or a stats file with no
    token count), in which case the epoch-budget check is skipped.
    """
    stats_path = Path(dataset).joinpath("stats.train.json")
    if not stats_path.exists():
        return None
    with open(stats_path, encoding="utf-8") as handle:
        stats = json.load(handle)
    tokens = stats.get("tokens")
    return int(tokens) if tokens is not None else None


def _check_corpus_epoch_budget(args: argparse.Namespace, plans: list[ScalingPlan]) -> None:
    """Fail loudly when a sweep would repeatedly re-epoch a small staged corpus.

    StreamingTextDataset loops its source stream forever with no warning, so
    a sweep whose iso-FLOP token horizons exceed the corpus size silently
    trains on memorization rather than the intended token budget, without
    any signal in val_bpb that this happened.
    """
    corpus_tokens = _corpus_token_budget(args.dataset)
    if corpus_tokens is None:
        return
    max_tokens = max(plan.tokens_trained for plan in plans)
    max_epochs = max_tokens / corpus_tokens
    print(f"corpus tokens={corpus_tokens:,}; sweep max epochs over corpus={max_epochs:.2f}")
    if max_epochs > args.max_corpus_epochs and not args.allow_repeated_epochs:
        raise ValueError(
            f"sweep would repeat the staged corpus at {args.dataset!r} up to "
            f"{max_epochs:.1f}x, above --max-corpus-epochs={args.max_corpus_epochs}; "
            "stage more documents with prepare_scaling_data.py, shrink --depths/"
            "--flops-budgets, or pass --allow-repeated-epochs to proceed anyway"
        )


def _numeric_result_row(result: Mapping[str, object]) -> dict[str, object]:
    """Drop non-numeric result fields before they reach the results CSV.

    load_result_csv coerces every column to float, so a descriptive string
    field in the training result (such as the optimizer name) would make the
    whole sweep unreadable at analyze time. The optimizer is already recorded
    numerically alongside it, and authoritatively in experiment.json.
    """
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def run_sweep(args: argparse.Namespace, plans: list[ScalingPlan]) -> None:
    results_dir = Path(args.results_dir)
    results_file = results_dir.joinpath("results.csv")
    if not args.dry_run:
        tokenizer = get_tokenizer(
            hf_name=args.tokenizer,
            prefer_hf=True,
            local_files_only=args.tokenizer_local_files_only,
        )
        actual_vocab_size = len(tokenizer)
        del tokenizer
        if actual_vocab_size != args.vocab_size:
            raise ValueError(
                f"--vocab-size={args.vocab_size} does not match tokenizer size "
                f"{actual_vocab_size}; fix the plan before launching GPU runs"
            )
        _check_corpus_epoch_budget(args, plans)
        _prepare_results_manifest(args, results_dir)
    completed = set()
    if results_file.exists():
        existing_rows = load_result_csv(results_file)
        expected_muon_flag = float(args.optimizer == "muon")
        if any(row.get("optimizer_muon", 0.0) != expected_muon_flag for row in existing_rows):
            raise ValueError(
                f"{results_file} was produced with a different optimizer recipe than "
                f"--optimizer={args.optimizer}; use a fresh --results-dir rather than "
                "mixing optimizer recipes in one scaling fit"
            )
        completed = {
            (float(row["flops_budget"]), int(row["depth"])) for row in existing_rows
        }

    for plan in plans:
        key = (float(plan.flops_budget), plan.depth)
        if key in completed:
            print(f"skip budget={plan.flops_budget:.6g} depth={plan.depth}: already complete")
            continue
        tag = f"flops_{plan.flops_budget:.6g}_d{plan.depth}".replace("+", "")
        result_json = results_dir.joinpath(f"{tag}.json")
        command = _training_command(args, plan, result_json)
        print("run:", " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        with open(result_json, encoding="utf-8") as handle:
            result = json.load(handle)
        if int(result["params_effective"]) != plan.params_effective:
            raise RuntimeError("trained model does not match the planned parameter count")
        if int(result["total_batch_size"]) != plan.total_batch_size:
            raise RuntimeError("training result does not match planned total_batch_size")
        recorded_optimizer = result.get("optimizer")
        if recorded_optimizer is not None and recorded_optimizer != args.optimizer:
            raise RuntimeError(
                f"training result used optimizer {recorded_optimizer!r}, not the planned "
                f"{args.optimizer!r}"
            )
        if result.get("optimizer_muon") != float(args.optimizer == "muon"):
            raise RuntimeError("training result did not use the planned optimizer recipe")
        if result.get("initialization_nanochat") != 1.0:
            raise RuntimeError("training result did not use the planned nanochat initialization")
        for field in (
            "matrix_learning_rate",
            "embedding_learning_rate",
            "unembedding_learning_rate",
            "scalar_learning_rate",
            "weight_decay",
            "flops_per_token",
        ):
            if not math.isclose(float(result[field]), float(getattr(plan, field))):
                raise RuntimeError(f"training result does not match planned {field}")
        append_result_csv(results_file, _numeric_result_row(result))
        completed.add(key)
        print(f"recorded {results_file}: budget={plan.flops_budget:.6g} depth={plan.depth}")


def analyze(args: argparse.Namespace) -> None:
    rows = load_result_csv(args.results_file)
    analysis = analyze_scaling_results(rows, metric=args.metric)
    if not analysis["optimums"]:
        raise RuntimeError("need at least three valid depths for each FLOP budget")
    payload = json.dumps(analysis, indent=2)
    print(payload)
    if args.output_json is not None:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if args.output_svg is not None:
        write_scaling_svg(
            args.output_svg,
            rows,
            analysis,
            metric=args.metric,
        )


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        analyze(args)
        return

    if args.command == "run":
        if (
            args.eval_batches < 1
            or args.num_workers < 0
            or args.warmup_steps < 0
            or args.muon_momentum_warmup_iters < 0
        ):
            raise ValueError("evaluation, worker, and warmup values are invalid")
        if not 0 < args.warmup_fraction <= 1:
            raise ValueError("warmup-fraction must be in (0, 1]")
        if not 0 <= args.warmdown_ratio <= 1:
            raise ValueError("warmdown-ratio must be in [0, 1]")
        if not 0 <= args.final_lr_fraction <= 1:
            raise ValueError("final-lr-fraction must be in [0, 1]")
        if args.max_corpus_epochs <= 0:
            raise ValueError("max-corpus-epochs must be positive")

    plans = build_plans(args)
    print_plans(plans)
    if args.command == "plan":
        if args.output_json is not None:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps([plan.to_dict() for plan in plans], indent=2) + "\n",
                encoding="utf-8",
            )
        return
    run_sweep(args, plans)


if __name__ == "__main__":
    main()

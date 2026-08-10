from __future__ import annotations

import csv
import html
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from ohara.models.llama import Config, Llama


def _round_up(value: int, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be positive")
    return multiple * ((value + multiple - 1) // multiple)


def _nearest_power_of_two(value: float) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 2 ** round(math.log2(value))


def llama_config_for_depth(
    depth: int,
    *,
    vocab_size: int,
    sequence_length: int = 2_048,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    ffn_multiplier: float = 8 / 3,
    ffn_multiple_of: int = 256,
    dropout: float = 0.0,
    weight_tying: bool = False,
) -> Config:
    """Derive every Llama shape from the single scaling dial, model depth."""
    if depth < 1:
        raise ValueError("depth must be positive")
    if vocab_size < 2 or sequence_length < 2:
        raise ValueError("vocab_size and sequence_length must be at least 2")
    if aspect_ratio < 1 or head_dim < 2 or head_dim % 2:
        raise ValueError("aspect_ratio must be positive and head_dim must be positive and even")
    if ffn_multiplier <= 0:
        raise ValueError("ffn_multiplier must be positive")

    hidden_size = _round_up(depth * aspect_ratio, head_dim)
    intermediate_size = _round_up(
        math.ceil(ffn_multiplier * hidden_size),
        ffn_multiple_of,
    )
    return Config(
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=hidden_size // head_dim,
        num_key_value_heads=hidden_size // head_dim,
        num_hidden_layers=depth,
        dropout=dropout,
        multiple_of=ffn_multiple_of,
        weight_tying=weight_tying,
        init_style="nanochat",
    )


def model_scaling_stats(config: Config) -> tuple[dict[str, int], float]:
    """Count a potentially large model without allocating its parameters."""
    with torch.device("meta"):
        model = Llama(config)
    return model.num_scaling_params(), model.estimate_flops(config.max_sequence_length)


@dataclass(frozen=True)
class ScalingPlan:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    sequence_length: int
    params_token_embeddings: int
    params_lm_head: int
    params_transformer: int
    params_norms_and_scalars: int
    params_total: int
    params_effective: int
    flops_per_token: float
    flops_budget: float
    total_batch_size: int
    device_batch_size: int
    world_size: int
    grad_accum_steps: int
    num_iterations: int
    tokens_trained: int
    actual_training_flops: float
    tokens_per_effective_param: float
    learning_rate: float
    matrix_learning_rate: float
    embedding_learning_rate: float
    unembedding_learning_rate: float
    scalar_learning_rate: float
    weight_decay: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def plan_scaling_run(
    depth: int,
    *,
    vocab_size: int,
    flops_budget: float | None = None,
    target_param_data_ratio: float | None = None,
    sequence_length: int = 2_048,
    device_batch_size: int = 32,
    world_size: int = 1,
    total_batch_size: int | None = None,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    ffn_multiplier: float = 8 / 3,
    ffn_multiple_of: int = 256,
    reference_depth: int = 12,
    reference_batch_size: int = 2**19,
    reference_tokens_per_param: float = 12.0,
    batch_exponent: float = 0.383,
    base_learning_rate: float = 0.02,
    base_embedding_learning_rate: float = 0.3,
    base_unembedding_learning_rate: float = 0.008,
    base_scalar_learning_rate: float = 0.5,
    base_weight_decay: float = 0.28,
) -> ScalingPlan:
    """Plan one fixed-compute or compute-optimal run using nanochat-style rules."""
    if flops_budget is None and target_param_data_ratio is None:
        raise ValueError("provide flops_budget or target_param_data_ratio")
    if flops_budget is not None and flops_budget <= 0:
        raise ValueError("flops_budget must be positive")
    if target_param_data_ratio is not None and target_param_data_ratio <= 0:
        raise ValueError("target_param_data_ratio must be positive")
    if device_batch_size < 1 or world_size < 1:
        raise ValueError("device_batch_size and world_size must be positive")
    if reference_batch_size < 1 or reference_tokens_per_param <= 0:
        raise ValueError("reference scaling values must be positive")
    if (
        any(
            value <= 0
            for value in (
                base_learning_rate,
                base_embedding_learning_rate,
                base_unembedding_learning_rate,
                base_scalar_learning_rate,
            )
        )
        or base_weight_decay < 0
    ):
        raise ValueError("optimizer values are invalid")

    config = llama_config_for_depth(
        depth,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        ffn_multiplier=ffn_multiplier,
        ffn_multiple_of=ffn_multiple_of,
    )
    counts, flops_per_token = model_scaling_stats(config)
    reference_config = llama_config_for_depth(
        reference_depth,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        ffn_multiplier=ffn_multiplier,
        ffn_multiple_of=ffn_multiple_of,
    )
    reference_counts, _ = model_scaling_stats(reference_config)
    reference_tokens = reference_tokens_per_param * reference_counts["effective"]
    nominal_tokens = reference_tokens_per_param * counts["effective"]

    tokens_per_micro_batch = device_batch_size * sequence_length * world_size
    if total_batch_size is None:
        predicted = reference_batch_size * (nominal_tokens / reference_tokens) ** batch_exponent
        total_batch_size = _nearest_power_of_two(predicted)
        if total_batch_size < tokens_per_micro_batch:
            raise ValueError(
                f"the scaling law predicts total_batch_size={total_batch_size:,} tokens for "
                f"depth={depth}, but one distributed micro-batch is already "
                f"{tokens_per_micro_batch:,} tokens (device_batch_size * sequence_length * "
                "world_size); flooring to the micro-batch size would silently break the "
                "batch-size scaling law across the sweep. Reduce device_batch_size or "
                "world_size for this depth, or pass an explicit total_batch_size."
            )
    elif total_batch_size < tokens_per_micro_batch:
        raise ValueError("total_batch_size is smaller than one distributed micro-batch")
    if total_batch_size % tokens_per_micro_batch:
        raise ValueError("total_batch_size must divide into whole distributed micro-batches")

    if flops_budget is not None:
        num_iterations = max(1, round(flops_budget / (flops_per_token * total_batch_size)))
        selected_budget = flops_budget
    else:
        assert target_param_data_ratio is not None
        target_tokens = target_param_data_ratio * counts["effective"]
        # Match nanochat: never exceed the requested token horizon merely to
        # round to a whole optimizer step.
        num_iterations = max(1, int(target_tokens // total_batch_size))
        selected_budget = flops_per_token * total_batch_size * num_iterations

    tokens_trained = total_batch_size * num_iterations
    actual_training_flops = flops_per_token * tokens_trained
    batch_lr_scale = math.sqrt(total_batch_size / reference_batch_size)
    matrix_learning_rate = base_learning_rate * batch_lr_scale
    embedding_learning_rate = base_embedding_learning_rate * batch_lr_scale
    unembedding_learning_rate = base_unembedding_learning_rate * batch_lr_scale
    scalar_learning_rate = base_scalar_learning_rate * batch_lr_scale
    weight_decay = (
        base_weight_decay
        * math.sqrt(total_batch_size / reference_batch_size)
        * (reference_tokens / nominal_tokens)
    )

    return ScalingPlan(
        depth=depth,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_heads=config.num_attention_heads,
        sequence_length=sequence_length,
        params_token_embeddings=counts["token_embeddings"],
        params_lm_head=counts["lm_head"],
        params_transformer=counts["transformer_matrices"],
        params_norms_and_scalars=counts["norms_and_scalars"],
        params_total=counts["total"],
        params_effective=counts["effective"],
        flops_per_token=flops_per_token,
        flops_budget=selected_budget,
        total_batch_size=total_batch_size,
        device_batch_size=device_batch_size,
        world_size=world_size,
        grad_accum_steps=total_batch_size // tokens_per_micro_batch,
        num_iterations=num_iterations,
        tokens_trained=tokens_trained,
        actual_training_flops=actual_training_flops,
        tokens_per_effective_param=tokens_trained / counts["effective"],
        learning_rate=matrix_learning_rate,
        matrix_learning_rate=matrix_learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        unembedding_learning_rate=unembedding_learning_rate,
        scalar_learning_rate=scalar_learning_rate,
        weight_decay=weight_decay,
    )


@dataclass(frozen=True)
class WarmupStableDecayScheduler:
    learning_rate: float
    max_iters: int
    warmup_iters: int = 40
    warmdown_ratio: float = 0.65
    final_lr_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.max_iters < 1:
            raise ValueError("learning_rate and max_iters must be positive")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters cannot be negative")
        if not 0 <= self.warmdown_ratio <= 1:
            raise ValueError("warmdown_ratio must be in [0, 1]")
        if not 0 <= self.final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in [0, 1]")

    def __call__(self, iteration: int) -> float:
        if not 0 <= iteration <= self.max_iters:
            raise ValueError("iteration must be in [0, max_iters]")
        if iteration == 0:
            return 0.0 if self.warmup_iters else self.learning_rate
        step = iteration - 1
        if self.warmup_iters and step < self.warmup_iters:
            return self.learning_rate * (step + 1) / self.warmup_iters
        warmdown_iters = round(self.warmdown_ratio * self.max_iters)
        warmdown_start = self.max_iters - warmdown_iters
        if warmdown_iters == 0 or step <= warmdown_start:
            return self.learning_rate
        remaining = (self.max_iters - step) / warmdown_iters
        multiplier = remaining + (1 - remaining) * self.final_lr_fraction
        return self.learning_rate * multiplier


@dataclass(frozen=True)
class MuonMomentumScheduler:
    """Nanochat's Muon momentum warmup and learning-rate-warmdown transfer."""

    max_iters: int
    warmdown_ratio: float = 0.65
    warmup_iters: int = 400
    initial_momentum: float = 0.85
    peak_momentum: float = 0.97
    final_momentum: float = 0.90

    def __post_init__(self) -> None:
        if self.max_iters < 1 or self.warmup_iters < 0:
            raise ValueError("max_iters must be positive and warmup_iters non-negative")
        if not 0 <= self.warmdown_ratio <= 1:
            raise ValueError("warmdown_ratio must be in [0, 1]")
        if any(
            not 0 <= value < 1
            for value in (
                self.initial_momentum,
                self.peak_momentum,
                self.final_momentum,
            )
        ):
            raise ValueError("momentum values must be in [0, 1)")

    def __call__(self, iteration: int) -> float:
        if not 0 <= iteration <= self.max_iters:
            raise ValueError("iteration must be in [0, max_iters]")
        # Trainer iterations are 1-based; nanochat's optimizer schedule is
        # evaluated on the corresponding 0-based step before each update.
        step = max(0, iteration - 1)
        warmdown_iters = round(self.warmdown_ratio * self.max_iters)
        warmdown_start = self.max_iters - warmdown_iters
        if self.warmup_iters and step < self.warmup_iters:
            fraction = step / self.warmup_iters
            return self.initial_momentum + fraction * (
                self.peak_momentum - self.initial_momentum
            )
        if warmdown_iters and step >= warmdown_start:
            progress = (step - warmdown_start) / warmdown_iters
            return self.peak_momentum + progress * (
                self.final_momentum - self.peak_momentum
            )
        return self.peak_momentum


@dataclass(frozen=True)
class CosineWeightDecayScheduler:
    """Cosine-decay Muon's cautious weight decay to zero over training."""

    weight_decay: float
    max_iters: int

    def __post_init__(self) -> None:
        if self.weight_decay < 0 or self.max_iters < 1:
            raise ValueError("weight_decay must be non-negative and max_iters positive")

    def __call__(self, iteration: int) -> float:
        if not 0 <= iteration <= self.max_iters:
            raise ValueError("iteration must be in [0, max_iters]")
        step = max(0, iteration - 1)
        return self.weight_decay * 0.5 * (
            1.0 + math.cos(math.pi * step / self.max_iters)
        )


def append_result_csv(path: str | Path, row: Mapping[str, object]) -> None:
    """Append one stable-schema scaling result, rejecting mismatched headers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing = next(reader, [])
        if existing != fieldnames:
            raise ValueError("result row does not match the existing CSV schema")
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_result_csv(path: str | Path) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_rows: list[dict[str, float]] = []
    for row in rows:
        numeric_rows.append({key: float(value) for key, value in row.items()})
    return numeric_rows


def _solve_three_by_three(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("cannot fit a quadratic to degenerate points")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][-1] for row in range(3)]


def _quadratic_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("quadratic fitting requires at least three paired points")
    sums = [sum(value**power for value in x) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum((value**2) * target for value, target in zip(x, y, strict=True)),
        sum(value * target for value, target in zip(x, y, strict=True)),
        sum(y),
    ]
    a, b, c = _solve_three_by_three(matrix, vector)
    return a, b, c


def _interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for left in range(len(xs) - 1):
        if xs[left] <= x <= xs[left + 1]:
            fraction = (x - xs[left]) / (xs[left + 1] - xs[left])
            return ys[left] + fraction * (ys[left + 1] - ys[left])
    raise RuntimeError("interpolation point was not bracketed")


def fit_isoflop_curves(
    rows: Iterable[Mapping[str, float]],
    *,
    metric: str = "val_bpb",
) -> list[dict[str, float]]:
    """Fit loss quadratics in log-parameter space for every FLOP budget."""
    grouped: dict[float, list[Mapping[str, float]]] = {}
    for row in rows:
        value = float(row[metric])
        if not math.isfinite(value) or value <= 0:
            continue
        grouped.setdefault(float(row["flops_budget"]), []).append(row)

    optimums: list[dict[str, float]] = []
    for budget in sorted(grouped):
        subset = sorted(grouped[budget], key=lambda row: float(row["params_effective"]))
        if len(subset) < 3:
            continue
        log_params = [math.log10(float(row["params_effective"])) for row in subset]
        losses = [float(row[metric]) for row in subset]

        # Center the parameter axis before fitting: log-parameters span many
        # orders of magnitude, and an uncentered quadratic fit ill-conditions
        # the Vandermonde system solved by _quadratic_fit. Fit in centered
        # coordinates, then shift the coefficients back onto the original
        # log-parameter axis so every downstream consumer (including
        # write_scaling_svg) can keep using absolute log10(params).
        mean_log_params = sum(log_params) / len(log_params)
        centered = [value - mean_log_params for value in log_params]
        centered_a, centered_b, centered_c = _quadratic_fit(centered, losses)
        a = centered_a
        b = centered_b - 2 * centered_a * mean_log_params
        c = centered_a * mean_log_params**2 - centered_b * mean_log_params + centered_c

        candidate = -b / (2 * a) if a > 0 else float("nan")
        interior_optimum = a > 0 and log_params[0] <= candidate <= log_params[-1]
        if interior_optimum:
            log_optimum = candidate
            loss_optimum = a * log_optimum**2 + b * log_optimum + c
        else:
            best = min(range(len(losses)), key=losses.__getitem__)
            log_optimum = log_params[best]
            loss_optimum = losses[best]

        # Derive tokens_trained from the exact budget = flops_per_token * tokens
        # identity rather than interpolating tokens_trained directly. Near the
        # optimum, tokens_trained is a steep, convex function of log-parameters
        # (it is roughly budget / params), while flops_per_token is close to
        # linear in params; interpolating the latter is far more accurate.
        flops_per_token = _interpolate(
            log_optimum,
            log_params,
            [float(row["flops_per_token"]) for row in subset],
        )
        params = 10**log_optimum
        tokens = budget / flops_per_token
        optimums.append(
            {
                "flops_budget": budget,
                "params_effective": params,
                "tokens_trained": tokens,
                "tokens_per_effective_param": tokens / params,
                metric: loss_optimum,
                "quadratic_a": a,
                "quadratic_b": b,
                "quadratic_c": c,
                "interior_optimum": float(interior_optimum),
            }
        )
    return optimums


def fit_power_law(
    rows: Sequence[Mapping[str, float]],
    *,
    x_key: str,
    y_key: str,
) -> dict[str, float]:
    if len(rows) < 2:
        raise ValueError("power-law fitting requires at least two points")
    x = [math.log10(float(row[x_key])) for row in rows]
    y = [math.log10(float(row[y_key])) for row in rows]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("power-law x values must not all match")
    exponent = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    intercept = mean_y - exponent * mean_x
    predictions = [intercept + exponent * value for value in x]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y, predictions))
    total = sum((actual - mean_y) ** 2 for actual in y)
    return {
        "coefficient": 10**intercept,
        "exponent": exponent,
        "r_squared": 1.0 if total == 0 and residual == 0 else 1 - residual / total,
    }


def analyze_scaling_results(
    rows: Sequence[Mapping[str, float]],
    *,
    metric: str = "val_bpb",
) -> dict[str, object]:
    optimums = fit_isoflop_curves(rows, metric=metric)
    analysis: dict[str, object] = {"metric": metric, "optimums": optimums}
    interior_optimums = [row for row in optimums if row["interior_optimum"] == 1.0]
    analysis["num_interior_optimums"] = len(interior_optimums)
    if len(interior_optimums) >= 2:
        analysis["optimal_params_power_law"] = fit_power_law(
            interior_optimums,
            x_key="flops_budget",
            y_key="params_effective",
        )
        analysis["optimal_tokens_power_law"] = fit_power_law(
            interior_optimums,
            x_key="flops_budget",
            y_key="tokens_trained",
        )
    else:
        analysis["power_law_warning"] = (
            "Need interior iso-FLOP minima for at least two compute budgets; "
            "expand the depth grid before interpreting scaling exponents."
        )
    return analysis


def write_scaling_svg(
    path: str | Path,
    rows: Sequence[Mapping[str, float]],
    analysis: Mapping[str, object],
    *,
    metric: str = "val_bpb",
) -> None:
    """Write a dependency-free three-panel scaling-law visualization."""
    optimums = list(analysis.get("optimums", []))
    if not optimums:
        raise ValueError("analysis contains no iso-FLOP optimums")

    width, height = 1_260, 420
    panel_width, panel_height = 340, 290
    panel_y = 55
    panel_xs = (65, 465, 865)
    colors = ("#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#111827}.title{font-size:16px;'
        'font-weight:600}.label{font-size:12px;fill:#4b5563}.axis{stroke:#9ca3af;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}</style>',
    ]

    def bounds(values: Sequence[float], padding: float = 0.08) -> tuple[float, float]:
        low, high = min(values), max(values)
        if low == high:
            return low - 0.5, high + 0.5
        extra = (high - low) * padding
        return low - extra, high + extra

    def point(
        panel: int,
        x_value: float,
        y_value: float,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> tuple[float, float]:
        left = panel_xs[panel]
        x = left + panel_width * (x_value - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
        y = panel_y + panel_height * (1 - (y_value - y_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        return x, y

    def axes(panel: int, title: str, x_label: str, y_label: str) -> None:
        left = panel_xs[panel]
        elements.append(
            f'<rect class="axis" x="{left}" y="{panel_y}" width="{panel_width}" '
            f'height="{panel_height}" fill="none"/>'
        )
        for fraction in (0.25, 0.5, 0.75):
            x = left + panel_width * fraction
            y = panel_y + panel_height * fraction
            elements.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" '
                f'y2="{panel_y + panel_height}"/>'
            )
            elements.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                f'x2="{left + panel_width}" y2="{y:.1f}"/>'
            )
        elements.append(
            f'<text class="title" x="{left + panel_width / 2}" y="27" '
            f'text-anchor="middle">{html.escape(title)}</text>'
        )
        elements.append(
            f'<text class="label" x="{left + panel_width / 2}" y="385" '
            f'text-anchor="middle">{html.escape(x_label)}</text>'
        )
        elements.append(
            f'<text class="label" transform="translate({left - 45},'
            f'{panel_y + panel_height / 2}) rotate(-90)" text-anchor="middle">'
            f'{html.escape(y_label)}</text>'
        )

    valid_rows = [
        row
        for row in rows
        if math.isfinite(float(row[metric])) and float(row[metric]) > 0
    ]
    budgets = sorted({float(row["flops_budget"]) for row in valid_rows})
    log_params_all = [math.log10(float(row["params_effective"])) for row in valid_rows]
    losses_all = [float(row[metric]) for row in valid_rows]
    x_bounds = bounds(log_params_all)
    y_bounds = bounds(losses_all)
    axes(0, "Iso-FLOP curves", "Effective parameters (log10)", metric)

    optimum_by_budget = {float(row["flops_budget"]): row for row in optimums}
    for index, budget in enumerate(budgets):
        color = colors[index % len(colors)]
        subset = sorted(
            (row for row in valid_rows if float(row["flops_budget"]) == budget),
            key=lambda row: float(row["params_effective"]),
        )
        for row in subset:
            x, y = point(
                0,
                math.log10(float(row["params_effective"])),
                float(row[metric]),
                x_bounds,
                y_bounds,
            )
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        optimum = optimum_by_budget.get(budget)
        if optimum is None:
            continue
        a = float(optimum["quadratic_a"])
        b = float(optimum["quadratic_b"])
        c = float(optimum["quadratic_c"])
        start = math.log10(float(subset[0]["params_effective"]))
        end = math.log10(float(subset[-1]["params_effective"]))
        curve = []
        for step in range(81):
            log_parameter = start + (end - start) * step / 80
            loss = a * log_parameter**2 + b * log_parameter + c
            curve.append(point(0, log_parameter, loss, x_bounds, y_bounds))
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in curve)
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        x, y = point(
            0,
            math.log10(float(optimum["params_effective"])),
            float(optimum[metric]),
            x_bounds,
            y_bounds,
        )
        elements.append(
            f'<path d="M{x:.1f},{y - 7:.1f} L{x + 2:.1f},{y - 2:.1f} '
            f'L{x + 7:.1f},{y - 2:.1f} L{x + 3:.1f},{y + 1:.1f} '
            f'L{x + 5:.1f},{y + 7:.1f} L{x:.1f},{y + 3:.1f} '
            f'L{x - 5:.1f},{y + 7:.1f} L{x - 3:.1f},{y + 1:.1f} '
            f'L{x - 7:.1f},{y - 2:.1f} L{x - 2:.1f},{y - 2:.1f} Z" fill="{color}"/>'
        )

    def power_panel(panel: int, y_key: str, title: str, y_label: str, fit_key: str) -> None:
        axes(panel, title, "Training FLOPs (log10)", y_label)
        log_x = [math.log10(float(row["flops_budget"])) for row in optimums]
        log_y = [math.log10(float(row[y_key])) for row in optimums]
        panel_x_bounds = bounds(log_x)
        panel_y_bounds = bounds(log_y)
        for row in optimums:
            x, y = point(
                panel,
                math.log10(float(row["flops_budget"])),
                math.log10(float(row[y_key])),
                panel_x_bounds,
                panel_y_bounds,
            )
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#111827"/>')
        fit = analysis.get(fit_key)
        if isinstance(fit, Mapping):
            exponent = float(fit["exponent"])
            coefficient = float(fit["coefficient"])
            line = []
            for log_flops in (min(log_x), max(log_x)):
                log_value = math.log10(coefficient) + exponent * log_flops
                line.append(point(panel, log_flops, log_value, panel_x_bounds, panel_y_bounds))
            elements.append(
                f'<line x1="{line[0][0]:.1f}" y1="{line[0][1]:.1f}" '
                f'x2="{line[1][0]:.1f}" y2="{line[1][1]:.1f}" '
                'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>'
            )
            elements.append(
                f'<text class="label" x="{panel_xs[panel] + 12}" y="{panel_y + 22}">'
                f'exponent = {exponent:.3f}</text>'
            )

    power_panel(
        1,
        "params_effective",
        "Compute-optimal model size",
        "Effective parameters (log10)",
        "optimal_params_power_law",
    )
    power_panel(
        2,
        "tokens_trained",
        "Compute-optimal training tokens",
        "Training tokens (log10)",
        "optimal_tokens_power_law",
    )
    elements.append("</svg>")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")

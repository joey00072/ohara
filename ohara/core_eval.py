from __future__ import annotations

import csv
import io
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.distributed as dist
import yaml

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def _get_bos_token_id(tokenizer) -> int:
    if hasattr(tokenizer, "get_bos_token_id"):
        bos = tokenizer.get_bos_token_id()
        if bos is not None:
            return int(bos)
    for attr in ("bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for token in ("<|bos|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != -1:
                return int(token_id)
    raise ValueError("Failed to infer BOS token id from tokenizer")


def _encode_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    raise TypeError("Tokenizer must provide an encode method")


def _encode_with_bos(tokenizer, text: str) -> list[int]:
    return [_get_bos_token_id(tokenizer)] + _encode_text(tokenizer, text)


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = ""
    for example in fewshot_examples:
        prefix += (
            f"{example['query']}{continuation_delimiter}"
            f"{example['choices'][example['gold']]}\n\n"
        )
    return [f"{prefix}{item['query']}{continuation_delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = ""
    for example in fewshot_examples:
        prefix += (
            f"{example['context_options'][example['gold']]}"
            f"{continuation_delimiter}{example['continuation']}\n\n"
        )
    return [
        f"{prefix}{context_option}{continuation_delimiter}{item['continuation']}"
        for context_option in item["context_options"]
    ]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = ""
    for example in fewshot_examples:
        prefix += (
            f"{example['context'].strip()}{continuation_delimiter}"
            f"{example['continuation']}\n\n"
        )
    prompt_without = f"{prefix}{item['context'].strip()}{continuation_delimiter}".strip()
    prompt_with = f"{prompt_without}{item['continuation']}"
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    min_len = min(len(seq) for seq in token_sequences)
    if direction == "left":
        positions = range(min_len)
    elif direction == "right":
        positions = range(-1, -min_len - 1, -1)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    for idx, pos in enumerate(positions):
        token = token_sequences[0][pos]
        if not all(seq[pos] == token for seq in token_sequences):
            return idx
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz = len(tokens)
    seq_len = max(len(row) for row in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for idx, row in enumerate(tokens):
        input_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = [_encode_with_bos(tokenizer, prompt) for prompt in prompts]
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = [_encode_with_bos(tokenizer, prompt) for prompt in prompts]
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = [_encode_with_bos(tokenizer, prompt) for prompt in prompts]
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    if not (start_idx < end_idx and tokens_without == tokens_with[:start_idx]):
        raise AssertionError("LM prompts must satisfy prefix relationship in token space")
    return [tokens_with], [start_idx], [end_idx]


def infer_model_max_seq_len(model) -> int | None:
    candidates = [
        getattr(model, "max_seq_len", None),
        getattr(model, "max_sequence_length", None),
    ]
    cfg = getattr(model, "config", None)
    if cfg is not None:
        candidates.extend(
            [
                getattr(cfg, "max_seq_len", None),
                getattr(cfg, "max_sequence_length", None),
                getattr(cfg, "seq_len", None),
                getattr(cfg, "sequence_len", None),
            ]
        )
    wrapped = getattr(model, "model", None)
    if wrapped is not None:
        wrapped_cfg = getattr(wrapped, "config", None)
        if wrapped_cfg is not None:
            candidates.extend(
                [
                    getattr(wrapped_cfg, "max_seq_len", None),
                    getattr(wrapped_cfg, "max_sequence_length", None),
                    getattr(wrapped_cfg, "seq_len", None),
                    getattr(wrapped_cfg, "sequence_len", None),
                ]
            )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


@torch.no_grad()
def forward_model(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    batch_size, seq_len = input_ids.size()
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    losses[:, -1] = float("nan")
    predictions = logits.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    max_tokens = infer_model_max_seq_len(model)
    if max_tokens is not None:
        cropped_tokens, cropped_start, cropped_end = [], [], []
        for tok, start_idx, end_idx in zip(tokens, start_idxs, end_idxs):
            if len(tok) > max_tokens:
                trim = len(tok) - max_tokens
                if start_idx - trim < 0 or end_idx - trim < 0:
                    # If continuation region is truncated away, mark incorrect.
                    return False
                cropped_tokens.append(tok[-max_tokens:])
                cropped_start.append(start_idx - trim)
                cropped_end.append(end_idx - trim)
            else:
                cropped_tokens.append(tok)
                cropped_start.append(start_idx)
                cropped_end.append(end_idx)
        tokens, start_idxs, end_idxs = cropped_tokens, cropped_start, cropped_end

    pad_token_id = _get_bos_token_id(tokenizer)
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    if task_type == "language_modeling":
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        return torch.all(predicted_tokens == actual_tokens).item()

    mean_losses = [
        losses[row_idx, si - 1 : ei - 1].mean().item()
        for row_idx, (si, ei) in enumerate(zip(start_idxs, end_idxs))
    ]
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


def evaluate_task(model, tokenizer, data, device, task_meta):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core_from_bundle(model, tokenizer, device, eval_bundle_dir, max_per_task=-1):
    eval_bundle_path = Path(eval_bundle_dir)
    config_path = eval_bundle_path.joinpath("core.yaml")
    data_base_path = eval_bundle_path.joinpath("eval_data")
    eval_meta_path = eval_bundle_path.joinpath("eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}
    for task in tasks:
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }

        data_path = data_base_path.joinpath(task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy

        random_baseline = random_baselines[label]
        centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }


def _download_eval_bundle(eval_bundle_dir: Path, bundle_url: str) -> None:
    eval_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(bundle_url) as response:
        payload = response.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpzip = Path(tmpdir).joinpath("eval_bundle.zip")
        tmpzip.write_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
            zf.extractall(tmpdir)
        extracted = Path(tmpdir).joinpath("eval_bundle")
        if eval_bundle_dir.exists():
            shutil.rmtree(eval_bundle_dir)
        shutil.move(str(extracted), str(eval_bundle_dir))


def evaluate_core(
    model,
    tokenizer,
    device,
    *,
    max_per_task: int = -1,
    eval_bundle_dir: str | Path = "./eval_bundle",
    bundle_url: str = EVAL_BUNDLE_URL,
):
    eval_bundle_path = Path(eval_bundle_dir)
    if not eval_bundle_path.exists():
        _download_eval_bundle(eval_bundle_path, bundle_url)
    return evaluate_core_from_bundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_bundle_dir=eval_bundle_path,
        max_per_task=max_per_task,
    )

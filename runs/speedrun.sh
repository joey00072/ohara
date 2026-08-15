#!/bin/bash
# Pretrain a small Llama on ClimbMix, finetune it into a chat model, then serve it.
#
# The nanochat pipeline, on ohara's stack. One dial controls model size: DEPTH.
# Everything else (width, batch size, learning rates, weight decay, the token
# horizon) is derived from it by examples/scaling_laws.py's planner.
#
#   bash runs/speedrun.sh
#
# The run takes hours, so use a detachable session:
#
#   screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
#
# Tunables (all overridable from the environment):
#   DEPTH             model depth; the single complexity dial          (default 12)
#   NPROC             GPUs to train on                                 (default: all visible)
#   DEVICE_BATCH_SIZE per-GPU micro-batch in sequences                 (default 16)
#   SEQ_LEN           context window in tokens                         (default 2048)
#   TOKEN_RATIO       training tokens per effective parameter          (default 12)
#   SHARDS            ClimbMix parquet shards to stage                 (default 24)
#   NUM_WORKERS       dataloader processes per rank                    (default 8)
#   SERVE             launch the chat web UI when training finishes    (default 1)
#   PORT              port for the chat web UI                         (default 8080)
set -euo pipefail

DEPTH="${DEPTH:-12}"
SEQ_LEN="${SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOKEN_RATIO="${TOKEN_RATIO:-12}"
SHARDS="${SHARDS:-24}"
# The corpus is stored as text, so every batch is tokenized on the fly. At 0 the
# main process does that work and the GPUs stall waiting on it; 1 worker moves it
# off the training loop and prefetches the next batch during the backward pass.
#
# Do not raise this above 1 for the staged corpus. It is a single JSONL file, so
# the streaming reader exposes exactly one shard: extra workers would get an empty
# stream, and StreamingTextDataset's own modulo sharding would additionally thin
# the corpus by a factor of num_workers. Use NUM_WORKERS=0 with --resume, which
# needs to replay the data stream exactly.
NUM_WORKERS="${NUM_WORKERS:-1}"
TRAIN_DOCS="${TRAIN_DOCS:-4000000}"
VAL_DOCS="${VAL_DOCS:-20000}"
TOKENIZER="${TOKENIZER:-EleutherAI/gpt-neo-125m}"
DATA_DIR="${DATA_DIR:-./data/scaling_corpus}"
CKPT_DIR="${CKPT_DIR:-./ckpt}"
SERVE="${SERVE:-1}"
PORT="${PORT:-8080}"
SFT_ITERS="${SFT_ITERS:-800}"

export OMP_NUM_THREADS=1

cd "$(dirname "$0")/.."

if [ -z "${NPROC:-}" ]; then
  NPROC=$(python - <<'PY'
try:
    import torch
    print(max(1, torch.cuda.device_count()))
except Exception:
    print(1)
PY
)
fi

RUN="d${DEPTH}"
BASE_CKPT="${CKPT_DIR}/base_${RUN}.pt"
SFT_CKPT="${CKPT_DIR}/sft_${RUN}.pt"
mkdir -p "$CKPT_DIR" runs

echo "=============================================================="
echo " ohara speedrun | depth=${DEPTH} gpus=${NPROC} seq_len=${SEQ_LEN}"
echo " tokens/param=${TOKEN_RATIO} device_batch=${DEVICE_BATCH_SIZE}"
echo "=============================================================="

# --------------------------------------------------------------------------
# 1) Data. Stage ClimbMix shards locally so training never blocks on the network.

if [ -d "${DATA_DIR}" ] && [ -n "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
  echo "[1/4] corpus already staged at ${DATA_DIR}, skipping"
else
  echo "[1/4] staging ClimbMix corpus -> ${DATA_DIR}"
  python examples/prepare_scaling_data.py \
    --climbmix-train-shards "${SHARDS}" \
    --train-documents "${TRAIN_DOCS}" \
    --validation-documents "${VAL_DOCS}" \
    --tokenizer "${TOKENIZER}" \
    --output-dir "${DATA_DIR}" \
    --skip-existing
fi

# --------------------------------------------------------------------------
# 2) Plan. Derive every hyperparameter from DEPTH, exactly as the sweep does.
#    --chat-tokens is reflected here so the planned vocabulary matches the one
#    pretraining will actually build.

echo "[2/4] planning depth=${DEPTH}"
eval "$(python - <<PY
from ohara.chat import CHAT_SPECIAL_TOKENS
from ohara.scaling import plan_scaling_run
from ohara.tokenizer import get_tokenizer

vocab = len(get_tokenizer(hf_name="${TOKENIZER}", prefer_hf=True)) + len(CHAT_SPECIAL_TOKENS)
plan = plan_scaling_run(
    ${DEPTH},
    vocab_size=vocab,
    target_param_data_ratio=${TOKEN_RATIO},
    sequence_length=${SEQ_LEN},
    device_batch_size=${DEVICE_BATCH_SIZE},
    world_size=${NPROC},
)
for key, value in {
    "HIDDEN_SIZE": plan.hidden_size,
    "INTERMEDIATE_SIZE": plan.intermediate_size,
    "NUM_HEADS": plan.num_heads,
    "GRAD_ACCUM": plan.grad_accum_steps,
    "MAX_ITERS": plan.num_iterations,
    "MATRIX_LR": plan.matrix_learning_rate,
    "EMBEDDING_LR": plan.embedding_learning_rate,
    "UNEMBEDDING_LR": plan.unembedding_learning_rate,
    "SCALAR_LR": plan.scalar_learning_rate,
    "WEIGHT_DECAY": plan.weight_decay,
    "PARAMS_EFFECTIVE": plan.params_effective,
    "TOKENS_TRAINED": plan.tokens_trained,
    "TOTAL_BATCH_SIZE": plan.total_batch_size,
}.items():
    print(f"{key}={value}")
PY
)"

printf ' model:  %s effective params, hidden=%s ffn=%s heads=%s\n' \
  "$PARAMS_EFFECTIVE" "$HIDDEN_SIZE" "$INTERMEDIATE_SIZE" "$NUM_HEADS"
printf ' budget: %s tokens over %s iters, batch %s tokens (accum %s)\n' \
  "$TOKENS_TRAINED" "$MAX_ITERS" "$TOTAL_BATCH_SIZE" "$GRAD_ACCUM"

# --------------------------------------------------------------------------
# 3) Pretrain.

if [ "$NPROC" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=${NPROC}"
else
  LAUNCH="python"
fi

echo "[3/4] pretraining -> ${BASE_CKPT}"
$LAUNCH examples/train_llama_engine.py \
  --dataset "${DATA_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --chat-tokens \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${DEVICE_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --max-iters "${MAX_ITERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --intermediate-size "${INTERMEDIATE_SIZE}" \
  --num-layers "${DEPTH}" \
  --num-heads "${NUM_HEADS}" \
  --optimizer muon \
  --init-style nanochat \
  --no-weight-tying \
  --lr-schedule wsd \
  --matrix-learning-rate "${MATRIX_LR}" \
  --embedding-learning-rate "${EMBEDDING_LR}" \
  --unembedding-learning-rate "${UNEMBEDDING_LR}" \
  --scalar-learning-rate "${SCALAR_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --evaluate-bpb \
  --num-workers "${NUM_WORKERS}" \
  --eval-every 250 \
  --save-every 500 \
  --checkpoint-path "${BASE_CKPT}" \
  --result-json "runs/base_${RUN}.json"

# --------------------------------------------------------------------------
# 4) SFT. Teaches the conversation format; minutes of work next to pretraining.

echo "[4/4] supervised finetuning -> ${SFT_CKPT}"
$LAUNCH examples/train_sft.py \
  --pretrained-checkpoint "${BASE_CKPT}" \
  --checkpoint-path "${SFT_CKPT}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${DEVICE_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --max-iters "${SFT_ITERS}" \
  --num-workers "${NUM_WORKERS}" \
  --result-json "runs/sft_${RUN}.json"

echo
echo "done. base=${BASE_CKPT} sft=${SFT_CKPT}"

if [ "$SERVE" = "1" ]; then
  echo "starting chat UI on port ${PORT}"
  echo "from your laptop:  ssh -N -L ${PORT}:localhost:${PORT} <user>@<host>"
  python examples/chat_web.py --checkpoint "${SFT_CKPT}" --port "${PORT}"
else
  echo "chat with it:  python examples/chat_web.py --checkpoint ${SFT_CKPT}"
fi

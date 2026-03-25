#!/usr/bin/env bash
# DAgger online IL training for Seq2Seq on SatNav.
#
# Single-GPU (default):
#   bash scripts/seq2seq/train_dagger.sh
#
# Multi-GPU (4 cards):
#   CUDA_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash scripts/seq2seq/train_dagger.sh
#
# Key overrides (env vars):
#   PRETRAIN_CKPT       Path to pre-trained offline checkpoint to fine-tune from.
#                       Defaults to output/seq2seq_offline/checkpoints/latest/best.pth.
#   DAGGER_ITERATIONS   Number of DAgger outer iterations (default: 5).
#   UPDATE_SIZE         Global new episodes to collect per iteration (default: 200).
#   BETA_DECAY          Exponential beta decay factor (default: 0.5).
#   EPOCHS_PER_ITER     Training epochs per DAgger iteration (default: 1).
#   RESET_DATASET       Clear aggregated dataset at startup (default: false).
#   PER_GPU_BATCH_SIZE  Batch size per GPU (default: 8).
#   LEARNING_RATE       Optimizer learning rate (default: 1e-4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav

# ── Hardware ─────────────────────────────────────────────────────────────────
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/seq2seq_dagger.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(echo "${CUDA_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-output/seq2seq_dagger}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-output/seq2seq_offline/checkpoints/latest/best.pth}"

# ── Training hyper-params ─────────────────────────────────────────────────────
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# ── DAgger-specific params ────────────────────────────────────────────────────
DAGGER_ITERATIONS="${DAGGER_ITERATIONS:-5}"
UPDATE_SIZE="${UPDATE_SIZE:-200}"
BETA_DECAY="${BETA_DECAY:-0.5}"
EPOCHS_PER_ITER="${EPOCHS_PER_ITER:-1}"
RESET_DATASET="${RESET_DATASET:-false}"

# ── SwanLab ───────────────────────────────────────────────────────────────────
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${OUTPUT_ROOT}/swanlab}"
USE_WXWORK_NOTIFICATION="${USE_WXWORK_NOTIFICATION:-false}"
SWANLAB_WEBHOOK_URL="${SWANLAB_WEBHOOK_URL:-}"
SWANLAB_SECRET="${SWANLAB_SECRET:-}"

# ── Experiment name ───────────────────────────────────────────────────────────
RUN_PREFIX="${RUN_PREFIX:-seq2seq-dagger}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
EFFECTIVE_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * GPUS_PER_NODE))
EXP_NAME="${RUN_PREFIX}-g${GPUS_PER_NODE}-bs${EFFECTIVE_BATCH_SIZE}-iter${DAGGER_ITERATIONS}-lr${LEARNING_RATE}-${TIMESTAMP}"
if [ -n "${RUN_SUFFIX}" ]; then
    EXP_NAME="${EXP_NAME}-${RUN_SUFFIX}"
fi
if [ -n "${SWANLAB_EXP_NAME}" ]; then
    EXP_NAME="${SWANLAB_EXP_NAME}"
fi

# ── Resolve absolute paths ────────────────────────────────────────────────────
resolve_path() {
    local input_path="$1"
    if [[ "${input_path}" = /* ]]; then
        echo "${input_path}"
    else
        echo "${REPO_ROOT}/${input_path}"
    fi
}

OUTPUT_ROOT_ABS="$(resolve_path "${OUTPUT_ROOT}")"
CHECKPOINT_DIR="${OUTPUT_ROOT_ABS}/checkpoints/${EXP_NAME}"
RESULTS_DIR="${OUTPUT_ROOT_ABS}/results/${EXP_NAME}"
VIDEO_DIR="${OUTPUT_ROOT_ABS}/videos/${EXP_NAME}"
# Dataset storage: keep {split} placeholder for Python to resolve at runtime.
DATASET_STORAGE_DIR="${OUTPUT_ROOT_ABS}/datasets/{split}"
LOG_DIR="${OUTPUT_ROOT_ABS}/logs"
LATEST_LINK="${OUTPUT_ROOT_ABS}/checkpoints/latest"

PRETRAIN_CKPT_ABS="$(resolve_path "${PRETRAIN_CKPT}")"
SWANLAB_LOGDIR_ABS="$(resolve_path "${SWANLAB_LOGDIR}")"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

if [ "${SWANLAB_MODE}" != "cloud" ]; then
    mkdir -p "${SWANLAB_LOGDIR_ABS}"
fi

# ── Validate pre-trained checkpoint ──────────────────────────────────────────
if [ ! -f "${PRETRAIN_CKPT_ABS}" ]; then
    echo "[ERROR] Pre-trained checkpoint not found: ${PRETRAIN_CKPT_ABS}"
    echo "        Run seq2seq offline training first, or set PRETRAIN_CKPT."
    exit 1
fi

# ── Launcher: torchrun for multi-GPU, plain python for single-GPU ─────────────
if [ "${GPUS_PER_NODE}" -gt 1 ]; then
    if command -v torchrun >/dev/null 2>&1; then
        LAUNCH_CMD=(torchrun
            --nnodes=1
            --node_rank=0
            --nproc_per_node="${GPUS_PER_NODE}"
            --master_addr="${MASTER_ADDR}"
            --master_port="${MASTER_PORT}")
    else
        LAUNCH_CMD=(python -m torch.distributed.run
            --nnodes=1
            --node_rank=0
            --nproc_per_node="${GPUS_PER_NODE}"
            --master_addr="${MASTER_ADDR}"
            --master_port="${MASTER_PORT}")
    fi
    DISTRIBUTED_ENABLED="true"
else
    LAUNCH_CMD=(python)
    DISTRIBUTED_ENABLED="false"
fi

SWANLAB_MODE_OVERRIDE="disabled"
if [ "${USE_SWANLAB}" = "true" ]; then
    SWANLAB_MODE_OVERRIDE="${SWANLAB_MODE}"
fi

# ── Config overrides ──────────────────────────────────────────────────────────
OVERRIDES=(
    DISTRIBUTED.enabled "${DISTRIBUTED_ENABLED}"
    IL.lr "${LEARNING_RATE}"
    IL.batch_size "${PER_GPU_BATCH_SIZE}"
    IL.epochs "${EPOCHS_PER_ITER}"
    IL.load_from_ckpt true
    IL.ckpt_to_load "${PRETRAIN_CKPT_ABS}"
    IL.DAGGER.iterations "${DAGGER_ITERATIONS}"
    IL.DAGGER.update_size "${UPDATE_SIZE}"
    IL.DAGGER.beta_decay "${BETA_DECAY}"
    IL.DAGGER.reset_dataset_on_start "${RESET_DATASET}"
    IL.DAGGER.storage_dir "${DATASET_STORAGE_DIR}"
    CHECKPOINT_FOLDER "${CHECKPOINT_DIR}"
    RESULTS_DIR "${RESULTS_DIR}"
    VIDEO_DIR "${VIDEO_DIR}"
    SWANLAB.project "${SWANLAB_PROJECT}"
    SWANLAB.experiment_name "${EXP_NAME}"
    SWANLAB.mode "${SWANLAB_MODE_OVERRIDE}"
    SWANLAB.logdir "${SWANLAB_LOGDIR_ABS}"
    SWANLAB.use_wxwork_notification "${USE_WXWORK_NOTIFICATION}"
)

if [ -n "${SWANLAB_WORKSPACE}" ]; then
    OVERRIDES+=(SWANLAB.workspace "${SWANLAB_WORKSPACE}")
fi
if [ -n "${SWANLAB_WEBHOOK_URL}" ]; then
    OVERRIDES+=(SWANLAB.webhook_url "${SWANLAB_WEBHOOK_URL}")
fi
if [ -n "${SWANLAB_SECRET}" ]; then
    OVERRIDES+=(SWANLAB.secret "${SWANLAB_SECRET}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# ── Print summary ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "SatNav Seq2Seq DAgger Training"
echo "=========================================="
echo "Config         : ${CONFIG_PATH}"
echo "GPUs           : ${GPUS_PER_NODE} (${CUDA_DEVICES})"
echo "Batch          : ${PER_GPU_BATCH_SIZE} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR             : ${LEARNING_RATE}"
echo "Iterations     : ${DAGGER_ITERATIONS}"
echo "Update size    : ${UPDATE_SIZE} global eps/iter"
echo "Epochs/iter    : ${EPOCHS_PER_ITER}"
echo "Beta decay     : ${BETA_DECAY}"
echo "Reset dataset  : ${RESET_DATASET}"
echo "Pretrain ckpt  : ${PRETRAIN_CKPT_ABS}"
echo "Dataset storage: ${DATASET_STORAGE_DIR}"
echo "Output root    : ${OUTPUT_ROOT_ABS}"
echo "SwanLab        : ${USE_SWANLAB} (${SWANLAB_MODE_OVERRIDE})"
echo "Experiment     : ${EXP_NAME}"
echo "Checkpoint     : ${CHECKPOINT_DIR}"
echo "Log file       : ${TRAIN_LOG_FILE}"
echo "=========================================="

cd "${REPO_ROOT}"

"${LAUNCH_CMD[@]}" \
    run.py \
    --exp-config "${CONFIG_PATH}" \
    --run-type train \
    "${OVERRIDES[@]}" \
    2>&1 | tee "${TRAIN_LOG_FILE}"

# 训练成功后更新 latest 软链
ln -sfn "$(basename "${CHECKPOINT_DIR}")" "${LATEST_LINK}"
echo "Updated latest -> $(basename "${CHECKPOINT_DIR}")"

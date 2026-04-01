#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/seq2seq_offline.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(echo "${CUDA_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/seq2seq_offline}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

USE_SWANLAB="${USE_SWANLAB:-true}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-baseline}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${OUTPUT_ROOT}/swanlab}"
USE_WXWORK_NOTIFICATION="${USE_WXWORK_NOTIFICATION:-false}"
SWANLAB_WEBHOOK_URL="${SWANLAB_WEBHOOK_URL:-}"
SWANLAB_SECRET="${SWANLAB_SECRET:-}"

RUN_PREFIX="${RUN_PREFIX:-seq2seq-offline-ddp}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
EFFECTIVE_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * GPUS_PER_NODE))
EXP_NAME="${RUN_PREFIX}-g${GPUS_PER_NODE}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
if [ -n "${RUN_SUFFIX}" ]; then
    EXP_NAME="${EXP_NAME}-${RUN_SUFFIX}"
fi
if [ -n "${SWANLAB_EXP_NAME}" ]; then
    EXP_NAME="${SWANLAB_EXP_NAME}"
fi

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
LOG_DIR="${OUTPUT_ROOT_ABS}/logs"
LATEST_LINK="${OUTPUT_ROOT_ABS}/checkpoints/latest"

# 只预先创建训练本身需要的目录；results/videos 由 eval/video 步骤自己建
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# swanlab 本地目录仅在非 cloud 模式下需要预先创建
SWANLAB_LOGDIR_ABS="$(resolve_path "${SWANLAB_LOGDIR}")"
if [ "${SWANLAB_MODE}" != "cloud" ]; then
    mkdir -p "${SWANLAB_LOGDIR_ABS}"
fi

if command -v torchrun >/dev/null 2>&1; then
    DIST_LAUNCH=(torchrun)
else
    DIST_LAUNCH=(python -m torch.distributed.run)
fi

SWANLAB_MODE_OVERRIDE="disabled"
if [ "${USE_SWANLAB}" = "true" ]; then
    SWANLAB_MODE_OVERRIDE="${SWANLAB_MODE}"
fi

OVERRIDES=(
    DISTRIBUTED.enabled true
    IL.batch_size "${PER_GPU_BATCH_SIZE}"
    IL.lr "${LEARNING_RATE}"
    IL.epochs "${NUM_EPOCHS}"
    IL.OFFLINE.num_workers "${NUM_WORKERS}"
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

echo "=========================================="
echo "SatNav Offline Seq2Seq DDP Training"
echo "=========================================="
echo "Config        : ${CONFIG_PATH}"
echo "GPUs          : ${GPUS_PER_NODE} (${CUDA_DEVICES})"
echo "Batch         : ${PER_GPU_BATCH_SIZE} x ${GPUS_PER_NODE} = $((PER_GPU_BATCH_SIZE * GPUS_PER_NODE))"
echo "LR            : ${LEARNING_RATE}"
echo "Epochs        : ${NUM_EPOCHS}"
echo "Num Workers   : ${NUM_WORKERS}"
echo "Output Root   : ${OUTPUT_ROOT_ABS}"
echo "SwanLab       : ${USE_SWANLAB} (${SWANLAB_MODE_OVERRIDE})"
echo "WXWork Notice : ${USE_WXWORK_NOTIFICATION}"
echo "Experiment    : ${EXP_NAME}"
echo "Checkpoint    : ${CHECKPOINT_DIR}"
echo "Log File      : ${TRAIN_LOG_FILE}"
echo "=========================================="

if [ "${USE_WXWORK_NOTIFICATION}" = "true" ]; then
    echo "[WARN] WXWork notification depends on SwanLab callback compatibility."
    echo "[WARN] Current satnav env is Python 3.8; if callback init fails, training will continue without notifications."
fi

cd "${REPO_ROOT}"

"${DIST_LAUNCH[@]}" \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    run.py \
    --exp-config "${CONFIG_PATH}" \
    --run-type train \
    "${OVERRIDES[@]}" \
    2>&1 | tee "${TRAIN_LOG_FILE}"

# 训练成功后更新 latest 软链，指向本次 checkpoint 目录
# 使用相对目标（basename）确保可移植性
ln -sfn "$(basename "${CHECKPOINT_DIR}")" "${LATEST_LINK}"
echo "Updated latest -> $(basename "${CHECKPOINT_DIR}")"

#!/usr/bin/env bash
# Distributed NaVILA fine-tuning on SatNav trajectory_data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/navila/scripts/train.sh [continue|scratch] [options]

Options:
  --gpus N                 Processes/GPUs (default: 8)
  --trajectory-root PATH   Canonical trajectory_data directory
  --model-path PATH        Override mode-specific model
  --output-dir PATH        Exact output directory
  --max-steps N            Bounded smoke run
  --batch-size N
  --gradient-accumulation N
  --learning-rate VALUE
  --warmup-ratio VALUE
  --save-steps N
  --dataloader-workers N
  --train-components all|projector
  --max-episodes N         Deterministic annotation prefix for smoke
  --max-samples N          Deterministic training sample cap
  --head-keep N --sample-stride N --stop-repeat N
  --action-format compact|sentence
  --allow-upstream-mismatch
  -h, --help

The script never creates or mutates a conda environment. Prepare the isolated
environment exactly as documented in README.md.
EOF
}

MODE=continue
GPUS="${GPUS_PER_NODE:-8}"
TRAJECTORY_ROOT="${SATNAV_NAVILA_TRAIN_DATA:-}"
MODEL_PATH=""
OUTPUT_DIR=""
MAX_STEPS=""
BATCH_SIZE=""
GRAD_ACCUM=""
LEARNING_RATE=""
WARMUP_RATIO=""
SAVE_STEPS=""
DATALOADER_WORKERS=""
TRAIN_COMPONENTS=""
ALLOW_MISMATCH=false

if [[ $# -gt 0 && ( "$1" == continue || "$1" == scratch ) ]]; then
    MODE="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --trajectory-root) TRAJECTORY_ROOT="${2:?--trajectory-root requires a path}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --max-steps) MAX_STEPS="${2:?--max-steps requires a value}"; shift 2 ;;
        --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
        --gradient-accumulation) GRAD_ACCUM="${2:?--gradient-accumulation requires a value}"; shift 2 ;;
        --learning-rate) LEARNING_RATE="${2:?--learning-rate requires a value}"; shift 2 ;;
        --warmup-ratio) WARMUP_RATIO="${2:?--warmup-ratio requires a value}"; shift 2 ;;
        --save-steps) SAVE_STEPS="${2:?--save-steps requires a value}"; shift 2 ;;
        --dataloader-workers) DATALOADER_WORKERS="${2:?--dataloader-workers requires a value}"; shift 2 ;;
        --train-components) TRAIN_COMPONENTS="${2:?--train-components requires a value}"; shift 2 ;;
        --max-episodes) export SATNAV_MAX_EPISODES="${2:?--max-episodes requires a value}"; shift 2 ;;
        --max-samples) export SATNAV_MAX_SAMPLES="${2:?--max-samples requires a value}"; shift 2 ;;
        --head-keep) export SATNAV_HEAD_KEEP="${2:?--head-keep requires a value}"; shift 2 ;;
        --sample-stride) export SATNAV_SAMPLE_STRIDE="${2:?--sample-stride requires a value}"; shift 2 ;;
        --stop-repeat) export SATNAV_STOP_REPEAT="${2:?--stop-repeat requires a value}"; shift 2 ;;
        --action-format) export SATNAV_ACTION_FORMAT="${2:?--action-format requires a value}"; shift 2 ;;
        --allow-upstream-mismatch) ALLOW_MISMATCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    if [[ "${MODE}" == scratch ]]; then
        MODEL_PATH="${NAVILA_PRETRAIN_MODEL:-${NAVILA_MODEL_ROOT}/navila-siglip-llama3-8b-v1.5-pretrain}"
    else
        MODEL_PATH="${NAVILA_SFT_MODEL:-${NAVILA_MODEL_ROOT}/navila-llama3-8b-8f}"
    fi
fi
[[ -n "${TRAJECTORY_ROOT}" ]] || { echo "[ERROR] Set --trajectory-root or SATNAV_NAVILA_TRAIN_DATA." >&2; exit 2; }
for required in "${NAVILA_REPO}" "${MODEL_PATH}" "${TRAJECTORY_ROOT}/annotations.json"; do
    [[ -e "${required}" ]] || { echo "[ERROR] Required path not found: ${required}" >&2; exit 1; }
done
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_NAVILA_OUTPUT%/}/train/navila-${MODE}-$(date +%Y%m%d-%H%M%S)"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${SATNAV_ROOT}/${OUTPUT_DIR#./}"
fi
mkdir -p "${OUTPUT_DIR}"
navila_prepare_python
cd "${SATNAV_ROOT}"
ARGS=(
    --config "${NAVILA_BASELINE_DIR}/configs/train.yaml"
    --deepspeed-config "${NAVILA_BASELINE_DIR}/configs/zero2.json"
    --navila-repo "${NAVILA_REPO}"
    --model-path "${MODEL_PATH}"
    --trajectory-root "${TRAJECTORY_ROOT}"
    --data-validation-report "${OUTPUT_DIR}/data_validation.json"
    --output-dir "${OUTPUT_DIR}"
    --run-name "$(basename "${OUTPUT_DIR}")"
)
[[ -z "${MAX_STEPS}" ]] || ARGS+=(--max-steps "${MAX_STEPS}")
[[ -z "${BATCH_SIZE}" ]] || ARGS+=(--per-device-batch-size "${BATCH_SIZE}")
[[ -z "${GRAD_ACCUM}" ]] || ARGS+=(--gradient-accumulation-steps "${GRAD_ACCUM}")
[[ -z "${LEARNING_RATE}" ]] || ARGS+=(--learning-rate "${LEARNING_RATE}")
[[ -z "${WARMUP_RATIO}" ]] || ARGS+=(--warmup-ratio "${WARMUP_RATIO}")
[[ -z "${SAVE_STEPS}" ]] || ARGS+=(--save-steps "${SAVE_STEPS}")
[[ -z "${DATALOADER_WORKERS}" ]] || ARGS+=(--dataloader-num-workers "${DATALOADER_WORKERS}")
[[ -z "${TRAIN_COMPONENTS}" ]] || ARGS+=(--train-components "${TRAIN_COMPONENTS}")
[[ "${ALLOW_MISMATCH}" == false ]] || ARGS+=(--allow-upstream-mismatch)

export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SATNAV_VALIDATION_RUN_ID="${PPID}-$$-${RANDOM}-${RANDOM}"
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "[INFO] mode=${MODE} gpus=${GPUS} model=${MODEL_PATH} data=${TRAJECTORY_ROOT} output=${OUTPUT_DIR}"
"${NAVILA_PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS}" \
    -m baselines.vlm.navila.trainer "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

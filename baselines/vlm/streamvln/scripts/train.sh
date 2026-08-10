#!/usr/bin/env bash
# Distributed StreamVLN fine-tuning on SatNav trajectory_data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/streamvln/scripts/train.sh [continue|scratch] [options]

Options:
  --gpus N                 Processes/GPUs (default: 8)
  --trajectory-root PATH   Directory containing annotations.json and images/
  --model-path PATH        Override continue/scratch model selection
  --vision-tower PATH      Local SigLIP directory or explicit HF ID
  --output-dir PATH        Exact training output directory
  --epochs N               Training epochs (default from configs/train.yaml)
  --batch-size N           Per-device batch size
  --gradient-accumulation N
  --learning-rate VALUE
  --max-steps N            Optional bounded smoke run
  --dataloader-workers N   DataLoader worker count
  --no-data-augmentation   Disable stochastic image augmentation
  --no-torch-compile       Disable Torch compilation for a bounded smoke run
  --report-to none|swanlab
  --allow-upstream-mismatch
  -h, --help

This script never creates a conda environment.  Prepare the independent
environment from environment/conda.yml and README.md first.
EOF
}

MODE=continue
GPUS="${GPUS_PER_NODE:-8}"
TRAJECTORY_ROOT="${SATNAV_STREAMVLN_TRAIN_DATA:-}"
MODEL_PATH=""
VISION_TOWER="${STREAMVLN_VISION_TOWER:-${STREAMVLN_MODEL_ROOT}/siglip-so400m-patch14-384}"
OUTPUT_DIR=""
EPOCHS="${NUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
MAX_STEPS="${MAX_STEPS:-}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-}"
REPORT_TO="${STREAMVLN_REPORT_TO:-none}"
ALLOW_MISMATCH=false
DATA_AUGMENTATION=""
TORCH_COMPILE=""

if [[ $# -gt 0 && ( "$1" == continue || "$1" == scratch ) ]]; then
    MODE="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --trajectory-root) TRAJECTORY_ROOT="${2:?--trajectory-root requires a path}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --vision-tower) VISION_TOWER="${2:?--vision-tower requires a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --epochs) EPOCHS="${2:?--epochs requires a value}"; shift 2 ;;
        --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
        --gradient-accumulation) GRAD_ACCUM="${2:?--gradient-accumulation requires a value}"; shift 2 ;;
        --learning-rate) LEARNING_RATE="${2:?--learning-rate requires a value}"; shift 2 ;;
        --max-steps) MAX_STEPS="${2:?--max-steps requires a value}"; shift 2 ;;
        --dataloader-workers) DATALOADER_WORKERS="${2:?--dataloader-workers requires a value}"; shift 2 ;;
        --data-augmentation) DATA_AUGMENTATION=on; shift ;;
        --no-data-augmentation) DATA_AUGMENTATION=off; shift ;;
        --torch-compile) TORCH_COMPILE=on; shift ;;
        --no-torch-compile) TORCH_COMPILE=off; shift ;;
        --report-to) REPORT_TO="${2:?--report-to requires a value}"; shift 2 ;;
        --allow-upstream-mismatch) ALLOW_MISMATCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    if [[ "${MODE}" == scratch ]]; then
        MODEL_PATH="${STREAMVLN_MODEL_ROOT}/LLaVA-Video-7B-Qwen2"
    else
        MODEL_PATH="${STREAMVLN_MODEL_ROOT}/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
    fi
fi
if [[ -z "${TRAJECTORY_ROOT}" ]]; then
    echo "[ERROR] Set --trajectory-root or SATNAV_STREAMVLN_TRAIN_DATA." >&2
    exit 2
fi
for required in "${STREAMVLN_REPO}" "${MODEL_PATH}" "${TRAJECTORY_ROOT}"; do
    [[ -e "${required}" ]] || { echo "[ERROR] Required path not found: ${required}" >&2; exit 1; }
done
if [[ "${VISION_TOWER}" == /* || "${VISION_TOWER}" == ./* ]]; then
    [[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] Vision tower not found: ${VISION_TOWER}" >&2; exit 1; }
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_STREAMVLN_OUTPUT%/}/train/streamvln-${MODE}-${TIMESTAMP}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${SATNAV_ROOT}/${OUTPUT_DIR#./}"
fi
mkdir -p "${OUTPUT_DIR}"
streamvln_prepare_python
PYTHON_BIN="${STREAMVLN_PYTHON_BIN}"
# STREAMVLN_PYTHON may point into an environment that is not activated.  Keep
# its console tools (notably DeepSpeed's declared Ninja dependency) discoverable.
ARGS=(
    --config "${STREAMVLN_BASELINE_DIR}/configs/train.yaml"
    --deepspeed-config "${STREAMVLN_BASELINE_DIR}/configs/zero2.json"
    --streamvln-repo "${STREAMVLN_REPO}"
    --model-path "${MODEL_PATH}"
    --vision-tower "${VISION_TOWER}"
    --trajectory-root "${TRAJECTORY_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "$(basename "${OUTPUT_DIR}")"
    --report-to "${REPORT_TO}"
)
[[ -z "${EPOCHS}" ]] || ARGS+=(--num-train-epochs "${EPOCHS}")
[[ -z "${BATCH_SIZE}" ]] || ARGS+=(--per-device-batch-size "${BATCH_SIZE}")
[[ -z "${GRAD_ACCUM}" ]] || ARGS+=(--gradient-accumulation-steps "${GRAD_ACCUM}")
[[ -z "${LEARNING_RATE}" ]] || ARGS+=(--learning-rate "${LEARNING_RATE}")
[[ -z "${MAX_STEPS}" ]] || ARGS+=(--max-steps "${MAX_STEPS}")
[[ -z "${DATALOADER_WORKERS}" ]] || ARGS+=(--dataloader-num-workers "${DATALOADER_WORKERS}")
[[ "${DATA_AUGMENTATION}" != on ]] || ARGS+=(--data-augmentation)
[[ "${DATA_AUGMENTATION}" != off ]] || ARGS+=(--no-data-augmentation)
[[ "${TORCH_COMPILE}" != on ]] || ARGS+=(--torch-compile)
[[ "${TORCH_COMPILE}" != off ]] || ARGS+=(--no-torch-compile)
[[ "${ALLOW_MISMATCH}" == false ]] || ARGS+=(--allow-upstream-mismatch)

export STREAMVLN_OFFLINE="${STREAMVLN_OFFLINE:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
cd "${SATNAV_ROOT}"
echo "[INFO] mode=${MODE} gpus=${GPUS} data=${TRAJECTORY_ROOT} output=${OUTPUT_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${GPUS}" \
    -m baselines.vlm.streamvln.trainer \
    "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

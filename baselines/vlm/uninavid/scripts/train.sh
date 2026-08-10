#!/usr/bin/env bash
# Distributed Uni-NaVid fine-tuning on canonical SatNav trajectory_data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/uninavid/scripts/train.sh [options]

Required via option, exported variable, or .local/env.sh:
  --model-path PATH        Full Uni-NaVid checkpoint (UNINAVID_MODEL)
  --eva-path PATH          Verified eva_vit_g.pth (UNINAVID_EVA)
  --trajectory-root PATH   SatNav-v0.1 trajectory_data

Options:
  --gpus N                 Processes/GPUs (default: 8)
  --output-dir PATH        Exact output directory
  --resume                 Resume an existing matching checkpoint-* run
  --max-steps N            Bounded smoke run
  --batch-size N
  --gradient-accumulation N
  --learning-rate VALUE
  --warmup-ratio VALUE
  --save-steps N
  --save-strategy no|steps
  --dataloader-workers N
  --train-components all  Maintained full-checkpoint training contract
  --max-episodes N         Deterministic annotation prefix
  --max-samples N          Deterministic window prefix
  --disable-augmentation   Exact, deterministic smoke inputs
  --allow-upstream-mismatch
  --no-flash-attention
  -h, --help

Rank 0 always runs a fresh strict full-content v0.1 frame preflight before any
model load. It never creates or mutates a conda environment.
EOF
}

GPUS="${GPUS_PER_NODE:-8}"
MODEL_PATH="${UNINAVID_MODEL:-}"
EVA_PATH="${UNINAVID_EVA:-}"
TRAJECTORY_ROOT="${SATNAV_UNINAVID_TRAIN_DATA:-}"
OUTPUT_DIR=""
RESUME=false
ALLOW_MISMATCH=false
NO_FLASH=false
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --eva-path) EVA_PATH="${2:?--eva-path requires a path}"; shift 2 ;;
        --trajectory-root) TRAJECTORY_ROOT="${2:?--trajectory-root requires a path}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --max-steps) PY_ARGS+=(--max-steps "${2:?--max-steps requires a value}"); shift 2 ;;
        --batch-size) PY_ARGS+=(--per-device-batch-size "${2:?--batch-size requires a value}"); shift 2 ;;
        --gradient-accumulation) PY_ARGS+=(--gradient-accumulation-steps "${2:?--gradient-accumulation requires a value}"); shift 2 ;;
        --learning-rate) PY_ARGS+=(--learning-rate "${2:?--learning-rate requires a value}"); shift 2 ;;
        --warmup-ratio) PY_ARGS+=(--warmup-ratio "${2:?--warmup-ratio requires a value}"); shift 2 ;;
        --save-steps) PY_ARGS+=(--save-steps "${2:?--save-steps requires a value}"); shift 2 ;;
        --save-strategy) PY_ARGS+=(--save-strategy "${2:?--save-strategy requires a value}"); shift 2 ;;
        --dataloader-workers) PY_ARGS+=(--dataloader-num-workers "${2:?--dataloader-workers requires a value}"); shift 2 ;;
        --train-components) PY_ARGS+=(--train-components "${2:?--train-components requires a value}"); shift 2 ;;
        --max-episodes) export SATNAV_MAX_EPISODES="${2:?--max-episodes requires a value}"; shift 2 ;;
        --max-samples) export SATNAV_MAX_SAMPLES="${2:?--max-samples requires a value}"; shift 2 ;;
        --disable-augmentation) export SATNAV_UNINAVID_AUGMENTATION=false; shift ;;
        --allow-upstream-mismatch) ALLOW_MISMATCH=true; shift ;;
        --no-flash-attention) NO_FLASH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] --gpus must be positive." >&2; exit 2; }
[[ -n "${MODEL_PATH}" ]] || { echo "[ERROR] --model-path or UNINAVID_MODEL is required." >&2; exit 2; }
[[ -n "${EVA_PATH}" ]] || { echo "[ERROR] --eva-path or UNINAVID_EVA is required." >&2; exit 2; }
[[ -n "${TRAJECTORY_ROOT}" ]] || { echo "[ERROR] --trajectory-root is required." >&2; exit 2; }
uninavid_prepare_python
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}-${RANDOM}"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_UNINAVID_OUTPUT%/}/train/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
VALIDATION_REPORT="${OUTPUT_DIR}/data_validation.json"

export SATNAV_UNINAVID_RUN_ID="${RUN_ID}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

cd "${SATNAV_ROOT}"
ARGS=(
    --uninavid-repo "${UNINAVID_REPO}"
    --model-path "${MODEL_PATH}"
    --eva-path "${EVA_PATH}"
    --processor-path "${UNINAVID_PROCESSOR}"
    --trajectory-root "${TRAJECTORY_ROOT}"
    --data-validation-report "${VALIDATION_REPORT}"
    --output-dir "${OUTPUT_DIR}"
    "${PY_ARGS[@]}"
)
[[ "${RESUME}" == false ]] || ARGS+=(--resume)
[[ "${ALLOW_MISMATCH}" == false ]] || ARGS+=(--allow-upstream-mismatch)
[[ "${NO_FLASH}" == false ]] || ARGS+=(--no-flash-attention)

echo "[INFO] gpus=${GPUS} model=${MODEL_PATH} data=${TRAJECTORY_ROOT} output=${OUTPUT_DIR}"
"${UNINAVID_PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS}" \
    -m baselines.vlm.uninavid.trainer "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

CHECK_ARGS=(
    --model-path "${OUTPUT_DIR}"
    --compare-model "${MODEL_PATH}"
    --expected-train-components all
    --uninavid-repo "${UNINAVID_REPO}"
    --eva-path "${EVA_PATH}"
    --processor-path "${UNINAVID_PROCESSOR}"
    --device cuda:0
    --report "${OUTPUT_DIR}/post_train_checkpoint_report.json"
)
[[ "${NO_FLASH}" == false ]] || CHECK_ARGS+=(--no-flash-attention)
"${UNINAVID_PYTHON_BIN}" -m baselines.vlm.uninavid.checkpoint \
    "${CHECK_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/post_train_checkpoint_check.log"

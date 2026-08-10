#!/usr/bin/env bash
# Single- or multi-GPU OpenFly training with full data and checkpoint gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/openfly/scripts/train.sh --backend continue|scratch [options]

Required via option or .local/env.sh:
  --backend NAME          continue loads a complete HF snapshot; scratch loads native .pt
  --model-path PATH       HF snapshot, native run, or native checkpoints/*.pt
  --trajectory-root PATH Canonical SatNav-v0.1 trajectory_data

Options:
  --processor-path PATH   Required when scratch cannot resolve a nearby processor
  --gpus N                Processes/GPUs (default: 1)
  --output-dir PATH       Exact output directory
  --resume                Resume latest complete matching checkpoint-N
  --action-format FORMAT  compact or original
  --max-steps N           Bounded smoke; defaults save-steps to 1
  --save-steps N
  --batch-size N
  --gradient-accumulation N
  --learning-rate VALUE
  --dataloader-workers N
  --max-episodes N        Requires explicit bounded-data acknowledgement
  --max-samples N         Requires explicit bounded-data acknowledgement
  --no-flash-attention
  -h, --help

Rank 0 performs a fresh strict full-content validation of all canonical frames
before model loading. This script never creates or mutates a Python environment.
EOF
}

BACKEND=""
MODEL_PATH=""
PROCESSOR_PATH="${OPENFLY_PROCESSOR_PATH:-}"
TRAJECTORY_ROOT="${SATNAV_OPENFLY_TRAIN_DATA:-}"
OUTPUT_DIR=""
GPUS="${GPUS_PER_NODE:-1}"
RESUME=false
BOUNDED=false
MAX_STEPS_SET=false
SAVE_STEPS_SET=false
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?--backend requires a value}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --processor-path) PROCESSOR_PATH="${2:?--processor-path requires a path}"; shift 2 ;;
        --trajectory-root) TRAJECTORY_ROOT="${2:?--trajectory-root requires a path}"; shift 2 ;;
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --action-format) PY_ARGS+=(--action-format "${2:?--action-format requires a value}"); shift 2 ;;
        --max-steps) PY_ARGS+=(--max-steps "${2:?--max-steps requires a value}"); MAX_STEPS_SET=true; shift 2 ;;
        --save-steps) PY_ARGS+=(--save-steps "${2:?--save-steps requires a value}"); SAVE_STEPS_SET=true; shift 2 ;;
        --batch-size) PY_ARGS+=(--per-device-batch-size "${2:?--batch-size requires a value}"); shift 2 ;;
        --gradient-accumulation) PY_ARGS+=(--gradient-accumulation-steps "${2:?--gradient-accumulation requires a value}"); shift 2 ;;
        --learning-rate) PY_ARGS+=(--learning-rate "${2:?--learning-rate requires a value}"); shift 2 ;;
        --dataloader-workers) PY_ARGS+=(--dataloader-num-workers "${2:?--dataloader-workers requires a value}"); shift 2 ;;
        --max-episodes) export SATNAV_MAX_EPISODES="${2:?--max-episodes requires a value}"; BOUNDED=true; shift 2 ;;
        --max-samples) export SATNAV_MAX_SAMPLES="${2:?--max-samples requires a value}"; BOUNDED=true; shift 2 ;;
        --no-flash-attention) PY_ARGS+=(--no-flash-attention); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${BACKEND}" == continue || "${BACKEND}" == scratch ]] || {
    echo "[ERROR] --backend must be continue or scratch." >&2; exit 2;
}
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ "${BACKEND}" == continue ]]; then
        MODEL_PATH="${OPENFLY_CONTINUE_MODEL:-}"
    else
        MODEL_PATH="${OPENFLY_NATIVE_RUN:-}"
    fi
fi
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] --gpus must be positive." >&2; exit 2; }
[[ -n "${MODEL_PATH}" ]] || { echo "[ERROR] --model-path is required." >&2; exit 2; }
[[ -n "${TRAJECTORY_ROOT}" ]] || { echo "[ERROR] --trajectory-root is required." >&2; exit 2; }
if [[ "${MAX_STEPS_SET}" == true && "${SAVE_STEPS_SET}" == false ]]; then
    PY_ARGS+=(--save-steps 1)
fi
[[ "${BOUNDED}" == false ]] || PY_ARGS+=(--allow-bounded-data)
openfly_prepare_python

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}-${RANDOM}"
if [[ "${RESUME}" == true ]]; then
    [[ -n "${OUTPUT_DIR}" && -d "${OUTPUT_DIR}" ]] || {
        echo "[ERROR] --resume requires an existing --output-dir." >&2; exit 2;
    }
else
    OUTPUT_DIR="${OUTPUT_DIR:-${SATNAV_OPENFLY_OUTPUT%/}/train/${BACKEND}-${RUN_ID}}"
    if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
        echo "[ERROR] Fresh output exists and is not a directory: ${OUTPUT_DIR}" >&2
        exit 2
    fi
    if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
        echo "[ERROR] Fresh output must be nonexistent or strictly empty: ${OUTPUT_DIR}" >&2
        exit 2
    fi
fi
mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}.logs"
mkdir -p "${LOG_DIR}"
export SATNAV_VALIDATION_RUN_ID="${RUN_ID}"
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

ARGS=(
    --backend "${BACKEND}"
    --model-path "${MODEL_PATH}"
    --trajectory-root "${TRAJECTORY_ROOT}"
    --data-validation-report "${OUTPUT_DIR}/data_validation.json"
    --output-dir "${OUTPUT_DIR}"
    "${PY_ARGS[@]}"
)
[[ -z "${PROCESSOR_PATH}" ]] || ARGS+=(--processor-path "${PROCESSOR_PATH}")
[[ "${RESUME}" == false ]] || ARGS+=(--resume-from-checkpoint latest)

cd "${SATNAV_ROOT}"
"${OPENFLY_PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS}" \
    -m baselines.vlm.openfly.trainer "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

CHECKPOINT_ARGS=(
    --model-path "${OUTPUT_DIR}"
    --device cuda:0
    --dtype bfloat16
    --report "${LOG_DIR}/strict_reload_report.json"
)
if [[ "${BACKEND}" == continue ]]; then
    CHECKPOINT_ARGS+=(--compare-model "${MODEL_PATH}")
else
    CACHE_ID="$(
        "${OPENFLY_PYTHON_BIN}" -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["hf_cache_id"])' \
            "${OUTPUT_DIR}/backend_meta.json"
    )"
    [[ "${CACHE_ID}" =~ ^openfly-native-[0-9a-f]{24}$ ]] || {
        echo "[ERROR] Scratch backend emitted an invalid comparison cache id." >&2
        exit 2
    }
    COMPARISON_MODEL="${OUTPUT_DIR}.source-cache/${CACHE_ID}"
    [[ -d "${COMPARISON_MODEL}" ]] || {
        echo "[ERROR] Scratch comparison model is missing: ${COMPARISON_MODEL}" >&2
        exit 2
    }
    CHECKPOINT_ARGS+=(--compare-model "${COMPARISON_MODEL}")
fi
"${OPENFLY_PYTHON_BIN}" -m baselines.vlm.openfly.checkpoint \
    "${CHECKPOINT_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/strict_reload.log"

#!/usr/bin/env bash
# Single- or multi-GPU NaVILA rollout through satnav.evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/navila/scripts/eval.sh --model-path PATH [options]

Launcher options:
  --gpus N                 Processes/GPUs (default: 1)
  -h, --help

Evaluator options are forwarded, including:
  --split val_seen|val_unseen
  --episodes PATH --scenes-dir PATH --output-dir PATH
  --offset N --limit N --base-seed N --max-steps N
  --resume --fail-fast --fail-on-episode-error
  --action-format compact|sentence --dry-run

Every rank receives the full episode list; satnav.evaluation owns deterministic
selection/sharding, rank-local JSONL/done markers, resume, and aggregation.
EOF
}

GPUS="${EVAL_GPUS:-1}"
MODEL_PATH=""
OUTPUT_DIR=""
SPLIT=val_seen
DRY_RUN=false
FAIL_ON_ERROR=false
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; PY_ARGS+=("$1" "$2"); shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; PY_ARGS+=("$1" "$2"); shift 2 ;;
        --split) SPLIT="${2:?--split requires a value}"; PY_ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; PY_ARGS+=("$1"); shift ;;
        --fail-on-episode-error) FAIL_ON_ERROR=true; PY_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done
[[ -n "${MODEL_PATH}" ]] || { echo "[ERROR] --model-path is required." >&2; exit 2; }
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_NAVILA_OUTPUT%/}/eval/$(basename "${MODEL_PATH}")/${SPLIT}"
    PY_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
PY_ARGS+=(--navila-repo "${NAVILA_REPO}")
navila_prepare_python
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${SATNAV_ROOT}"
if [[ "${DRY_RUN}" == true || "${GPUS}" -eq 1 ]]; then
    "${NAVILA_PYTHON_BIN}" -m baselines.vlm.navila.evaluate "${PY_ARGS[@]}"
else
    "${NAVILA_PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nproc_per_node="${GPUS}" \
        -m baselines.vlm.navila.evaluate "${PY_ARGS[@]}"
    AGGREGATE_ARGS=("${OUTPUT_DIR}")
    [[ "${FAIL_ON_ERROR}" == false ]] || AGGREGATE_ARGS+=(--fail-on-episode-error)
    "${NAVILA_PYTHON_BIN}" scripts/evaluation/aggregate.py "${AGGREGATE_ARGS[@]}"
fi

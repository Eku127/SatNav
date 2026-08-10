#!/usr/bin/env bash
# Deterministic single- or multi-GPU OpenFly evaluation via satnav.evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/openfly/scripts/eval.sh --model-path PATH [options]

Options:
  --split val_seen|val_unseen|test
  --episodes PATH
  --scenes-dir PATH
  --output-dir PATH
  --gpus N                 Processes/GPUs (default: 1)
  --limit N
  --max-steps N
  --action-format auto|compact|original
  --resume
  --fail-on-episode-error
  --dry-run
  -h, --help
EOF
}

MODEL_PATH="${OPENFLY_CONTINUE_MODEL:-}"
SPLIT=val_seen
EPISODES="${SATNAV_OPENFLY_EVAL_EPISODES:-}"
SCENES="${SATNAV_OPENFLY_SCENES_DIR:-}"
OUTPUT_DIR=""
GPUS="${GPUS_PER_NODE:-1}"
FAIL_ERRORS=false
DRY_RUN=false
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --split) SPLIT="${2:?--split requires a value}"; shift 2 ;;
        --episodes) EPISODES="${2:?--episodes requires a path}"; shift 2 ;;
        --scenes-dir) SCENES="${2:?--scenes-dir requires a path}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --limit) PY_ARGS+=(--limit "${2:?--limit requires a value}"); shift 2 ;;
        --max-steps) PY_ARGS+=(--max-steps "${2:?--max-steps requires a value}"); shift 2 ;;
        --action-format) PY_ARGS+=(--action-format "${2:?--action-format requires a value}"); shift 2 ;;
        --resume) PY_ARGS+=(--resume); shift ;;
        --fail-on-episode-error) PY_ARGS+=(--fail-on-episode-error); FAIL_ERRORS=true; shift ;;
        --dry-run) PY_ARGS+=(--dry-run); DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] --gpus must be positive." >&2; exit 2; }
[[ -n "${MODEL_PATH}" ]] || { echo "[ERROR] --model-path is required." >&2; exit 2; }
if [[ "${DRY_RUN}" == false && "${GPUS}" -gt 1 && -z "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] multi-GPU evaluation requires explicit --output-dir." >&2
    exit 2
fi
openfly_prepare_python
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

ARGS=(--model-path "${MODEL_PATH}" --split "${SPLIT}" "${PY_ARGS[@]}")
[[ -z "${EPISODES}" ]] || ARGS+=(--episodes "${EPISODES}")
[[ -z "${SCENES}" ]] || ARGS+=(--scenes-dir "${SCENES}")
[[ -z "${OUTPUT_DIR}" ]] || ARGS+=(--output-dir "${OUTPUT_DIR}")

cd "${SATNAV_ROOT}"
"${OPENFLY_PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS}" \
    -m baselines.vlm.openfly.evaluate "${ARGS[@]}"

if [[ "${DRY_RUN}" == false && "${GPUS}" -gt 1 ]]; then
    AGG_ARGS=("${OUTPUT_DIR}")
    [[ "${FAIL_ERRORS}" == false ]] || AGG_ARGS+=(--fail-on-episode-error)
    "${OPENFLY_PYTHON_BIN}" scripts/evaluation/aggregate.py "${AGG_ARGS[@]}"
fi

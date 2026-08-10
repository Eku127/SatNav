#!/usr/bin/env bash
# Single- or multi-GPU StreamVLN rollout through satnav.evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/streamvln/scripts/eval.sh [launcher options] --model-path PATH [evaluator options]

Launcher options:
  --gpus N                 Processes/GPUs (default: 1)
  -h, --help               Show launcher help

Common evaluator options:
  --split val_seen|val_unseen
  --benchmark-manifest PATH
  --episodes PATH --scenes-dir PATH
  --output-dir PATH
  --offset N --limit N --base-seed N
  --resume --fail-fast --fail-on-episode-error --no-action-trace
  --vision-tower PATH --tokenizer-path PATH
  --dry-run

Every rank receives the full episode list.  satnav.evaluation performs stable
sort/offset/limit/strided sharding and writes rank-local JSONL + done markers;
this launcher aggregates only after all multi-GPU workers exit successfully.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
GPUS="${EVAL_GPUS:-1}"
MODEL_PATH=""
OUTPUT_DIR=""
SPLIT=val_seen
VISION_TOWER=""
FAIL_ON_ERROR=false
DRY_RUN=false
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --model-path)
            MODEL_PATH="${2:?--model-path requires a path}"
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --output-dir)
            OUTPUT_DIR="${2:?--output-dir requires a path}"
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --split)
            SPLIT="${2:?--split requires a value}"
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --vision-tower)
            VISION_TOWER="${2:?--vision-tower requires a value}"
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --fail-on-episode-error)
            FAIL_ON_ERROR=true; PY_ARGS+=("$1"); shift ;;
        --dry-run)
            DRY_RUN=true; PY_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] --model-path is required." >&2
    exit 2
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_STREAMVLN_OUTPUT%/}/eval/$(basename "${MODEL_PATH}")/${SPLIT}"
    PY_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -z "${VISION_TOWER}" ]]; then
    DEFAULT_VISION_TOWER="${STREAMVLN_VISION_TOWER:-${STREAMVLN_MODEL_ROOT}/siglip-so400m-patch14-384}"
    if [[ -d "${DEFAULT_VISION_TOWER}" ]]; then
        PY_ARGS+=(--vision-tower "${DEFAULT_VISION_TOWER}")
    fi
fi
PY_ARGS+=(--streamvln-repo "${STREAMVLN_REPO}")
streamvln_prepare_python
PYTHON_BIN="${STREAMVLN_PYTHON_BIN}"
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
cd "${SATNAV_ROOT}"

if [[ "${DRY_RUN}" == true || "${GPUS}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m baselines.vlm.streamvln.evaluate "${PY_ARGS[@]}"
else
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${GPUS}" \
        -m baselines.vlm.streamvln.evaluate \
        "${PY_ARGS[@]}"
    AGGREGATE_ARGS=("${OUTPUT_DIR}")
    [[ "${FAIL_ON_ERROR}" == false ]] || AGGREGATE_ARGS+=(--fail-on-episode-error)
    "${PYTHON_BIN}" scripts/evaluation/aggregate.py "${AGGREGATE_ARGS[@]}"
fi

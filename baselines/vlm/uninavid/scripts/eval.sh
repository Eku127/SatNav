#!/usr/bin/env bash
# Single- or multi-GPU Uni-NaVid rollout through satnav.evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/uninavid/scripts/eval.sh --model-path PATH [options]

Required via option, exported variable, or .local/env.sh:
  --model-path PATH        Full Uni-NaVid checkpoint
  --eva-path PATH          Verified eva_vit_g.pth

Launcher options:
  --gpus N                 Processes/GPUs (default: 1)
  -h, --help

Evaluator options are forwarded, including:
  --split val_seen|val_unseen --benchmark-manifest PATH
  --episodes PATH --scenes-dir PATH --output-dir PATH
  --offset N --limit N --base-seed N --max-steps N
  --resume --fail-fast --fail-on-episode-error
  --max-new-tokens N --no-flash-attention --dry-run

Every rank receives the same selected episode list. satnav.evaluation owns
stable-key sharding, rank-local JSONL/done markers, resume, and aggregation.
EOF
}

GPUS="${EVAL_GPUS:-1}"
MODEL_PATH="${UNINAVID_MODEL:-}"
EVA_PATH="${UNINAVID_EVA:-}"
OUTPUT_DIR=""
SPLIT=val_seen
DRY_RUN=false
FAIL_ON_ERROR=false
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a path}"; shift 2 ;;
        --eva-path) EVA_PATH="${2:?--eva-path requires a path}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        --split) SPLIT="${2:?--split requires a value}"; PY_ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; PY_ARGS+=("$1"); shift ;;
        --fail-on-episode-error) FAIL_ON_ERROR=true; PY_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] --gpus must be positive." >&2; exit 2; }
[[ -n "${MODEL_PATH}" ]] || { echo "[ERROR] --model-path or UNINAVID_MODEL is required." >&2; exit 2; }
[[ -n "${EVA_PATH}" ]] || { echo "[ERROR] --eva-path or UNINAVID_EVA is required." >&2; exit 2; }
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SATNAV_UNINAVID_OUTPUT%/}/eval/$(basename "${MODEL_PATH}")/${SPLIT}"
fi
uninavid_prepare_python
PY_ARGS+=(
    --model-path "${MODEL_PATH}"
    --eva-path "${EVA_PATH}"
    --processor-path "${UNINAVID_PROCESSOR}"
    --uninavid-repo "${UNINAVID_REPO}"
    --output-dir "${OUTPUT_DIR}"
)

export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ -z "${CUDA_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
cd "${SATNAV_ROOT}"
if [[ "${DRY_RUN}" == true || "${GPUS}" -eq 1 ]]; then
    "${UNINAVID_PYTHON_BIN}" -m baselines.vlm.uninavid.evaluate "${PY_ARGS[@]}"
else
    "${UNINAVID_PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nproc_per_node="${GPUS}" \
        -m baselines.vlm.uninavid.evaluate "${PY_ARGS[@]}"
    AGGREGATE_ARGS=("${OUTPUT_DIR}")
    [[ "${FAIL_ON_ERROR}" == false ]] || AGGREGATE_ARGS+=(--fail-on-episode-error)
    "${UNINAVID_PYTHON_BIN}" scripts/evaluation/aggregate.py "${AGGREGATE_ARGS[@]}"
fi

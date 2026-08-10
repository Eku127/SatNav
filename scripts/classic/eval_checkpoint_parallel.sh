#!/usr/bin/env bash
# Parallel compatibility front-end for neural classic evaluation.

set -euo pipefail

METHOD="${1:-}"
INPUT="${2:-}"
SPLIT="${3:-val_seen}"
WORLD_SIZE_ARG="${4:-8}"
GPU_LIST="${5:-0,1,2,3,4,5,6,7}"
LIMIT="${6:--1}"
if [[ "${METHOD}" != "seq2seq" && "${METHOD}" != "cma" ]] || [[ -z "${INPUT}" ]]; then
    echo "Usage: bash scripts/classic/eval_checkpoint_parallel.sh METHOD EXP_OR_CKPT [split] [world_size] [gpu_list] [limit]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${REPO_ROOT}/scripts/${METHOD}"

# Child ranks activate their own environment in eval_checkpoint.sh.  The
# parent must activate it as well before invoking the aggregate CLI.
CONDA_INIT="${CONDA_INIT:-${SATNAV_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-${SATNAV_CONDA_ENV:-satnav}}"
if [[ -f "${CONDA_INIT}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV}"
fi

if [[ "${METHOD}" == "seq2seq" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-${SEQ2SEQ_OUTPUT_ROOT:-output/seq2seq_offline}}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-${CMA_OUTPUT_ROOT:-output/cma}}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
if [[ "${INPUT}" == /* || "${INPUT}" == *.pth ]]; then
    EXPERIMENT_NAME="$(basename "$(dirname "${INPUT}")")"
    DEFAULT_OUTPUT="${OUTPUT_ROOT}/results/by-path/${EXPERIMENT_NAME}/${SPLIT}/generic/${WORLD_SIZE_ARG}rank"
else
    DEFAULT_OUTPUT="${OUTPUT_ROOT}/results/${INPUT}/${SPLIT}/generic/${WORLD_SIZE_ARG}rank"
fi
RUN_OUTPUT="${SATNAV_CLASSIC_RUN_OUTPUT:-${DEFAULT_OUTPUT}}"
mkdir -p "$(dirname "${RUN_OUTPUT}")"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${WORLD_SIZE_ARG}" ]]; then
    echo "gpu_list contains ${#GPUS[@]} entries for world_size=${WORLD_SIZE_ARG}" >&2
    exit 2
fi

declare -a PIDS=()
cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

for ((rank=0; rank<WORLD_SIZE_ARG; rank++)); do
    CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" \
    CUDA_DEVICES="${GPUS[$rank]}" \
    SATNAV_RANK="${rank}" \
    SATNAV_WORLD_SIZE="${WORLD_SIZE_ARG}" \
    SATNAV_CLASSIC_RUN_OUTPUT="${RUN_OUTPUT}" \
        bash "${SCRIPT_DIR}/eval_checkpoint.sh" \
        "${METHOD}" "${INPUT}" "${SPLIT}" "${LIMIT}" \
        > "${RUN_OUTPUT}.rank-${rank}.log" 2>&1 &
    PIDS[$rank]=$!
done

FAILED=0
for ((rank=0; rank<WORLD_SIZE_ARG; rank++)); do
    if ! wait "${PIDS[$rank]}"; then
        echo "rank ${rank} failed; inspect ${RUN_OUTPUT}.rank-${rank}.log" >&2
        FAILED=$((FAILED + 1))
    fi
done
trap - INT TERM
if [[ "${FAILED}" -ne 0 ]]; then
    exit 1
fi

cd "${REPO_ROOT}"
python scripts/evaluation/aggregate.py "${RUN_OUTPUT}" --fail-on-episode-error

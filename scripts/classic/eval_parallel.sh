#!/usr/bin/env bash
# Launch strided rank-local generic evaluation and validate the merged union.

set -euo pipefail

METHOD="${1:-}"
SPLIT="${2:-val_seen}"
WORLD_SIZE_ARG="${3:-2}"
GPU_LIST="${4:-0,1}"
LIMIT="${5:--1}"

if [[ -z "${METHOD}" || ! "${WORLD_SIZE_ARG}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/classic/eval_parallel.sh METHOD [split] [world_size] [gpu_list] [limit]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${REPO_ROOT}/baselines/classic"

# Child ranks activate their own environment in eval.sh.  The parent process
# also needs the same interpreter for the final aggregation step.
CONDA_INIT="${CONDA_INIT:-${SATNAV_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-${SATNAV_CONDA_ENV:-satnav}}"
if [[ -f "${CONDA_INIT}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV}"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${WORLD_SIZE_ARG}" ]]; then
    echo "gpu_list contains ${#GPUS[@]} entries for world_size=${WORLD_SIZE_ARG}" >&2
    exit 2
fi

MAX_STEPS="${SATNAV_MAX_STEPS:-5}"
OUTPUT_ROOT="${SATNAV_CLASSIC_OUTPUT:-output/baselines/classic}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
NORMALIZED_METHOD="${METHOD}"
if [[ "${METHOD}" == "reference" ]]; then
    NORMALIZED_METHOD="reference_follower"
fi
RUN_OUTPUT="${SATNAV_CLASSIC_RUN_OUTPUT:-${OUTPUT_ROOT}/${NORMALIZED_METHOD}/${SPLIT}/${MAX_STEPS}steps/${WORLD_SIZE_ARG}rank}"
if [[ "${RUN_OUTPUT}" != /* ]]; then
    RUN_OUTPUT="${REPO_ROOT}/${RUN_OUTPUT}"
fi
mkdir -p "$(dirname "${RUN_OUTPUT}")"

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
        bash "${SCRIPT_DIR}/eval.sh" "${METHOD}" "${SPLIT}" "${LIMIT}" \
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

#!/usr/bin/env bash
# Compatibility front-end for neural classic evaluation by experiment/checkpoint.

set -euo pipefail

METHOD="${1:-}"
INPUT="${2:-}"
SPLIT="${3:-val_seen}"
LIMIT="${4:--1}"
if [[ "${METHOD}" != "seq2seq" && "${METHOD}" != "cma" ]] || [[ -z "${INPUT}" ]]; then
    echo "Usage: bash scripts/classic/eval_checkpoint.sh {seq2seq|cma} EXP_OR_CKPT [split] [limit]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${REPO_ROOT}/scripts/${METHOD}"

if [[ "${METHOD}" == "seq2seq" ]]; then
    CONFIG_PATH="${CONFIG_PATH:-${SEQ2SEQ_EVAL_CONFIG_PATH:-configs/baselines/seq2seq_eval.yaml}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${SEQ2SEQ_OUTPUT_ROOT:-output/seq2seq_offline}}"
    METHOD_CUDA_DEVICES="${SEQ2SEQ_CUDA_DEVICES:-}"
else
    CONFIG_PATH="${CONFIG_PATH:-${CMA_EVAL_CONFIG_PATH:-configs/baselines/cma_eval.yaml}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${CMA_OUTPUT_ROOT:-output/cma}}"
    METHOD_CUDA_DEVICES="${CMA_CUDA_DEVICES:-}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

if [[ "${INPUT}" == /* || "${INPUT}" == *.pth ]]; then
    CHECKPOINT_PATH="${INPUT}"
    if [[ "${CHECKPOINT_PATH}" != /* ]]; then
        CHECKPOINT_PATH="${REPO_ROOT}/${CHECKPOINT_PATH}"
    fi
    EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
    DEFAULT_OUTPUT="${OUTPUT_ROOT}/results/by-path/${EXPERIMENT_NAME}/${SPLIT}/generic/${SATNAV_WORLD_SIZE:-1}rank"
else
    EXPERIMENT_NAME="${INPUT}"
    CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}/best.pth"
    DEFAULT_OUTPUT="${OUTPUT_ROOT}/results/${EXPERIMENT_NAME}/${SPLIT}/generic/${SATNAV_WORLD_SIZE:-1}rank"
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 2
fi
RUN_OUTPUT="${SATNAV_CLASSIC_RUN_OUTPUT:-${DEFAULT_OUTPUT}}"
CUDA_DEVICES="${CUDA_DEVICES:-${METHOD_CUDA_DEVICES}}"
if [[ -n "${CUDA_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

CONDA_INIT="${CONDA_INIT:-${SATNAV_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-${SATNAV_CONDA_ENV:-satnav}}"
if [[ -f "${CONDA_INIT}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV}"
fi

OPTIONAL_ARGS=()
if [[ -n "${SATNAV_VOCAB_PATH:-}" ]]; then
    OPTIONAL_ARGS+=(--vocab "${SATNAV_VOCAB_PATH}")
fi
if [[ -n "${SATNAV_BENCHMARK_PATH:-}" ]]; then
    OPTIONAL_ARGS+=(--benchmark "${SATNAV_BENCHMARK_PATH}")
fi
if [[ "${SATNAV_CLASSIC_RESUME:-true}" == "true" ]]; then
    OPTIONAL_ARGS+=(--resume)
fi

cd "${REPO_ROOT}"
python -m baselines.classic \
    --method "${METHOD}" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${RUN_OUTPUT}" \
    --split "${SPLIT}" \
    --limit "${LIMIT}" \
    --rank "${SATNAV_RANK:-${RANK:-0}}" \
    --world-size "${SATNAV_WORLD_SIZE:-${WORLD_SIZE:-1}}" \
    --seed "${SATNAV_CLASSIC_SEED:-0}" \
    --device "${SATNAV_DEVICE:-cuda:0}" \
    "${OPTIONAL_ARGS[@]}"

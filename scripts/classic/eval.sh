#!/usr/bin/env bash
# Unified canonical v0.1 evaluation for all four classic methods.

set -euo pipefail

METHOD="${1:-}"
SPLIT="${2:-val_seen}"
LIMIT="${3:--1}"

if [[ -z "${METHOD}" ]]; then
    echo "Usage: bash scripts/classic/eval.sh {random|reference_follower|seq2seq|cma} [val_seen|val_unseen] [limit]" >&2
    exit 2
fi
if [[ "${SPLIT}" != "val_seen" && "${SPLIT}" != "val_unseen" ]]; then
    echo "classic canonical eval supports val_seen or val_unseen, got: ${SPLIT}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${REPO_ROOT}/baselines/classic"

DATA_ROOT="${SATNAV_DATA_ROOT:-}"
if [[ -z "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
    echo "SATNAV_DATA_ROOT must point to an extracted SatNav-v0.1 directory" >&2
    exit 2
fi
SCENES_DIR="${SATNAV_SCENES_DIR:-${DATA_ROOT}/../scenes}"
EPISODES_PATH="${SATNAV_EPISODES_PATH:-${DATA_ROOT}/episodes/eval/${SPLIT}/all_episodes.json}"
if [[ ! -f "${EPISODES_PATH}" ]]; then
    echo "episode file not found: ${EPISODES_PATH}" >&2
    exit 2
fi
if [[ ! -d "${SCENES_DIR}" ]]; then
    echo "scene directory not found: ${SCENES_DIR}" >&2
    exit 2
fi

BENCHMARK_KIND="${SATNAV_BENCHMARK_KIND:-smoke}"
case "${BENCHMARK_KIND}" in
    smoke)
        BENCHMARK_PATH="configs/benchmark/satnav_v0_1_${SPLIT}_smoke.json"
        ;;
    official)
        BENCHMARK_PATH="configs/benchmark/satnav_v0_1_${SPLIT}.json"
        ;;
    *)
        echo "SATNAV_BENCHMARK_KIND must be smoke or official" >&2
        exit 2
        ;;
esac

CHECKPOINT_ARGS=()
case "${METHOD}" in
    random)
        CONFIG_PATH="${SATNAV_RANDOM_EVAL_CONFIG:-configs/baselines/random_agent.yaml}"
        ;;
    reference|reference_follower)
        METHOD="reference_follower"
        CONFIG_PATH="${SATNAV_REFERENCE_EVAL_CONFIG:-configs/baselines/reference_follower.yaml}"
        ;;
    seq2seq)
        CONFIG_PATH="${SATNAV_SEQ2SEQ_EVAL_CONFIG:-configs/baselines/seq2seq_eval.yaml}"
        CHECKPOINT_PATH="${SATNAV_SEQ2SEQ_CHECKPOINT:-}"
        if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
            echo "SATNAV_SEQ2SEQ_CHECKPOINT must name an existing checkpoint" >&2
            exit 2
        fi
        if [[ -z "${SATNAV_VOCAB_PATH:-}" || ! -f "${SATNAV_VOCAB_PATH}" ]]; then
            echo "SATNAV_VOCAB_PATH must name the checkpoint's vocabulary" >&2
            exit 2
        fi
        CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT_PATH}" --vocab "${SATNAV_VOCAB_PATH}")
        ;;
    cma)
        CONFIG_PATH="${SATNAV_CMA_EVAL_CONFIG:-configs/baselines/cma_eval.yaml}"
        CHECKPOINT_PATH="${SATNAV_CMA_CHECKPOINT:-}"
        if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
            echo "SATNAV_CMA_CHECKPOINT must name an existing checkpoint" >&2
            exit 2
        fi
        if [[ -z "${SATNAV_VOCAB_PATH:-}" || ! -f "${SATNAV_VOCAB_PATH}" ]]; then
            echo "SATNAV_VOCAB_PATH must name the checkpoint's vocabulary" >&2
            exit 2
        fi
        CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT_PATH}" --vocab "${SATNAV_VOCAB_PATH}")
        ;;
    *)
        echo "unknown classic method: ${METHOD}" >&2
        exit 2
        ;;
esac

RANK="${SATNAV_RANK:-${RANK:-0}}"
WORLD_SIZE="${SATNAV_WORLD_SIZE:-${WORLD_SIZE:-1}}"
OUTPUT_ROOT="${SATNAV_CLASSIC_OUTPUT:-output/baselines/classic}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
OUTPUT_DIR="${SATNAV_CLASSIC_RUN_OUTPUT:-${OUTPUT_ROOT}/${METHOD}/${SPLIT}/${BENCHMARK_KIND}/${WORLD_SIZE}rank}"
DEVICE="${SATNAV_DEVICE:-cuda:0}"
CUDA_DEVICES="${CUDA_DEVICES:-${SATNAV_CUDA_DEVICES:-}}"
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

RESUME_ARGS=()
if [[ "${SATNAV_CLASSIC_RESUME:-true}" == "true" ]]; then
    RESUME_ARGS=(--resume)
fi

cd "${REPO_ROOT}"
python -m baselines.classic \
    --method "${METHOD}" \
    --config "${CONFIG_PATH}" \
    --benchmark "${BENCHMARK_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --limit "${LIMIT}" \
    --rank "${RANK}" \
    --world-size "${WORLD_SIZE}" \
    --seed "${SATNAV_CLASSIC_SEED:-0}" \
    --device "${DEVICE}" \
    --set "BASE_TASK_CONFIG_PATH=configs/satnav_eval_task.yaml" \
    --set "DATASET.DATA_PATH=${EPISODES_PATH}" \
    --set "DATASET.SCENES_DIR=${SCENES_DIR}" \
    --set "DATASET.SPLIT=${SPLIT}" \
    --set "EVAL.SPLIT=${SPLIT}" \
    --set 'TASK.MEASUREMENTS=[DISTANCE_TO_GOAL,SUCCESS,ORACLE_SUCCESS,SPL,PATH_LENGTH]' \
    "${CHECKPOINT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

#!/usr/bin/env bash
# ==============================================================================
# Parallel Seq2Seq Evaluation — splits one eval split across N GPUs.
#
# Usage:
#   bash scripts/seq2seq/eval_parallel.sh <exp_name_or_ckpt> [split] [num_gpus] [gpu_list] [max_episodes]
#
# Examples:
#   bash scripts/seq2seq/eval_parallel.sh seq2seq-exp val_seen 8 0,1,2,3,4,5,6,7
#   bash scripts/seq2seq/eval_parallel.sh seq2seq-exp val_unseen 8 0,1,2,3,4,5,6,7
#   bash scripts/seq2seq/eval_parallel.sh /abs/path/to/best.pth val_seen 4 0,1,2,3 200
#
# Output:
#   output/seq2seq_offline/results/<exp_name>/<split>/eval_ckpt_0_<split>.json  (merged)
#   output/seq2seq_offline/results/<exp_name>/<split>/shard_<i>/                (per-shard)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

INPUT="${1:-}"
SPLIT="${2:-val_seen}"
NUM_GPUS="${3:-8}"
GPU_LIST="${4:-0,1,2,3,4,5,6,7}"
MAX_EPISODES="${5:-}"

if [ -z "${INPUT}" ]; then
    print_error "Usage: bash scripts/seq2seq/eval_parallel.sh <exp_name_or_ckpt> [split] [num_gpus] [gpu_list] [max_episodes]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${REPO_ROOT}/scripts/seq2seq"

CONFIG_PATH="${CONFIG_PATH:-${SEQ2SEQ_EVAL_CONFIG_PATH:-configs/baselines/seq2seq_eval.yaml}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SEQ2SEQ_OUTPUT_ROOT:-output/seq2seq_offline}}"

EVAL_MODE="by_name"
if [[ "${INPUT}" = /* ]] || [[ "${INPUT}" = *.pth ]]; then
    EVAL_MODE="by_path"
fi

if [ "${EVAL_MODE}" = "by_name" ]; then
    EXP_NAME="${INPUT}"
    CKPT_PATH="${REPO_ROOT}/${OUTPUT_ROOT}/checkpoints/${EXP_NAME}/best.pth"
    if [ ! -f "${CKPT_PATH}" ]; then
        print_error "Checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi
    OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}/results/${EXP_NAME}"
else
    CKPT_PATH="${INPUT}"
    if [ ! -f "${CKPT_PATH}" ]; then
        print_error "Checkpoint file not found: ${CKPT_PATH}"
        exit 1
    fi
    EXP_NAME="$(basename "$(dirname "${CKPT_PATH}")")"
    OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}/results/by-path/${EXP_NAME}"
fi

RESULTS_DIR="${OUTPUT_BASE}/${SPLIT}"
mkdir -p "${RESULTS_DIR}"

CONDA_INIT="${CONDA_INIT:-${SATNAV_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}}"
CONDA_ENV="${CONDA_ENV:-${SATNAV_CONDA_ENV:-satnav}}"
if [ -f "${CONDA_INIT}" ]; then
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV}"
else
    print_warning "Conda init not found at ${CONDA_INIT}. Assuming satnav env is already active."
fi

cd "${REPO_ROOT}"

DATA_PATH=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG_PATH}')
print(cfg.DATASET.DATA_PATH.replace('{split}', '${SPLIT}'))
" 2>/dev/null)

TOTAL_EPISODES=$(python -c "
import json
with open('${DATA_PATH}') as f:
    data = json.load(f)
episodes = data.get('episodes', data) if isinstance(data, dict) else data
print(len(episodes))
" 2>/dev/null)

if [ -n "${MAX_EPISODES}" ]; then
    if ! [[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]]; then
        print_error "max_episodes must be a non-negative integer, got: ${MAX_EPISODES}"
        exit 1
    fi
    if [ "${MAX_EPISODES}" -lt "${TOTAL_EPISODES}" ]; then
        TOTAL_EPISODES="${MAX_EPISODES}"
    fi
fi

if [ "${TOTAL_EPISODES}" -le 0 ]; then
    print_error "No episodes found for split ${SPLIT} using config ${CONFIG_PATH}"
    exit 1
fi

print_info "Resolved checkpoint: ${CKPT_PATH}"
print_info "Resolved dataset: ${DATA_PATH}"
print_info "Total episodes to evaluate: ${TOTAL_EPISODES}"
print_info "Sharding across ${NUM_GPUS} GPUs: ${GPU_LIST}"

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${NUM_GPUS}" ]; then
    print_error "GPU list has ${#GPUS[@]} entries but NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

BASE=$(( TOTAL_EPISODES / NUM_GPUS ))
REMAINDER=$(( TOTAL_EPISODES % NUM_GPUS ))

declare -a OFFSETS COUNTS PIDS SHARD_LOGS
OFFSET=0
for (( i=0; i<NUM_GPUS; i++ )); do
    COUNT="${BASE}"
    if [ "${i}" -lt "${REMAINDER}" ]; then
        COUNT=$(( COUNT + 1 ))
    fi
    OFFSETS[$i]="${OFFSET}"
    COUNTS[$i]="${COUNT}"
    OFFSET=$(( OFFSET + COUNT ))
done

cleanup_children() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup_children INT TERM

print_info "Launching ${NUM_GPUS} parallel eval workers..."
for (( i=0; i<NUM_GPUS; i++ )); do
    GPU="${GPUS[$i]}"
    OFF="${OFFSETS[$i]}"
    CNT="${COUNTS[$i]}"
    SHARD_RESULTS_DIR="${RESULTS_DIR}/shard_${i}"
    SHARD_LOG="${SHARD_RESULTS_DIR}/eval.log"

    mkdir -p "${SHARD_RESULTS_DIR}"
    SHARD_LOGS[$i]="${SHARD_LOG}"

    if [ "${CNT}" -le 0 ]; then
        print_warning "  Shard ${i}: GPU=${GPU}, assigned 0 episodes, skipping"
        PIDS[$i]=""
        continue
    fi

    print_info "  Shard ${i}: GPU=${GPU}, episodes [${OFF}, $((OFF + CNT)))"

    CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
        --exp-config "${CONFIG_PATH}" \
        --run-type eval \
        EVAL.SPLIT "${SPLIT}" \
        EVAL.CKPT_PATH "${CKPT_PATH}" \
        EVAL.EPISODE_OFFSET "${OFF}" \
        EVAL.EPISODE_COUNT "${CNT}" \
        EVAL.SAVE_RESULTS true \
        RESULTS_DIR "${SHARD_RESULTS_DIR}" \
        DATASET.SPLIT "${SPLIT}" \
        > "${SHARD_LOG}" 2>&1 &

    PIDS[$i]=$!
done

print_info "Waiting for all shard workers..."
FAILED=0
for (( i=0; i<NUM_GPUS; i++ )); do
    PID="${PIDS[$i]:-}"
    if [ -z "${PID}" ]; then
        continue
    fi
    if wait "${PID}"; then
        print_success "Shard ${i} (PID=${PID}) completed"
    else
        print_error "Shard ${i} (PID=${PID}) failed — check ${SHARD_LOGS[$i]}"
        FAILED=$(( FAILED + 1 ))
    fi
done
trap - INT TERM

if [ "${FAILED}" -gt 0 ]; then
    print_error "${FAILED} shard(s) failed. Merge aborted."
    exit 1
fi

print_info "Merging shard results..."
RESULTS_DIR="${RESULTS_DIR}" SPLIT="${SPLIT}" NUM_GPUS="${NUM_GPUS}" python - <<'PYEOF'
import json
import os
import sys

results_dir = os.environ["RESULTS_DIR"]
split = os.environ["SPLIT"]
num_gpus = int(os.environ["NUM_GPUS"])

all_episodes = []
total_action_counts = {}
total_failure_modes = {}

for i in range(num_gpus):
    shard_file = os.path.join(
        results_dir,
        f"shard_{i}",
        f"eval_ckpt_0_{split}_diagnostics.json",
    )
    if not os.path.exists(shard_file):
        continue
    with open(shard_file) as f:
        data = json.load(f)
    all_episodes.extend(data.get("episodes", []))
    for key, value in data.get("global_action_counts", {}).items():
        total_action_counts[key] = total_action_counts.get(key, 0) + value
    for key, value in data.get("failure_mode_counts", {}).items():
        total_failure_modes[key] = total_failure_modes.get(key, 0) + value

n = len(all_episodes)
if n == 0:
    print("[ERROR] No episode diagnostics found to merge.")
    sys.exit(1)

merged = {
    "spl": sum(ep["spl"] for ep in all_episodes) / n,
    "success": sum(ep["success"] for ep in all_episodes) / n,
    "distance_to_goal": sum(ep["final_distance_to_goal"] for ep in all_episodes) / n,
    "path_length": sum(ep["path_length"] for ep in all_episodes) / n,
    "steps_taken": sum(ep["steps_taken"] for ep in all_episodes) / n,
    "num_episodes": n,
    "split": split,
    "checkpoint_index": 0,
}

total_steps = sum(total_action_counts.values())
action_dist = (
    {key: value / total_steps for key, value in total_action_counts.items()}
    if total_steps
    else {}
)

merged_diag = {
    "checkpoint_index": 0,
    "split": split,
    "num_episodes": n,
    "global_action_counts": total_action_counts,
    "global_action_distribution": action_dist,
    "failure_mode_counts": total_failure_modes,
    "episodes": all_episodes,
}

out_json = os.path.join(results_dir, f"eval_ckpt_0_{split}.json")
out_diag = os.path.join(results_dir, f"eval_ckpt_0_{split}_diagnostics.json")

with open(out_json, "w") as f:
    json.dump(merged, f, indent=4)
with open(out_diag, "w") as f:
    json.dump(merged_diag, f, indent=4)

print(f'{"=" * 60}')
print(f"Merged eval results ({n} episodes):")
print(f"  SR  (Success Rate): {merged['success']:.4f}  ({merged['success'] * 100:.2f}%)")
print(f"  SPL:                {merged['spl']:.4f}")
print(f"  NE  (dist to goal): {merged['distance_to_goal']:.1f} m")
print(f"  Path Length:        {merged['path_length']:.1f} m")
print(f"  Mean Steps:         {merged['steps_taken']:.1f}")
print(f"  Failure modes:      {total_failure_modes}")
print(f'{"=" * 60}')
print(f"Results: {out_json}")
PYEOF

print_success "Parallel evaluation completed"
print_success "Merged results: ${RESULTS_DIR}/eval_ckpt_0_${SPLIT}.json"

#!/usr/bin/env bash
# ==============================================================================
# Parallel CMA Evaluation — splits val_seen episodes across N GPUs.
#
# Usage:
#   bash scripts/cma/eval_parallel.sh <exp_name> [split] [num_gpus] [gpu_list]
#
# Examples:
#   bash scripts/cma/eval_parallel.sh cma-stop-fix val_seen 8 0,1,2,3,4,5,6,7
#   bash scripts/cma/eval_parallel.sh cma-stop-fix val_seen 4 0,1,2,3
#
# Output:
#   output/cma/results/<exp_name>/<split>/eval_ckpt_0_<split>.json  (merged)
#   output/cma/results/<exp_name>/<split>/shard_<i>/               (per-shard)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Args ----
EXP_NAME="${1:-}"
SPLIT="${2:-val_seen}"
NUM_GPUS="${3:-8}"
GPU_LIST="${4:-0,1,2,3,4,5,6,7}"

if [ -z "$EXP_NAME" ]; then
    print_error "Usage: bash scripts/cma/eval_parallel.sh <exp_name> [split] [num_gpus] [gpu_list]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/cma_eval.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/cma}"

CKPT_PATH="${REPO_ROOT}/${OUTPUT_ROOT}/checkpoints/${EXP_NAME}/best.pth"
if [ ! -f "$CKPT_PATH" ]; then
    print_error "Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

RESULTS_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/results/${EXP_NAME}/${SPLIT}"
mkdir -p "${RESULTS_DIR}"

# ---- Conda ----
CONDA_INIT="/mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh"
[ -f "${CONDA_INIT}" ] && source "${CONDA_INIT}" && conda activate satnav

cd "${REPO_ROOT}"

# ---- Count total episodes ----
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

print_info "Total episodes: ${TOTAL_EPISODES}, sharding across ${NUM_GPUS} GPUs"

# ---- Split GPUs into array ----
IFS=',' read -ra GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -lt "$NUM_GPUS" ]; then
    print_error "GPU list has ${#GPUS[@]} entries but NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

# ---- Compute shard offsets ----
BASE=$(( TOTAL_EPISODES / NUM_GPUS ))
REMAINDER=$(( TOTAL_EPISODES % NUM_GPUS ))

declare -a OFFSETS COUNTS
OFFSET=0
for (( i=0; i<NUM_GPUS; i++ )); do
    COUNT=$BASE
    [ $i -lt $REMAINDER ] && COUNT=$(( COUNT + 1 ))
    OFFSETS[$i]=$OFFSET
    COUNTS[$i]=$COUNT
    OFFSET=$(( OFFSET + COUNT ))
done

# ---- Launch parallel shards ----
print_info "Launching ${NUM_GPUS} parallel eval workers..."
PIDS=()
SHARD_LOGS=()

for (( i=0; i<NUM_GPUS; i++ )); do
    GPU="${GPUS[$i]}"
    OFF="${OFFSETS[$i]}"
    CNT="${COUNTS[$i]}"
    SHARD_RESULTS_DIR="${RESULTS_DIR}/shard_${i}"
    mkdir -p "${SHARD_RESULTS_DIR}"
    SHARD_LOG="${SHARD_RESULTS_DIR}/eval.log"
    SHARD_LOGS+=("$SHARD_LOG")

    print_info "  Shard ${i}: GPU=${GPU}, episodes [${OFF}, $((OFF+CNT)))"

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

    PIDS+=($!)
done

# ---- Wait for all shards ----
print_info "Waiting for all ${NUM_GPUS} shards to complete..."
FAILED=0
for (( i=0; i<NUM_GPUS; i++ )); do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        print_success "Shard ${i} (PID=${PID}) completed"
    else
        print_error "Shard ${i} (PID=${PID}) FAILED — check ${SHARD_LOGS[$i]}"
        FAILED=$(( FAILED + 1 ))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    print_error "${FAILED} shard(s) failed. Merge aborted."
    exit 1
fi

# ---- Merge shard results ----
print_info "Merging ${NUM_GPUS} shards..."

RESULTS_DIR="$RESULTS_DIR" SPLIT="$SPLIT" NUM_GPUS="$NUM_GPUS" python - <<'PYEOF'
import json, os, sys

results_dir = os.environ['RESULTS_DIR']
split       = os.environ['SPLIT']
num_gpus    = int(os.environ['NUM_GPUS'])

all_episodes        = []
total_action_counts = {}
total_failure_modes = {}

for i in range(num_gpus):
    shard_file = os.path.join(results_dir, f'shard_{i}',
                              f'eval_ckpt_0_{split}_diagnostics.json')
    if not os.path.exists(shard_file):
        print(f'[WARN] Missing shard diagnostics: {shard_file}')
        continue
    with open(shard_file) as f:
        data = json.load(f)
    all_episodes.extend(data.get('episodes', []))
    for k, v in data.get('global_action_counts', {}).items():
        total_action_counts[k] = total_action_counts.get(k, 0) + v
    for k, v in data.get('failure_mode_counts', {}).items():
        total_failure_modes[k] = total_failure_modes.get(k, 0) + v

n = len(all_episodes)
if n == 0:
    print('[ERROR] No episodes merged'); sys.exit(1)

spl     = sum(ep['spl']                    for ep in all_episodes) / n
success = sum(ep['success']                for ep in all_episodes) / n
ne      = sum(ep['final_distance_to_goal'] for ep in all_episodes) / n
pl      = sum(ep['path_length']            for ep in all_episodes) / n
steps   = sum(ep['steps_taken']            for ep in all_episodes) / n

total_steps = sum(total_action_counts.values())
action_dist = {k: v / total_steps for k, v in total_action_counts.items()} if total_steps else {}

merged = {
    'spl': spl, 'success': success, 'distance_to_goal': ne,
    'path_length': pl, 'steps_taken': steps,
    'num_episodes': n, 'split': split, 'checkpoint_index': 0,
}
merged_diag = {
    'checkpoint_index': 0, 'split': split, 'num_episodes': n,
    'global_action_counts': total_action_counts,
    'global_action_distribution': action_dist,
    'failure_mode_counts': total_failure_modes,
    'episodes': all_episodes,
}

out_json = os.path.join(results_dir, f'eval_ckpt_0_{split}.json')
out_diag = os.path.join(results_dir, f'eval_ckpt_0_{split}_diagnostics.json')
with open(out_json, 'w') as f:
    json.dump(merged, f, indent=4)
with open(out_diag, 'w') as f:
    json.dump(merged_diag, f, indent=4)

print(f'\n{"="*60}')
print(f'Merged eval results ({n} episodes):')
print(f'  SR  (Success Rate): {success:.4f}  ({success*100:.2f}%)')
print(f'  SPL:                {spl:.4f}')
print(f'  NE  (dist to goal): {ne:.1f} m')
print(f'  Path Length:        {pl:.1f} m')
print(f'  Mean Steps:         {steps:.1f}')
print(f'  Failure modes:      {total_failure_modes}')
print(f'{"="*60}')
print(f'Results: {out_json}')
PYEOF

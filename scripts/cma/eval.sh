#!/usr/bin/env bash
# ==============================================================================
# Evaluate CMA Baseline on SatNav task.
#
# Supports two calling modes:
#
#   1. Eval by exp name (recommended):
#      bash scripts/cma/eval.sh <exp_name> [split] [max_episodes]
#      - Looks for checkpoint in output/cma/checkpoints/<exp_name>/best.pth
#
#   2. Eval by checkpoint path (direct):
#      bash scripts/cma/eval.sh /path/to/best.pth [split] [max_episodes]
#      - Uses the provided .pth file directly
#
# Arguments:
#   exp_name / ckpt_path  First argument: experiment name or full .pth path
#   split                 Evaluation split: val_seen / val_unseen (default: val_seen)
#   max_episodes          Limit episodes for quick debugging (optional, default: all)
#
# Environment variables:
#   CONFIG_PATH           Override eval config (default: configs/baselines/cma_eval.yaml)
#   OUTPUT_ROOT           Override output root (default: output/cma)
#   CUDA_DEVICES          GPU to use (default: 0)
#
# Output:
#   output/cma/results/<exp_name>/<split>/eval_ckpt_0_<split>.json
#
# Environment: conda env satnav
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Args ----
INPUT="${1:-}"
SPLIT="${2:-val_seen}"
MAX_EPISODES="${3:-}"

if [ -z "$INPUT" ]; then
    print_error "Usage: bash scripts/cma/eval.sh <exp_name | ckpt_path> [split] [max_episodes]"
    echo ""
    echo "Examples:"
    echo "  # Eval by exp name (recommended)"
    echo "  bash scripts/cma/eval.sh cma-ddp-g8-bs8-lr1e-4-20260320-164653"
    echo ""
    echo "  # Eval specific split"
    echo "  bash scripts/cma/eval.sh cma-ddp-g8-bs8-lr1e-4-20260320-164653 val_seen"
    echo ""
    echo "  # Quick debug with limited episodes"
    echo "  bash scripts/cma/eval.sh cma-ddp-g8-bs8-lr1e-4-20260320-164653 val_seen 20"
    echo ""
    echo "  # Eval by checkpoint path"
    echo "  bash scripts/cma/eval.sh /path/to/best.pth val_seen"
    exit 1
fi

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-configs/baselines/cma_eval.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/cma}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

# ---- Detect mode: eval-by-name vs eval-by-path ----
EVAL_MODE="by_name"
if [[ "$INPUT" = /* ]] || [[ "$INPUT" = *.pth ]]; then
    EVAL_MODE="by_path"
fi

# ---- Resolve checkpoint and output paths ----
if [ "$EVAL_MODE" = "by_name" ]; then
    EXP_NAME="$INPUT"
    CKPT_PATH="${REPO_ROOT}/${OUTPUT_ROOT}/checkpoints/${EXP_NAME}/best.pth"
    if [ ! -f "$CKPT_PATH" ]; then
        print_error "Checkpoint not found: ${CKPT_PATH}"
        print_error "Make sure training has completed and best.pth exists."
        exit 1
    fi
    OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}/results/${EXP_NAME}"
else
    CKPT_PATH="$INPUT"
    if [ ! -f "$CKPT_PATH" ]; then
        print_error "Checkpoint file not found: ${CKPT_PATH}"
        exit 1
    fi
    EXP_NAME="$(basename "$(dirname "${CKPT_PATH}")")"
    OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}/results/by-path/${EXP_NAME}"
fi

RESULTS_DIR="${OUTPUT_BASE}/${SPLIT}"
mkdir -p "${RESULTS_DIR}"

# ---- Conda activation ----
CONDA_INIT="/mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh"
if [ -f "${CONDA_INIT}" ]; then
    source "${CONDA_INIT}"
    conda activate satnav
else
    print_warning "Conda init not found at ${CONDA_INIT}. Assuming satnav env is already active."
fi

# ---- Print summary ----
echo ""
echo "=========================================="
echo "CMA SatNav Evaluation"
echo "=========================================="
echo "  Eval mode   : ${EVAL_MODE}"
echo "  Exp name    : ${EXP_NAME}"
echo "  Checkpoint  : ${CKPT_PATH}"
echo "  Config      : ${CONFIG_PATH}"
echo "  Split       : ${SPLIT}"
echo "  Results dir : ${RESULTS_DIR}"
echo "  CUDA devices: ${CUDA_DEVICES}"
[ -n "$MAX_EPISODES" ] && echo "  Max episodes: ${MAX_EPISODES}"
echo "=========================================="

# ---- Build CLI overrides ----
OVERRIDES=(
    EVAL.SPLIT "${SPLIT}"
    EVAL.CKPT_PATH "${CKPT_PATH}"
    RESULTS_DIR "${RESULTS_DIR}"
    DATASET.SPLIT "${SPLIT}"
)

if [ -n "$MAX_EPISODES" ]; then
    OVERRIDES+=(EVAL.EPISODE_COUNT "${MAX_EPISODES}")
fi

# ---- Run evaluation ----
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

python run.py \
    --exp-config "${CONFIG_PATH}" \
    --run-type eval \
    "${OVERRIDES[@]}" \
    2>&1 | tee "${RESULTS_DIR}/eval.log"

print_success "Evaluation completed!"
echo "  Results: ${RESULTS_DIR}"

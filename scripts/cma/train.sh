#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-configs/baselines/cma_offline_train.yaml}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29600}"

exec "${SCRIPT_DIR}/train_ddp.sh" "$@"

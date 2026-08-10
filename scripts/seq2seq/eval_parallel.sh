#!/usr/bin/env bash
# Backward-compatible arguments; sharding/merge use satnav.evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../classic/eval_checkpoint_parallel.sh" seq2seq "$@"

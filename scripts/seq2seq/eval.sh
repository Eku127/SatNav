#!/usr/bin/env bash
# Backward-compatible arguments; execution uses the generic classic evaluator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../classic/eval_checkpoint.sh" seq2seq "$@"

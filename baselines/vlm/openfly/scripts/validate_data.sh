#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
openfly_prepare_python

TRAJECTORY_ROOT="${1:-${SATNAV_OPENFLY_TRAIN_DATA:-}}"
REPORT="${2:-}"
[[ -n "${TRAJECTORY_ROOT}" ]] || {
    echo "Usage: $0 TRAJECTORY_ROOT [REPORT]" >&2
    exit 2
}
ARGS=("${TRAJECTORY_ROOT}" --strict-frames --hash-frame-content)
[[ -z "${REPORT}" ]] || ARGS+=(--report "${REPORT}")
cd "${SATNAV_ROOT}"
"${OPENFLY_PYTHON_BIN}" -m baselines.vlm.openfly.dataset "${ARGS[@]}"

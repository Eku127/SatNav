#!/usr/bin/env bash

STREAMVLN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMVLN_BASELINE_DIR="$(cd "${STREAMVLN_SCRIPT_DIR}/.." && pwd)"
SATNAV_ROOT="$(cd "${STREAMVLN_BASELINE_DIR}/../../.." && pwd)"
export SATNAV_ROOT STREAMVLN_BASELINE_DIR

# shellcheck source=../../../../scripts/lib/local_env.sh
source "${SATNAV_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${STREAMVLN_BASELINE_DIR}"

STREAMVLN_REPO="${STREAMVLN_REPO:-${SATNAV_ROOT}/../StreamVLN}"
STREAMVLN_MODEL_ROOT="${STREAMVLN_MODEL_ROOT:-${STREAMVLN_BASELINE_DIR}/model}"
SATNAV_STREAMVLN_OUTPUT="${SATNAV_STREAMVLN_OUTPUT:-output/baselines/vlm/streamvln}"
export STREAMVLN_REPO STREAMVLN_MODEL_ROOT SATNAV_STREAMVLN_OUTPUT

streamvln_python() {
    if [[ -n "${STREAMVLN_PYTHON:-}" ]]; then
        if [[ -x "${STREAMVLN_PYTHON}" ]]; then
            printf '%s\n' "${STREAMVLN_PYTHON}"
            return 0
        fi
        echo "[ERROR] STREAMVLN_PYTHON is not executable: ${STREAMVLN_PYTHON}" >&2
        return 1
    fi
    command -v python3 || command -v python || {
        echo "[ERROR] Python not found; set STREAMVLN_PYTHON." >&2
        return 1
    }
}

streamvln_prepare_python() {
    STREAMVLN_PYTHON_BIN="$(streamvln_python)" || return 1
    STREAMVLN_PYTHON_BIN_DIR="$(
        cd "$(dirname "${STREAMVLN_PYTHON_BIN}")" && pwd
    )" || return 1
    export STREAMVLN_PYTHON_BIN STREAMVLN_PYTHON_BIN_DIR
    export PATH="${STREAMVLN_PYTHON_BIN_DIR}:${PATH}"
}

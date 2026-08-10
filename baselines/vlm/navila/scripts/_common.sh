#!/usr/bin/env bash

NAVILA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVILA_BASELINE_DIR="$(cd "${NAVILA_SCRIPT_DIR}/.." && pwd)"
SATNAV_ROOT="$(cd "${NAVILA_BASELINE_DIR}/../../.." && pwd)"
export NAVILA_BASELINE_DIR SATNAV_ROOT

# shellcheck source=../../../../scripts/lib/local_env.sh
source "${SATNAV_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${NAVILA_BASELINE_DIR}"

NAVILA_REPO="${NAVILA_REPO:-${SATNAV_ROOT}/../NaVILA}"
NAVILA_MODEL_ROOT="${NAVILA_MODEL_ROOT:-${NAVILA_BASELINE_DIR}/model}"
SATNAV_NAVILA_OUTPUT="${SATNAV_NAVILA_OUTPUT:-output/baselines/vlm/navila}"
export NAVILA_REPO NAVILA_MODEL_ROOT SATNAV_NAVILA_OUTPUT
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

navila_prepare_python() {
    if [[ -n "${NAVILA_PYTHON:-}" ]]; then
        [[ -x "${NAVILA_PYTHON}" ]] || {
            echo "[ERROR] NAVILA_PYTHON is not executable: ${NAVILA_PYTHON}" >&2
            return 1
        }
        NAVILA_PYTHON_BIN="${NAVILA_PYTHON}"
    else
        NAVILA_PYTHON_BIN="$(command -v python3 || command -v python || true)"
        [[ -n "${NAVILA_PYTHON_BIN}" ]] || {
            echo "[ERROR] Python not found; set NAVILA_PYTHON." >&2
            return 1
        }
    fi
    NAVILA_PYTHON_BIN_DIR="$(cd "$(dirname "${NAVILA_PYTHON_BIN}")" && pwd)"
    export NAVILA_PYTHON_BIN NAVILA_PYTHON_BIN_DIR
    export PATH="${NAVILA_PYTHON_BIN_DIR}:${PATH}"
}

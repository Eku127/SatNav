#!/usr/bin/env bash

UNINAVID_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNINAVID_BASELINE_DIR="$(cd "${UNINAVID_SCRIPT_DIR}/.." && pwd)"
SATNAV_ROOT="$(cd "${UNINAVID_BASELINE_DIR}/../../.." && pwd)"
export UNINAVID_BASELINE_DIR SATNAV_ROOT

# shellcheck source=../../../../scripts/lib/local_env.sh
source "${SATNAV_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${UNINAVID_BASELINE_DIR}"

UNINAVID_REPO="${UNINAVID_REPO:-${SATNAV_ROOT}/../Uni-NaVid}"
UNINAVID_MODEL_ROOT="${UNINAVID_MODEL_ROOT:-${UNINAVID_BASELINE_DIR}/model}"
UNINAVID_PROCESSOR="${UNINAVID_PROCESSOR:-${UNINAVID_REPO}/uninavid/processor/clip-patch14-224}"
SATNAV_UNINAVID_OUTPUT="${SATNAV_UNINAVID_OUTPUT:-output/baselines/vlm/uninavid}"
export UNINAVID_REPO UNINAVID_MODEL_ROOT UNINAVID_PROCESSOR
export SATNAV_UNINAVID_OUTPUT
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

uninavid_prepare_python() {
    if [[ -n "${UNINAVID_PYTHON:-}" ]]; then
        [[ -x "${UNINAVID_PYTHON}" ]] || {
            echo "[ERROR] UNINAVID_PYTHON is not executable: ${UNINAVID_PYTHON}" >&2
            return 1
        }
        UNINAVID_PYTHON_BIN="${UNINAVID_PYTHON}"
    else
        UNINAVID_PYTHON_BIN="$(command -v python3 || command -v python || true)"
        [[ -n "${UNINAVID_PYTHON_BIN}" ]] || {
            echo "[ERROR] Python not found; set UNINAVID_PYTHON." >&2
            return 1
        }
    fi
    UNINAVID_PYTHON_BIN_DIR="$(cd "$(dirname "${UNINAVID_PYTHON_BIN}")" && pwd)"
    export UNINAVID_PYTHON_BIN UNINAVID_PYTHON_BIN_DIR
    export PATH="${UNINAVID_PYTHON_BIN_DIR}:${PATH}"
}

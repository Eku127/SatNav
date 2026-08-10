#!/usr/bin/env bash

OPENFLY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENFLY_BASELINE_DIR="$(cd "${OPENFLY_SCRIPT_DIR}/.." && pwd)"
SATNAV_ROOT="$(cd "${OPENFLY_BASELINE_DIR}/../../.." && pwd)"
export OPENFLY_BASELINE_DIR SATNAV_ROOT

# shellcheck source=../../../../scripts/lib/local_env.sh
source "${SATNAV_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env "${OPENFLY_BASELINE_DIR}"

OPENFLY_MODEL_ROOT="${OPENFLY_MODEL_ROOT:-${OPENFLY_BASELINE_DIR}/model}"
SATNAV_OPENFLY_OUTPUT="${SATNAV_OPENFLY_OUTPUT:-${SATNAV_ROOT}/output/baselines/vlm/openfly}"
export OPENFLY_MODEL_ROOT SATNAV_OPENFLY_OUTPUT
# OpenFly owns a fully isolated dependency stack.  Never let an inherited
# PYTHONNOUSERSITE=0 make user-level packages shadow the frozen environment.
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

openfly_prepare_python() {
    if [[ -n "${OPENFLY_PYTHON:-}" ]]; then
        [[ -x "${OPENFLY_PYTHON}" ]] || {
            echo "[ERROR] OPENFLY_PYTHON is not executable: ${OPENFLY_PYTHON}" >&2
            return 1
        }
        OPENFLY_PYTHON_BIN="${OPENFLY_PYTHON}"
    else
        OPENFLY_PYTHON_BIN="$(command -v python3 || command -v python || true)"
        [[ -n "${OPENFLY_PYTHON_BIN}" ]] || {
            echo "[ERROR] Python not found; set OPENFLY_PYTHON." >&2
            return 1
        }
    fi
    OPENFLY_PYTHON_BIN_DIR="$(cd "$(dirname "${OPENFLY_PYTHON_BIN}")" && pwd)"
    export OPENFLY_PYTHON_BIN OPENFLY_PYTHON_BIN_DIR
    export PATH="${OPENFLY_PYTHON_BIN_DIR}:${PATH}"
}

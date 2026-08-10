#!/usr/bin/env bash
# Apply the pinned NaVILA Transformers/DeepSpeed runtime replacements.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
navila_prepare_python

SITE_PACKAGES="$("${NAVILA_PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
for package in transformers deepspeed; do
    SOURCE="${NAVILA_REPO}/llava/train/${package}_replace"
    DESTINATION="${SITE_PACKAGES}/${package}"
    [[ -d "${SOURCE}" ]] || { echo "[ERROR] Missing pinned patch source: ${SOURCE}" >&2; exit 1; }
    [[ -d "${DESTINATION}" ]] || { echo "[ERROR] Missing installed package: ${DESTINATION}" >&2; exit 1; }
    cp -rv "${SOURCE}/." "${DESTINATION}/"
done
echo "[OK] Applied pinned NaVILA runtime replacements to ${SITE_PACKAGES}"

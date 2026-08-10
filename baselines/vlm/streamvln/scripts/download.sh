#!/usr/bin/env bash
# Download the official StreamVLN checkpoint or another model dependency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/streamvln/scripts/download.sh [options]

Options:
  --repo OWNER/NAME        Repository (default: official StreamVLN checkpoint)
  --name DIRECTORY         Destination name (default: repository basename)
  --model-root DIRECTORY   Parent destination (default: STREAMVLN_MODEL_ROOT)
  --source hf|modelscope   Downloader backend (default: hf)
  --endpoint URL           Optional Hugging Face endpoint/mirror
  --dry-run                Print the resolved download without writing
  -h, --help               Show this help
EOF
}

REPO="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
NAME=""
MODEL_ROOT="${STREAMVLN_MODEL_ROOT}"
SOURCE="hf"
ENDPOINT="${HF_ENDPOINT:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="${2:?--repo requires a value}"; shift 2 ;;
        --name) NAME="${2:?--name requires a value}"; shift 2 ;;
        --model-root) MODEL_ROOT="${2:?--model-root requires a value}"; shift 2 ;;
        --source) SOURCE="${2:?--source requires a value}"; shift 2 ;;
        --endpoint) ENDPOINT="${2:?--endpoint requires a value}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${NAME}" ]] || NAME="${REPO##*/}"
DESTINATION="${MODEL_ROOT%/}/${NAME}"
if [[ "${SOURCE}" != hf && "${SOURCE}" != modelscope ]]; then
    echo "[ERROR] --source must be hf or modelscope" >&2
    exit 2
fi
echo "[INFO] source=${SOURCE} repo=${REPO} destination=${DESTINATION}"
if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi
mkdir -p "${DESTINATION}"
PYTHON_BIN="$(streamvln_python)"

if [[ "${SOURCE}" == hf ]]; then
    [[ -z "${ENDPOINT}" ]] || export HF_ENDPOINT="${ENDPOINT}"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${REPO}" --local-dir "${DESTINATION}"
    else
        "${PYTHON_BIN}" -m huggingface_hub.commands.huggingface_cli \
            download "${REPO}" --local-dir "${DESTINATION}"
    fi
else
    "${PYTHON_BIN}" - "${REPO}" "${DESTINATION}" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi
echo "[OK] Download complete: ${DESTINATION}"

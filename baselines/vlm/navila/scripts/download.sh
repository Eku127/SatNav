#!/usr/bin/env bash
# Download an official NaVILA model snapshot without embedding credentials.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/navila/scripts/download.sh [options]

Options:
  --repo OWNER/NAME        Default: a8cheng/navila-llama3-8b-8f
  --name DIRECTORY         Destination name (default: repository basename)
  --model-root DIRECTORY   Parent destination
  --endpoint URL           Optional Hugging Face endpoint/mirror
  --dry-run
  -h, --help
EOF
}

REPO=a8cheng/navila-llama3-8b-8f
NAME=""
MODEL_ROOT="${NAVILA_MODEL_ROOT}"
ENDPOINT="${HF_ENDPOINT:-}"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="${2:?--repo requires a value}"; shift 2 ;;
        --name) NAME="${2:?--name requires a value}"; shift 2 ;;
        --model-root) MODEL_ROOT="${2:?--model-root requires a path}"; shift 2 ;;
        --endpoint) ENDPOINT="${2:?--endpoint requires a value}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "${NAME}" ]] || NAME="${REPO##*/}"
DESTINATION="${MODEL_ROOT%/}/${NAME}"
echo "[INFO] repo=${REPO} destination=${DESTINATION}"
[[ "${DRY_RUN}" == false ]] || exit 0
mkdir -p "${DESTINATION}"
navila_prepare_python
[[ -z "${ENDPOINT}" ]] || export HF_ENDPOINT="${ENDPOINT}"
"${NAVILA_PYTHON_BIN}" -m huggingface_hub.commands.huggingface_cli \
    download "${REPO}" --local-dir "${DESTINATION}"

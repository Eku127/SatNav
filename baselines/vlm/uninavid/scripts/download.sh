#!/usr/bin/env bash
# Fetch pinned source, a revision-resolved model snapshot, and verified EVA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

SOURCE_REVISION=79ef5ea3fea14c205342d1ab070563d84c7a966a
EVA_URL=https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
EVA_SHA256=99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f
MODEL_REPO=Jzzhang/Uni-NaVid
MODEL_REVISION=main
MODEL_SUBFOLDER=uninavid-7b-full-224-video-fps-1-grid-2
SOURCE_DIR="${UNINAVID_REPO}"
MODEL_ROOT="${UNINAVID_MODEL_ROOT}"
DOWNLOAD_SOURCE=true
DOWNLOAD_MODEL=true
DOWNLOAD_EVA=true
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: bash baselines/vlm/uninavid/scripts/download.sh [options]

Options:
  --source-dir PATH        Clean external Uni-NaVid checkout
  --model-root PATH        Parent directory for downloaded artifacts
  --model-repo OWNER/NAME  Default: Jzzhang/Uni-NaVid
  --model-revision REV     Tag/branch/commit resolved to an immutable commit
  --model-subfolder PATH   Checkpoint subfolder within the model repository
  --skip-source --skip-model --skip-eva
  --dry-run
  -h, --help

The source checkout is detached at the maintained commit. The resolved model
revision and EVA SHA-256 are printed; EVA is accepted only at the pinned hash.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir) SOURCE_DIR="${2:?--source-dir requires a path}"; shift 2 ;;
        --model-root) MODEL_ROOT="${2:?--model-root requires a path}"; shift 2 ;;
        --model-repo) MODEL_REPO="${2:?--model-repo requires a value}"; shift 2 ;;
        --model-revision) MODEL_REVISION="${2:?--model-revision requires a value}"; shift 2 ;;
        --model-subfolder) MODEL_SUBFOLDER="${2:?--model-subfolder requires a value}"; shift 2 ;;
        --skip-source) DOWNLOAD_SOURCE=false; shift ;;
        --skip-model) DOWNLOAD_MODEL=false; shift ;;
        --skip-eva) DOWNLOAD_EVA=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

MODEL_DEST="${MODEL_ROOT%/}/huggingface/$(basename "${MODEL_REPO}")"
EVA_DEST="${MODEL_ROOT%/}/eva_vit_g.pth"
echo "[INFO] source=${SOURCE_DIR} revision=${SOURCE_REVISION}"
echo "[INFO] model=${MODEL_REPO}@${MODEL_REVISION} destination=${MODEL_DEST}/${MODEL_SUBFOLDER}"
echo "[INFO] eva=${EVA_DEST} sha256=${EVA_SHA256}"
[[ "${DRY_RUN}" == false ]] || exit 0

if [[ "${DOWNLOAD_SOURCE}" == true ]]; then
    if [[ ! -e "${SOURCE_DIR}" ]]; then
        git clone https://github.com/jzhzhang/Uni-NaVid.git "${SOURCE_DIR}"
        git -C "${SOURCE_DIR}" checkout --detach "${SOURCE_REVISION}"
    fi
    [[ -d "${SOURCE_DIR}/.git" ]] || { echo "[ERROR] Not a Git checkout: ${SOURCE_DIR}" >&2; exit 1; }
    [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${SOURCE_REVISION}" ]] || {
        echo "[ERROR] Existing source is not at ${SOURCE_REVISION}: ${SOURCE_DIR}" >&2
        exit 1
    }
    [[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)" ]] || {
        echo "[ERROR] Existing source checkout is dirty: ${SOURCE_DIR}" >&2
        exit 1
    }
fi

mkdir -p "${MODEL_ROOT}"
if [[ "${DOWNLOAD_EVA}" == true ]]; then
    if [[ -f "${EVA_DEST}" ]] && echo "${EVA_SHA256}  ${EVA_DEST}" | sha256sum --check --status; then
        echo "[INFO] verified existing EVA: ${EVA_DEST}"
    else
        PARTIAL="${EVA_DEST}.partial"
        curl --fail --location --continue-at - --output "${PARTIAL}" "${EVA_URL}"
        echo "${EVA_SHA256}  ${PARTIAL}" | sha256sum --check --status || {
            echo "[ERROR] EVA SHA-256 mismatch; partial retained at ${PARTIAL}" >&2
            exit 1
        }
        mv "${PARTIAL}" "${EVA_DEST}"
        echo "[INFO] verified EVA: ${EVA_DEST}"
    fi
fi

if [[ "${DOWNLOAD_MODEL}" == true ]]; then
    uninavid_prepare_python
    mkdir -p "${MODEL_DEST}"
    "${UNINAVID_PYTHON_BIN}" - "${MODEL_REPO}" "${MODEL_REVISION}" "${MODEL_SUBFOLDER}" "${MODEL_DEST}" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, requested, subfolder, destination = sys.argv[1:]
resolved = HfApi().model_info(repo, revision=requested).sha
snapshot_download(
    repo_id=repo,
    revision=resolved,
    allow_patterns=[subfolder.rstrip("/") + "/*"],
    local_dir=destination,
    local_dir_use_symlinks=False,
)
record = {
    "model_repo": repo,
    "requested_revision": requested,
    "resolved_revision": resolved,
    "subfolder": subfolder,
}
path = Path(destination) / ".satnav_download.json"
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2, sort_keys=True))
PY
fi

#!/usr/bin/env bash
# Explicitly create a fresh, isolated OpenFly environment. Never run from train/eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

ENV_NAME="${1:-satnav-openfly}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "${CONDA_BIN}" ]] || { echo "[ERROR] conda is required." >&2; exit 2; }

"${CONDA_BIN}" env create --name "${ENV_NAME}" --file "${OPENFLY_BASELINE_DIR}/environment/conda.yml"
"${CONDA_BIN}" run --name "${ENV_NAME}" python -m pip --isolated install \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121
"${CONDA_BIN}" run --name "${ENV_NAME}" python -m pip --isolated install \
    --index-url https://pypi.org/simple \
    --requirement "${OPENFLY_BASELINE_DIR}/requirements.txt"
"${CONDA_BIN}" run --name "${ENV_NAME}" python -m pip --isolated install \
    --editable "${SATNAV_ROOT}" --no-deps --no-build-isolation
"${CONDA_BIN}" run --name "${ENV_NAME}" python -m pip --isolated check
"${CONDA_BIN}" run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import sys
import site
from importlib.metadata import version

import flash_attn
import numpy
import torch
import transformers

assert sys.version_info[:3] == (3, 10, 14), sys.version
assert site.ENABLE_USER_SITE is False, site.ENABLE_USER_SITE
assert torch.__version__ == "2.3.0+cu121", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert transformers.__version__ == "4.48.1", transformers.__version__
assert flash_attn.__version__ == "2.5.8", flash_attn.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert version("torchvision") == "0.18.0+cu121", version("torchvision")
assert version("accelerate") == "0.33.0", version("accelerate")
assert version("deepspeed") == "0.14.4", version("deepspeed")
assert version("timm") == "0.9.16", version("timm")
assert version("tokenizers") == "0.21.1", version("tokenizers")
assert version("safetensors") == "0.4.5", version("safetensors")
print("fresh OpenFly environment verified")
PY

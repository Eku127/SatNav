#!/usr/bin/env bash
# End-to-end local quickstart for Seq2Seq and CMA baselines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/local_env.sh
source "${REPO_ROOT}/scripts/lib/local_env.sh"
satnav_load_local_env

cd "${REPO_ROOT}"

OUT_ROOT="${OUT_ROOT:-output/quickstart_baselines}"
EPISODES="${EPISODES:-applications/resources/satnav_example_episodes.json}"
TASK_CONFIG="${TASK_CONFIG:-applications/resources/satnav_example_task.yaml}"
RUN_SEQ2SEQ="${RUN_SEQ2SEQ:-1}"
RUN_CMA="${RUN_CMA:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

VOCAB_PATH="${OUT_ROOT}/artifacts/vocab/satnav_example_vocab.json"
GLOVE_DIR="${OUT_ROOT}/artifacts/glove"
GLOVE_ZIP="${GLOVE_DIR}/glove.6B.zip"
GLOVE_TXT="${GLOVE_DIR}/glove.6B.50d.txt"
LOCAL_GLOVE_TXT="${LOCAL_GLOVE_TXT:-${SATNAV_GLOVE_TXT:-}}"
EMBEDDING_PATH="${OUT_ROOT}/artifacts/embeddings/satnav_example_glove50d.json.gz"
TRAJECTORY_DIR="${OUT_ROOT}/trajectory_data"

echo "[1/7] Build vocabulary"
python -m satnav.utils.build_vocab \
  --dataset "${EPISODES}" \
  --output "${VOCAB_PATH}"

echo "[2/7] Prepare GloVe 6B 50d"
mkdir -p "${GLOVE_DIR}"
if [ ! -f "${GLOVE_TXT}" ]; then
  if [ -n "${LOCAL_GLOVE_TXT}" ] && [ -f "${LOCAL_GLOVE_TXT}" ]; then
    ln -sf "${LOCAL_GLOVE_TXT}" "${GLOVE_TXT}"
    echo "Using local GloVe file: ${LOCAL_GLOVE_TXT}"
  else
    if [ ! -f "${GLOVE_ZIP}" ] || ! unzip -t "${GLOVE_ZIP}" >/dev/null 2>&1; then
      echo "Downloading or resuming ${GLOVE_ZIP}"
    fi
    while [ ! -f "${GLOVE_ZIP}" ] || ! unzip -t "${GLOVE_ZIP}" >/dev/null 2>&1; do
      if command -v curl >/dev/null 2>&1; then
        curl -L -C - http://nlp.stanford.edu/data/glove.6B.zip -o "${GLOVE_ZIP}"
      elif command -v wget >/dev/null 2>&1; then
        wget -c -O "${GLOVE_ZIP}" http://nlp.stanford.edu/data/glove.6B.zip
      else
        echo "Neither curl nor wget is available. Please install one of them."
        exit 1
      fi
    done
    unzip -o "${GLOVE_ZIP}" glove.6B.50d.txt -d "${GLOVE_DIR}"
  fi
fi

echo "[3/7] Build GloVe embeddings"
python -m satnav.utils.build_glove_embeddings \
  --vocab "${VOCAB_PATH}" \
  --glove "${GLOVE_TXT}" \
  --output "${EMBEDDING_PATH}" \
  --embedding-dim 50

echo "[4/7] Download TorchVision ResNet50 weights"
python -c "from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"

echo "[5/7] Generate offline trajectory data"
python -m applications.trajectory_generation.generate \
  --config "${TASK_CONFIG}" \
  --output_dir "${TRAJECTORY_DIR}"

if [ "${RUN_SEQ2SEQ}" = "1" ]; then
  echo "[6/7] Train Seq2Seq"
  python run.py \
    --exp-config configs/baselines/seq2seq_offline_train.yaml \
    --run-type train

  if [ "${RUN_EVAL}" = "1" ]; then
    echo "[6/7] Evaluate Seq2Seq"
    python run.py \
      --exp-config configs/baselines/seq2seq_eval.yaml \
      --run-type eval
  fi
fi

if [ "${RUN_CMA}" = "1" ]; then
  echo "[7/7] Train CMA"
  python run.py \
    --exp-config configs/baselines/cma_offline_train.yaml \
    --run-type train

  if [ "${RUN_EVAL}" = "1" ]; then
    echo "[7/7] Evaluate CMA"
    python run.py \
      --exp-config configs/baselines/cma_eval.yaml \
      --run-type eval
  fi
fi

echo "Done. Outputs are under ${OUT_ROOT}/"

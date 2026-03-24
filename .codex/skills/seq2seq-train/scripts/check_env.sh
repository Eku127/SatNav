#!/usr/bin/env bash
# check_env.sh - Environment & data sanity check for seq2seq training
# Run from REPO root: bash .codex/skills/seq2seq-train/scripts/check_env.sh

set -euo pipefail

REPO=/mnt/data1/home/jiangjiajun/workspace/SatNav
DATASET_ROOT=/mnt/data3/jiangjiajun/dataset/satnav_datasets
VER=ver_260317
GLOVE_DIR=/mnt/data3/jiangjiajun/dataset/glove

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; FAILED=1; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }

FAILED=0

echo "================================================================"
echo " SatNav Seq2Seq Training — Environment Check"
echo "================================================================"

# --- Conda / Python ---
echo ""
echo "[ Python / Conda ]"
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    ok "torch $TORCH_VER"
else
    fail "torch not importable — activate satnav conda env first"
fi

# --- GPU ---
echo ""
echo "[ GPU ]"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "$GPU_COUNT GPU(s) available — $GPU_NAME"
else
    fail "No CUDA GPU detected"
fi

# --- Dataset files ---
echo ""
echo "[ Dataset — $VER ]"

TRAIN_FILE="$DATASET_ROOT/$VER/episodes/train/all_episodes.json"
if [ -f "$TRAIN_FILE" ]; then
    EP_COUNT=$(python -c "import json; d=json.load(open('$TRAIN_FILE')); print(len(d['episodes']))" 2>/dev/null || echo "?")
    ok "train episodes: $EP_COUNT  ($TRAIN_FILE)"
else
    fail "train episodes not found: $TRAIN_FILE"
fi

VAL_SEEN="$DATASET_ROOT/$VER/episodes/eval/val_seen/all_episodes.json"
if [ -f "$VAL_SEEN" ]; then
    ok "val_seen: $VAL_SEEN"
else
    warn "val_seen not found (needed for eval): $VAL_SEEN"
fi

OFFLINE_ANN="$DATASET_ROOT/$VER/trajectory_data/annotations.json"
OFFLINE_IMG_DIR="$DATASET_ROOT/$VER/trajectory_data/images"
if [ -f "$OFFLINE_ANN" ]; then
    ok "offline annotations: $OFFLINE_ANN"
else
    fail "offline annotations not found: $OFFLINE_ANN"
fi

if [ -d "$OFFLINE_IMG_DIR" ] && [ "$(ls -A "$OFFLINE_IMG_DIR" 2>/dev/null)" ]; then
    ok "offline images dir: $OFFLINE_IMG_DIR"
else
    fail "offline images dir empty or missing: $OFFLINE_IMG_DIR"
fi

# --- Vocab ---
echo ""
echo "[ Vocab ]"
VOCAB_FILE="$REPO/output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json"
if [ -f "$VOCAB_FILE" ]; then
    VOCAB_SIZE=$(python -c "import json; d=json.load(open('$VOCAB_FILE')); print(d['vocab_size'])" 2>/dev/null || echo "?")
    ok "vocab_size=$VOCAB_SIZE  ($VOCAB_FILE)"
else
    warn "vocab not found — run Step 2 to build it"
fi

# --- GloVe ---
echo ""
echo "[ GloVe ]"
GLOVE_FILE="$GLOVE_DIR/glove.6B.50d.txt"
if [ -f "$GLOVE_FILE" ]; then
    ok "glove.6B.50d.txt found ($GLOVE_FILE)"
else
    warn "GloVe not found — run Step 3 to download"
fi

# --- Embedding ---
echo ""
echo "[ Embedding ]"
EMBED_FILE="$REPO/output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz"
if [ -f "$EMBED_FILE" ]; then
    ok "embedding file found ($EMBED_FILE)"
else
    warn "embedding not built yet — run Step 4"
fi

# --- Config ---
echo ""
echo "[ Config ]"
OFFLINE_CONFIG="$REPO/configs/baselines/seq2seq_offline.yaml"
if [ -f "$OFFLINE_CONFIG" ]; then
    if grep -q "BASE_TASK_CONFIG_PATH: configs/satnav_task.yaml" "$OFFLINE_CONFIG"; then
        ok "offline BASE_TASK_CONFIG_PATH = configs/satnav_task.yaml"
    else
        fail "offline BASE_TASK_CONFIG_PATH not pointing to satnav_task.yaml in $OFFLINE_CONFIG"
    fi
    if grep -q "TRAINER_NAME: offline_trainer" "$OFFLINE_CONFIG"; then
        ok "offline trainer = offline_trainer"
    else
        fail "offline trainer name mismatch in $OFFLINE_CONFIG"
    fi
    if grep -q "ver_260317" "$OFFLINE_CONFIG"; then
        ok "offline config references ver_260317"
    else
        warn "offline config may not reference ver_260317 — check paths manually"
    fi
else
    fail "Offline config not found: $OFFLINE_CONFIG"
fi

DDP_SCRIPT="$REPO/scripts/seq2seq/train_offline_ddp.sh"
if [ -x "$DDP_SCRIPT" ]; then
    ok "DDP launch script executable: $DDP_SCRIPT"
else
    fail "DDP launch script missing or not executable: $DDP_SCRIPT"
fi

# --- Latest symlink ---
echo ""
echo "[ Latest checkpoint symlink ]"
LATEST_LINK="$REPO/output/seq2seq_offline/checkpoints/latest"
if [ -L "$LATEST_LINK" ]; then
    TARGET=$(readlink "$LATEST_LINK")
    ok "latest -> $TARGET"
elif [ -d "$REPO/output/seq2seq_offline/checkpoints" ]; then
    warn "latest symlink not present — will be created after first training run"
else
    warn "checkpoints dir not present yet"
fi

# --- SwanLab ---
echo ""
echo "[ SwanLab ]"
if python -c "import swanlab" 2>/dev/null; then
    SWANLAB_VER=$(python -c "import swanlab; print(getattr(swanlab, '__version__', 'unknown'))")
    ok "swanlab $SWANLAB_VER"
else
    warn "swanlab not importable — install requirements or pip install 'swanlab[dashboard]'"
fi

# --- Summary ---
echo ""
echo "================================================================"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}Some checks FAILED. Fix the items above before training.${NC}"
    exit 1
fi

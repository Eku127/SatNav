#!/usr/bin/env bash
# build_embeddings.sh - Build GloVe embedding file for ver_260317 vocab
# Run from REPO root: bash .codex/skills/seq2seq-train/scripts/build_embeddings.sh

set -euo pipefail

REPO=/mnt/data1/home/jiangjiajun/workspace/SatNav
VOCAB="$REPO/output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json"
GLOVE=/mnt/data3/jiangjiajun/dataset/glove/glove.6B.50d.txt
OUTPUT_DIR="$REPO/output/seq2seq_offline/artifacts/embeddings"
OUTPUT="$OUTPUT_DIR/embeddings_glove50d_260317.json.gz"

# Prereq checks
if [ ! -f "$VOCAB" ]; then
    echo "[ERROR] Vocab not found: $VOCAB"
    echo "  Run Step 2 first: build vocab"
    exit 1
fi

if [ ! -f "$GLOVE" ]; then
    echo "[ERROR] GloVe file not found: $GLOVE"
    echo "  Run Step 3 first: download_glove.sh"
    exit 1
fi

if [ -f "$OUTPUT" ]; then
    echo "[OK] Embedding already exists: $OUTPUT"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "Building GloVe embeddings ..."
echo "  vocab  : $VOCAB"
echo "  glove  : $GLOVE"
echo "  output : $OUTPUT"
echo ""

cd "$REPO"
python -m satnav.utils.build_glove_embeddings \
    --vocab "$VOCAB" \
    --glove "$GLOVE" \
    --output "$OUTPUT" \
    --embedding-dim 50

echo ""
echo "[DONE] $OUTPUT"
ls -lh "$OUTPUT"

#!/usr/bin/env bash
# download_glove.sh - Download GloVe 6B vectors (50d) from Stanford
# Saves to /mnt/data3/jiangjiajun/dataset/glove/glove.6B.50d.txt

set -euo pipefail

GLOVE_DIR=/mnt/data3/jiangjiajun/dataset/glove
TARGET="$GLOVE_DIR/glove.6B.50d.txt"
ZIP="$GLOVE_DIR/glove.6B.zip"
URL="https://nlp.stanford.edu/data/glove.6B.zip"

if [ -f "$TARGET" ]; then
    echo "[OK] glove.6B.50d.txt already exists: $TARGET"
    exit 0
fi

mkdir -p "$GLOVE_DIR"
echo "Downloading GloVe 6B (~822 MB) ..."
wget -c -O "$ZIP" "$URL"

echo "Unzipping (extracting only 50d file) ..."
unzip -o "$ZIP" "glove.6B.50d.txt" -d "$GLOVE_DIR"

echo "Cleaning up zip ..."
rm -f "$ZIP"

echo ""
echo "[DONE] $TARGET"
wc -l "$TARGET"

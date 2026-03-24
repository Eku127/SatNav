#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${REPO_ROOT}/output"
TARGET_ROOT="${OUTPUT_DIR}/seq2seq_offline"
ARTIFACTS_DIR="${TARGET_ROOT}/artifacts"
LEGACY_DIR="${TARGET_ROOT}/legacy"

mkdir -p \
    "${TARGET_ROOT}/checkpoints" \
    "${TARGET_ROOT}/results" \
    "${TARGET_ROOT}/videos" \
    "${TARGET_ROOT}/logs" \
    "${TARGET_ROOT}/swanlab" \
    "${ARTIFACTS_DIR}" \
    "${LEGACY_DIR}"

shopt -s nullglob dotglob

log() {
    echo "[organize_output] $*"
}

rmdir_if_empty() {
    local dir="$1"
    if [ -d "${dir}" ] && [ -z "$(ls -A "${dir}")" ]; then
        rmdir "${dir}"
    fi
}

move_file_if_needed() {
    local src="$1"
    local dst="$2"

    [ -e "${src}" ] || return 0
    mkdir -p "$(dirname "${dst}")"

    if [ ! -e "${dst}" ]; then
        mv "${src}" "${dst}"
        log "moved file ${src#${REPO_ROOT}/} -> ${dst#${REPO_ROOT}/}"
        return 0
    fi

    if cmp -s "${src}" "${dst}"; then
        rm -f "${src}"
        log "removed duplicate file ${src#${REPO_ROOT}/}"
        return 0
    fi

    log "kept conflicting file ${src#${REPO_ROOT}/}; destination already exists"
}

move_child_dir_if_needed() {
    local src="$1"
    local dst_root="$2"

    [ -e "${src}" ] || return 0
    mkdir -p "${dst_root}"

    local base
    base="$(basename "${src}")"
    local dst="${dst_root}/${base}"

    if [ ! -e "${dst}" ]; then
        mv "${src}" "${dst}"
        log "moved ${src#${REPO_ROOT}/} -> ${dst#${REPO_ROOT}/}"
        return 0
    fi

    if [ -d "${src}" ] && [ -d "${dst}" ]; then
        local child
        for child in "${src}"/*; do
            [ -e "${child}" ] || continue
            move_child_dir_if_needed "${child}" "${dst}"
        done
        rmdir_if_empty "${src}"
        return 0
    fi

    if [ -f "${src}" ] && [ -f "${dst}" ]; then
        move_file_if_needed "${src}" "${dst}"
        return 0
    fi

    log "skipped conflicting path ${src#${REPO_ROOT}/}"
}

move_matching_children() {
    local src_dir="$1"
    local dst_dir="$2"
    local pattern="$3"

    [ -d "${src_dir}" ] || return 0

    local child
    for child in "${src_dir}"/${pattern}; do
        [ -e "${child}" ] || continue
        move_child_dir_if_needed "${child}" "${dst_dir}"
    done

    rmdir_if_empty "${src_dir}"
}

move_dir_to_legacy() {
    local src="$1"
    local name="$2"

    [ -e "${src}" ] || return 0
    mkdir -p "${LEGACY_DIR}"

    local dst="${LEGACY_DIR}/${name}"
    if [ -e "${dst}" ]; then
        log "legacy destination exists, keeping ${src#${REPO_ROOT}/}"
        return 0
    fi

    mv "${src}" "${dst}"
    log "archived ${src#${REPO_ROOT}/} -> ${dst#${REPO_ROOT}/}"
}

move_matching_children "${OUTPUT_DIR}/checkpoints" "${TARGET_ROOT}/checkpoints" "seq2seq*"
move_matching_children "${OUTPUT_DIR}/results" "${TARGET_ROOT}/results" "seq2seq*"
move_matching_children "${OUTPUT_DIR}/videos" "${TARGET_ROOT}/videos" "seq2seq*"
move_matching_children "${OUTPUT_DIR}/logs" "${TARGET_ROOT}/logs" "seq2seq*.log"

# ── 旧 output/ 根目录下散放的 artifacts（早期版本遗留） ──────────────────────
move_child_dir_if_needed "${OUTPUT_DIR}/vocab_260317" "${ARTIFACTS_DIR}"
move_child_dir_if_needed "${OUTPUT_DIR}/embeddings_260317" "${ARTIFACTS_DIR}"
move_child_dir_if_needed "${OUTPUT_DIR}/smoke_260317" "${ARTIFACTS_DIR}"

# ── artifacts 内部：旧的 <type>_<ver>/ 目录结构 → 新的 <type>/<file>_<ver> 结构 ──
# 旧: artifacts/vocab_260317/train_vocab.json
# 新: artifacts/vocab/train_vocab_260317.json

migrate_artifact_file() {
    local old_dir="$1"   # e.g. artifacts/vocab_260317
    local new_dir="$2"   # e.g. artifacts/vocab
    local old_file="$3"  # e.g. train_vocab.json
    local new_file="$4"  # e.g. train_vocab_260317.json

    local src="${ARTIFACTS_DIR}/${old_dir}/${old_file}"
    local dst_dir="${ARTIFACTS_DIR}/${new_dir}"
    local dst="${dst_dir}/${new_file}"

    [ -f "${src}" ] || return 0
    mkdir -p "${dst_dir}"

    if [ -e "${dst}" ]; then
        log "artifact already at new path, skipping: ${dst#${REPO_ROOT}/}"
        return 0
    fi

    mv "${src}" "${dst}"
    log "migrated artifact ${src#${REPO_ROOT}/} -> ${dst#${REPO_ROOT}/}"
    rmdir_if_empty "${ARTIFACTS_DIR}/${old_dir}"
}

migrate_artifact_file "vocab_260317"      "vocab"      "train_vocab.json"             "train_vocab_260317.json"
migrate_artifact_file "embeddings_260317" "embeddings" "embeddings_glove50d.json.gz"  "embeddings_glove50d_260317.json.gz"
migrate_artifact_file "smoke_260317"      "smoke"      "offline_annotations_128.json" "offline_annotations_128_260317.json"

# ── SwanLab ──────────────────────────────────────────────────────────────────
if [ -d "${OUTPUT_DIR}/swanlab" ]; then
    if [ -z "$(ls -A "${TARGET_ROOT}/swanlab" 2>/dev/null)" ]; then
        rmdir_if_empty "${TARGET_ROOT}/swanlab"
        mv "${OUTPUT_DIR}/swanlab" "${TARGET_ROOT}/swanlab"
        log "moved output/swanlab -> output/seq2seq_offline/swanlab"
    else
        move_dir_to_legacy "${OUTPUT_DIR}/swanlab" "swanlab_old_root"
    fi
fi

move_dir_to_legacy "${OUTPUT_DIR}/swanlab_smoke" "swanlab_smoke"
move_dir_to_legacy "${OUTPUT_DIR}/swanlab_cloud_smoke" "swanlab_cloud_smoke"

rmdir_if_empty "${OUTPUT_DIR}/checkpoints"
rmdir_if_empty "${OUTPUT_DIR}/results"
rmdir_if_empty "${OUTPUT_DIR}/videos"
rmdir_if_empty "${OUTPUT_DIR}/logs"

log "done"

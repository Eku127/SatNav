#!/usr/bin/env bash
set -euo pipefail

validation_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
satnav_data_root="${SATNAV_DATA_ROOT:-}"
satnav_scenes_dir="${SATNAV_SCENES_DIR:-}"

if [[ -z "${satnav_data_root}" || -z "${satnav_scenes_dir}" ]]; then
  echo "Set SATNAV_DATA_ROOT and SATNAV_SCENES_DIR before validation." >&2
  exit 2
fi

if [[ ! -d "${satnav_data_root}" ]]; then
  echo "Dataset directory not found: ${satnav_data_root}" >&2
  exit 2
fi

if [[ ! -d "${satnav_scenes_dir}" ]]; then
  echo "Scene directory not found: ${satnav_scenes_dir}" >&2
  exit 2
fi

scene_count="$(find "${satnav_scenes_dir}" -maxdepth 1 -type f -name '*.tif' | wc -l)"
if [[ "${scene_count}" -ne 59 ]]; then
  echo "Expected 59 GeoTIFF scenes, found ${scene_count}: ${satnav_scenes_dir}" >&2
  exit 1
fi

echo "GeoTIFF scenes: ${scene_count}/59"
python "${validation_script_dir}/data_validation.py" \
  --split train="${satnav_data_root}/episodes/train/all_episodes.json" \
  --split val_seen="${satnav_data_root}/episodes/eval/val_seen/all_episodes.json" \
  --split val_unseen="${satnav_data_root}/episodes/eval/val_unseen/all_episodes.json" \
  --expected train=105164 \
  --expected val_seen=4574 \
  --expected val_unseen=8756 \
  --scenes-dir "${satnav_scenes_dir}"

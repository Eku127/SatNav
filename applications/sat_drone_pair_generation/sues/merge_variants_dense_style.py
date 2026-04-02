#!/usr/bin/env python3
"""Merge SUES origin/crop variants into a single DenseUAV-like layout.

Output layout:
  <output-dir>/
  ├── drone/{sample_id}.jpg
  ├── satellite/{sample_id}.jpg
  ├── pairs.csv
  └── dataset_info.json

`sample_id` is reassigned to be globally unique in the merged set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class VariantSpec:
    name: str
    sat_dir: str
    drone_dir: str


def parse_variant_map(text: str) -> List[VariantSpec]:
    # Format:
    #   "orig:satellite:drone,crop384:satellite_crop384:drone_crop384,..."
    specs: List[VariantSpec] = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid variant item: {item}")
        specs.append(VariantSpec(name=parts[0], sat_dir=parts[1], drone_dir=parts[2]))
    if not specs:
        raise ValueError("No variants parsed from --variant-map")
    return specs


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    # Prefer hard link to avoid storage duplication.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_sample_id(index: int, width: int) -> str:
    return f"{index:0{width}d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Source SUES exported directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged output directory")
    parser.add_argument(
        "--variant-map",
        type=str,
        default="orig:satellite:drone,crop384:satellite_crop384:drone_crop384,crop256:satellite_crop256:drone_crop256",
        help="variant_name:sat_dir:drone_dir list separated by comma",
    )
    parser.add_argument("--id-start", type=int, default=1, help="Starting integer for new sample_id")
    parser.add_argument("--id-width", type=int, default=6, help="Zero padding width for sample_id")
    parser.add_argument(
        "--mode",
        type=str,
        default="link",
        choices=["link", "copy"],
        help="link: hardlink if possible, else copy; copy: always copy",
    )
    args = parser.parse_args()

    pairs_csv = args.dataset_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing {pairs_csv}")

    with pairs_csv.open("r", encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))
    if not source_rows:
        raise RuntimeError("Source pairs.csv has no rows")

    specs = parse_variant_map(args.variant_map)

    out_sat = args.output_dir / "satellite"
    out_drone = args.output_dir / "drone"
    out_sat.mkdir(parents=True, exist_ok=True)
    out_drone.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, str]] = []
    next_id = int(args.id_start)
    variant_count: Dict[str, int] = {s.name: 0 for s in specs}

    for spec in specs:
        sat_root = args.dataset_dir / spec.sat_dir
        drone_root = args.dataset_dir / spec.drone_dir
        if not sat_root.exists() or not drone_root.exists():
            print(f"[skip variant] {spec.name}: missing {sat_root} or {drone_root}")
            continue

        for row in source_rows:
            sat_file_src = row.get("satellite_file", "").strip()
            drone_file_src = row.get("drone_file", "").strip()
            if not sat_file_src or not drone_file_src:
                continue

            sat_src = sat_root / sat_file_src
            drone_src = drone_root / drone_file_src
            if not sat_src.exists() or not drone_src.exists():
                continue

            sid = make_sample_id(next_id, args.id_width)
            next_id += 1

            sat_file_new = f"{sid}.jpg"
            drone_file_new = f"{sid}.jpg"

            sat_dst = out_sat / sat_file_new
            drone_dst = out_drone / drone_file_new
            link_or_copy(sat_src, sat_dst, args.mode)
            link_or_copy(drone_src, drone_dst, args.mode)

            out_row: Dict[str, str] = {}
            out_row["sample_id"] = sid
            out_row["source_sample_id"] = row.get("sample_id", "")
            out_row["variant"] = spec.name

            # Keep original metadata (except overwritten file columns/sample_id).
            for k, v in row.items():
                if k in {"sample_id", "satellite_file", "drone_file"}:
                    continue
                out_row[k] = v

            out_row["source_satellite_file"] = sat_file_src
            out_row["source_drone_file"] = drone_file_src
            out_row["satellite_file"] = sat_file_new
            out_row["drone_file"] = drone_file_new
            merged_rows.append(out_row)
            variant_count[spec.name] += 1

    if not merged_rows:
        raise RuntimeError("No merged rows generated")

    # Stable output columns.
    preferred = [
        "sample_id",
        "source_sample_id",
        "variant",
        "scene_id",
        "height",
        "drone_frame",
        "drone_rotation_ccw_deg",
        "sat_crop_frac",
        "match_score",
        "nadir_conf",
        "nadir_residual",
        "nadir_grad_norm",
        "nadir_valid_patches",
        "is_nadir",
        "source_satellite_file",
        "source_drone_file",
        "satellite_file",
        "drone_file",
    ]
    all_keys = []
    seen = set()
    for key in preferred:
        if any(key in r for r in merged_rows):
            all_keys.append(key)
            seen.add(key)
    for r in merged_rows:
        for key in r.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    out_csv = args.output_dir / "pairs.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(merged_rows)

    src_info_path = args.dataset_dir / "dataset_info.json"
    src_info = {}
    if src_info_path.exists():
        src_info = json.loads(src_info_path.read_text(encoding="utf-8"))

    info = {
        "name": "SUES merged variants (DenseUAV-like layout)",
        "source_dataset_dir": str(args.dataset_dir),
        "source_total_pairs": len(source_rows),
        "merged_total_pairs": len(merged_rows),
        "id_start": args.id_start,
        "id_width": args.id_width,
        "variants": variant_count,
        "output_structure": {
            "satellite_dir": "satellite",
            "drone_dir": "drone",
            "pairs_csv": "pairs.csv",
        },
        "source_dataset_info": src_info,
    }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Merged pairs: {len(merged_rows)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

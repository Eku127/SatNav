#!/usr/bin/env python3
"""Apply a synchronous center square recrop to exported SUES sat-drone pairs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def center_recrop(img: Image.Image, crop_size: int, output_size: int) -> Image.Image:
    if crop_size > img.width or crop_size > img.height:
        raise ValueError(f"crop_size={crop_size} exceeds image size {img.size}")
    left = (img.width - crop_size) // 2
    top = (img.height - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size)).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )


def row_paths(row: dict) -> tuple[str, str]:
    sat_rel = row.get("satellite_file", "").strip()
    drone_rel = row.get("drone_file", "").strip()
    if not sat_rel or not drone_rel:
        raise KeyError("pairs.csv must contain satellite_file and drone_file columns")
    return f"satellite/{sat_rel}", f"drone/{drone_rel}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = args.dataset_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing {pairs_csv}")

    with pairs_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        sat_rel, drone_rel = row_paths(row)
        for rel in (sat_rel, drone_rel):
            src = args.dataset_dir / rel
            dst = args.output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(src).convert("RGB")
            center_recrop(img, args.crop_size, args.output_size).save(
                dst, "JPEG", quality=args.jpeg_quality
            )

    shutil.copy2(pairs_csv, args.output_dir / "pairs.csv")

    info_path = args.dataset_dir / "dataset_info.json"
    if info_path.exists():
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        payload = {"name": "sues_pair_export"}

    payload.setdefault("postprocess", {})
    payload["postprocess"] = {
        "type": "center_recrop",
        "crop_size_px": args.crop_size,
        "output_size_px": args.output_size,
    }
    notes = payload.get("notes", [])
    notes.append(
        f"Pairs were postprocessed by synchronously center-cropping both images to {args.crop_size}x{args.crop_size} and resizing to {args.output_size}."
    )
    payload["notes"] = notes
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Center-recrops written to {args.output_dir}")


if __name__ == "__main__":
    main()

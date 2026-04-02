#!/usr/bin/env python3
"""Apply a center square recrop to exported sat-drone pairs."""

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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.dataset_dir / "pairs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key in ("export_satellite_path", "export_drone_path"):
            src = args.dataset_dir / row[key]
            dst = args.output_dir / row[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(src).convert("RGB")
            center_recrop(img, args.crop_size, args.output_size).save(
                dst, "JPEG", quality=args.jpeg_quality
            )

    shutil.copy2(args.dataset_dir / "pairs.csv", args.output_dir / "pairs.csv")

    payload = json.loads((args.dataset_dir / "dataset_info.json").read_text())
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
    (args.output_dir / "dataset_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")

    print(f"Center-recrops written to {args.output_dir}")


if __name__ == "__main__":
    main()

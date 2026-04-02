#!/usr/bin/env python3
"""Render a quick preview grid from an exported UAV-VisLoc pair dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL = 256
GAP = 6
LABEL_H = 40
PAD = 20
ROW_GAP = 8
BG = (20, 20, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_preview(dataset_dir: Path, rows: int, seed: int, output: Path) -> None:
    random.seed(seed)
    with (dataset_dir / "pairs.csv").open() as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        raise RuntimeError("No rows found in pairs.csv")

    picked = random.sample(all_rows, min(rows, len(all_rows)))
    font = ImageFont.load_default()

    width = PAD * 2 + PANEL * 2 + GAP
    height = PAD * 2 + 20 + len(picked) * (LABEL_H + PANEL + ROW_GAP)
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 8), "UAV-VisLoc  -  Satellite (north-up) | Drone (north-up)", fill=(220, 220, 220), font=font)

    y = PAD + 16
    for row in picked:
        label = (
            f"  {row['sample_id']}  "
            f"[seq={row['seq_id']} | heading={row['heading_source']}]  "
            f"crop={row['crop_side_m']}m  rot={row['north_up_rot']} deg"
        )
        draw.rectangle((PAD, y, width - PAD, y + LABEL_H - 4), fill=(38, 38, 44))
        draw.text((PAD + 6, y + 10), label, fill=(185, 185, 185), font=font)
        draw.text((PAD + 6, y + 2), "Satellite", fill=(160, 210, 255), font=font)
        draw.text((PAD + PANEL + GAP + 6, y + 2), "Drone", fill=(160, 255, 160), font=font)
        y += LABEL_H

        sat = Image.open(dataset_dir / row["export_satellite_path"]).convert("RGB")
        drone = Image.open(dataset_dir / row["export_drone_path"]).convert("RGB")
        sat = sat.resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        drone = drone.resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        canvas.paste(sat, (PAD, y))
        canvas.paste(drone, (PAD + PANEL + GAP, y))
        draw.rectangle((PAD, y, PAD + PANEL - 1, y + PANEL - 1), outline=(80, 140, 200), width=1)
        draw.rectangle(
            (PAD + PANEL + GAP, y, PAD + PANEL * 2 + GAP - 1, y + PANEL - 1),
            outline=(80, 180, 100),
            width=1,
        )
        y += PANEL + ROW_GAP

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=95)
    print(f"Saved preview: {output}")


def main() -> None:
    args = parse_args()
    make_preview(args.dataset_dir, args.rows, args.seed, args.output)


if __name__ == "__main__":
    main()

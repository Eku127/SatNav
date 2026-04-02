#!/usr/bin/env python3
"""Render sample preview for exported SUES satellite-drone pairs.

Layout: each row is [Satellite | Drone], left-right.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_preview(
    dataset_dir: Path,
    rows: int,
    seed: int,
    out_path: Path,
    panel: int = 320,
    min_nadir_conf: float = 0.0,
    heights: list[str] | None = None,
) -> None:
    csv_path = dataset_dir / "pairs.csv"
    sat_dir = dataset_dir / "satellite"
    drone_dir = dataset_dir / "drone"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    if "nadir_conf" in (all_rows[0].keys() if all_rows else []):
        all_rows = [r for r in all_rows if float(r.get("nadir_conf", "0") or 0.0) >= min_nadir_conf]

    if heights:
        height_set = {h.strip() for h in heights if h.strip()}
        all_rows = [r for r in all_rows if str(r.get("height", "")).strip() in height_set]

    if not all_rows:
        raise RuntimeError("pairs.csv has no rows")

    rows = max(1, min(rows, len(all_rows)))
    rng = random.Random(seed)
    samples = rng.sample(all_rows, rows)

    pad = 18
    gap = 10
    label_h = 42
    row_gap = 12
    bg = (20, 22, 28)

    width = pad * 2 + panel * 2 + gap
    height = pad * 2 + 28 + rows * (label_h + panel + row_gap)

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, 6), "SUES Pair Preview  |  Left: Satellite  Right: Drone", fill=(220, 220, 220), font=font)

    y = pad + 22
    for row in samples:
        sid = row["sample_id"]
        sat = Image.open(sat_dir / row["satellite_file"]).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
        drone = Image.open(drone_dir / row["drone_file"]).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)

        nadir_part = ""
        if "nadir_conf" in row:
            nadir_part = f" | nadir={row['nadir_conf']}"

        meta = (
            f"{sid} | frame={row['drone_frame']} | rot={row['drone_rotation_ccw_deg']}° "
            f"| crop={row['sat_crop_frac']} | score={row['match_score']}{nadir_part}"
        )

        draw.rectangle((pad, y, width - pad, y + label_h - 5), fill=(34, 37, 44))
        draw.text((pad + 6, y + 12), meta, fill=(190, 190, 190), font=font)
        y += label_h

        canvas.paste(sat, (pad, y))
        canvas.paste(drone, (pad + panel + gap, y))

        draw.rectangle((pad, y, pad + panel - 1, y + panel - 1), outline=(70, 130, 210), width=1)
        draw.rectangle((pad + panel + gap, y, pad + panel * 2 + gap - 1, y + panel - 1), outline=(80, 180, 110), width=1)
        draw.text((pad + 6, y + 6), "sat", fill=(160, 210, 255), font=font)
        draw.text((pad + panel + gap + 6, y + 6), "drone", fill=(170, 255, 170), font=font)

        y += panel + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=95)
    print(f"Saved preview: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/sues_pair_export"))
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/sues_pair_export/preview_samples.jpg"))
    parser.add_argument("--panel", type=int, default=320)
    parser.add_argument("--min-nadir-conf", type=float, default=0.0, help="Only sample rows with nadir_conf >= value")
    parser.add_argument("--heights", type=str, default="", help="Optional comma list, e.g. 150,200")
    args = parser.parse_args()

    heights = [h.strip() for h in args.heights.split(",") if h.strip()] if args.heights else None

    make_preview(
        args.dataset_dir,
        args.rows,
        args.seed,
        args.output,
        panel=args.panel,
        min_nadir_conf=args.min_nadir_conf,
        heights=heights,
    )


if __name__ == "__main__":
    main()

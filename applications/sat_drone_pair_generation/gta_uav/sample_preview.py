#!/usr/bin/env python3
"""Sample random exported GTA pairs and render a satellite vs drone preview grid.

Layout: one heading-status group per block, each row = [Satellite | Drone].
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL = 256
GAP = 6
LABEL_H = 38
GROUP_GAP = 28
PAD = 20
BG = (20, 20, 24)
GROUP_HDR = {
    "confirmed": (60, 160, 100),
    "corrected": (200, 120, 50),
    "ambiguous": (180, 80, 80),
    "unchecked": (90, 120, 200),
}
DEFAULT_GROUPS = ["confirmed"]


def sample_rows(rows, groups, rows_per_group, seed):
    random.seed(seed)
    grouped = {group: [row for row in rows if row["heading_status"] == group] for group in groups}
    samples = {}
    for group in groups:
        items = grouped[group]
        if not items:
            samples[group] = []
        else:
            k = min(rows_per_group, len(items))
            samples[group] = random.sample(items, k)
    return samples


def make_preview(dataset_dir: Path, groups, rows_per_group: int, seed: int, out_path: Path) -> None:
    with (dataset_dir / "pairs.csv").open() as f:
        all_rows = list(csv.DictReader(f))

    samples = sample_rows(all_rows, groups, rows_per_group, seed)
    font_sm = ImageFont.load_default()

    non_empty_groups = [group for group in groups if samples[group]]
    if not non_empty_groups:
        raise RuntimeError("No matching rows found for the requested preview groups.")

    pair_w = PANEL * 2 + GAP
    group_hs = [LABEL_H + len(samples[group]) * (LABEL_H + PANEL + GAP) for group in non_empty_groups]
    total_w = PAD * 2 + pair_w
    total_h = PAD * 2 + sum(group_hs) + max(0, len(group_hs) - 1) * GROUP_GAP

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 6), "GTA-UAV  —  Satellite (north-up) | Drone (north-up)", fill=(220, 220, 220), font=font_sm)

    y = PAD + 16
    for group in non_empty_groups:
        color = GROUP_HDR.get(group, (80, 80, 120))
        draw.rectangle((PAD, y, total_w - PAD, y + LABEL_H - 4), fill=color)
        hdr = f"  {group}  ({len(samples[group])} samples)"
        draw.text((PAD + 6, y + 10), hdr, fill=(255, 255, 255), font=font_sm)
        y += LABEL_H

        for row in samples[group]:
            sid = row["sample_id"]
            split = row["split"]
            area_mode = row["area_mode"]
            rot = row["north_up_rot"]
            iou = row["iou"]

            sat = Image.open(dataset_dir / row["export_satellite_path"]).convert("RGB")
            drone = Image.open(dataset_dir / row["export_drone_path"]).convert("RGB")
            sat = sat.resize((PANEL, PANEL), Image.Resampling.LANCZOS)
            drone = drone.resize((PANEL, PANEL), Image.Resampling.LANCZOS)

            label = f"  {sid}  [{split} | {area_mode}]  iou={iou}  rot={rot}°"
            draw.rectangle((PAD, y, total_w - PAD, y + LABEL_H - 4), fill=(38, 38, 44))
            draw.text((PAD + 6, y + 10), label, fill=(180, 180, 180), font=font_sm)
            draw.text((PAD + 4, y + 2), "Satellite ↑N", fill=(160, 210, 255), font=font_sm)
            draw.text((PAD + PANEL + GAP + 4, y + 2), "Drone ↑N", fill=(160, 255, 160), font=font_sm)
            y += LABEL_H

            canvas.paste(sat, (PAD, y))
            canvas.paste(drone, (PAD + PANEL + GAP, y))
            draw.rectangle((PAD, y, PAD + PANEL - 1, y + PANEL - 1), outline=(80, 140, 200), width=1)
            draw.rectangle(
                (PAD + PANEL + GAP, y, PAD + PANEL * 2 + GAP - 1, y + PANEL - 1),
                outline=(80, 180, 100),
                width=1,
            )
            y += PANEL + GAP

        y += GROUP_GAP

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "JPEG", quality=95)
    print(f"Saved: {out_path}  ({canvas.size[0]}x{canvas.size[1]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/data3/jiangjiajun/dataset/SatDronePair/gta"),
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=DEFAULT_GROUPS,
        help="Heading-status groups to preview. Default: confirmed",
    )
    parser.add_argument("--rows", type=int, default=4, help="Samples per group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data3/jiangjiajun/dataset/SatDronePair/gta/samples_preview.jpg"),
    )
    args = parser.parse_args()

    make_preview(args.dataset_dir, args.groups, args.rows, args.seed, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sample random pairs from denseuav and render a satellite vs drone comparison grid.

Layout:  N rows × 3 altitude groups, each group = [Satellite | Drone (north-up)]
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── layout ────────────────────────────────────────────────────────────────────
PANEL    = 256          # px per image panel
GAP      = 6            # gap between panels
LABEL_H  = 38
GROUP_GAP = 28          # gap between altitude groups
PAD      = 20
BG       = (20, 20, 24)
ALT_HDR  = {"H80": (70, 130, 200), "H90": (60, 180, 100), "H100": (200, 110, 50)}

ALTITUDES = ["H80", "H90", "H100"]
ROWS_PER_ALT = 4        # samples per altitude


def rotate_north_up(img: Image.Image, deg: int) -> Image.Image:
    """Mirror-pad then rotate CCW by deg, returning original-size square."""
    w, h = img.size
    fh  = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fv  = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    fhv = fh.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    c   = Image.new(img.mode, (w * 3, h * 3))
    for ri, imgs in enumerate([(fhv, fv, fhv), (fh, img, fh), (fhv, fv, fhv)]):
        for ci, tile in enumerate(imgs):
            c.paste(tile, (ci * w, ri * h))
    rot  = c.rotate(deg, resample=Image.Resampling.BICUBIC)
    cx   = rot.size[0] // 2
    half = w // 2
    return rot.crop((cx - half, cx - half, cx + half, cx + half))


def make_preview(
    base: Path,
    rows_per_alt: int,
    seed: int,
    out_path: Path,
) -> None:
    random.seed(seed)

    with open(base / "pairs.csv") as f:
        all_rows = list(csv.DictReader(f))

    by_alt = {a: [r for r in all_rows if r["altitude"] == a] for a in ALTITUDES}
    samples = {a: random.sample(by_alt[a], rows_per_alt) for a in ALTITUDES}

    font_sm = ImageFont.load_default()

    # ── canvas size ───────────────────────────────────────────────────────────
    # 3 altitude groups stacked vertically, each = header + rows of [sat | drone]
    group_h  = LABEL_H + rows_per_alt * (LABEL_H + PANEL + GAP)
    total_h  = PAD * 2 + 3 * group_h + 2 * GROUP_GAP
    pair_w   = PANEL * 2 + GAP
    total_w  = PAD * 2 + pair_w

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Title
    title = "DenseUAV  —  Satellite (north-up) | Drone (north-up)"
    draw.text((PAD, 6), title, fill=(220, 220, 220), font=font_sm)

    y = PAD + 16
    for alt in ALTITUDES:
        color = ALT_HDR[alt]

        # Altitude header bar
        draw.rectangle((PAD, y, total_w - PAD, y + LABEL_H - 4), fill=color)
        alt_m  = int(alt[1:])
        gsd_d  = {80: 18.5, 90: 20.8, 100: 23.1}[alt_m]
        hdr    = f"  {alt}  (altitude {alt_m} m  |  drone GSD ≈ {gsd_d} cm/px)"
        draw.text((PAD + 6, y + 10), hdr, fill=(255, 255, 255), font=font_sm)
        y += LABEL_H

        for row in samples[alt]:
            sid        = row["sample_id"]
            split      = row["split"]
            north_rot  = int(row["north_up_rot"])

            sat   = Image.open(base / "satellite" / f"{sid}.jpg").convert("RGB")
            drone = Image.open(base / "drone"     / f"{sid}.jpg").convert("RGB")

            sat_rot   = rotate_north_up(sat,   north_rot).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
            drone_rot = rotate_north_up(drone, north_rot).resize((PANEL, PANEL), Image.Resampling.LANCZOS)

            # Row label
            orig_id = sid.replace(f"_{alt}", "")
            label   = f"  {orig_id}  [{split}]  rot={north_rot}°"
            draw.rectangle((PAD, y, total_w - PAD, y + LABEL_H - 4), fill=(38, 38, 44))
            draw.text((PAD + 6, y + 10), label, fill=(180, 180, 180), font=font_sm)

            # Column labels (first row only)
            col_label_y = y + 2
            draw.text((PAD + 4,          col_label_y), "Satellite ↑N", fill=(160, 210, 255), font=font_sm)
            draw.text((PAD + PANEL + GAP + 4, col_label_y), "Drone ↑N",     fill=(160, 255, 160), font=font_sm)
            y += LABEL_H

            canvas.paste(sat_rot,   (PAD,                 y))
            canvas.paste(drone_rot, (PAD + PANEL + GAP,   y))

            # Subtle borders
            draw.rectangle((PAD, y, PAD + PANEL - 1, y + PANEL - 1),           outline=(80, 140, 200), width=1)
            draw.rectangle((PAD + PANEL + GAP, y, PAD + PANEL * 2 + GAP - 1, y + PANEL - 1),
                           outline=(80, 180, 100), width=1)
            y += PANEL + GAP

        y += GROUP_GAP

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "JPEG", quality=95)
    print(f"Saved: {out_path}  ({canvas.size[0]}×{canvas.size[1]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("/mnt/data3/jiangjiajun/dataset/SatDronePair/denseuav"))
    parser.add_argument("--rows",        type=int, default=ROWS_PER_ALT,
                        help="Samples per altitude group (default: 4)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      type=Path,
                        default=Path("/mnt/data3/jiangjiajun/dataset/SatDronePair/denseuav/samples_preview.jpg"))
    args = parser.parse_args()

    make_preview(args.dataset_dir, args.rows, args.seed, args.output)


if __name__ == "__main__":
    main()

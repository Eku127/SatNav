#!/usr/bin/env python3
"""Run SUES pair export pipeline end-to-end.

Stages:
1) Build aligned satellite-drone pairs with nadir filtering.
2) Generate mixed preview and per-height previews.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List

from PIL import Image


def parse_heights(value: str) -> List[str]:
    heights = [x.strip() for x in value.split(",") if x.strip()]
    if not heights:
        raise ValueError("--heights cannot be empty")
    return heights


def parse_int_list(value: str) -> List[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        return []
    return [int(x) for x in items]


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def center_recrop(img: Image.Image, crop_size: int, output_size: int) -> Image.Image:
    if crop_size > img.width or crop_size > img.height:
        raise ValueError(f"crop_size={crop_size} exceeds image size {img.size}")
    left = (img.width - crop_size) // 2
    top = (img.height - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size)).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )


def materialize_recrops(
    output_dir: Path,
    pairs: List[dict],
    crop_sizes: List[int],
    output_size: int,
    jpeg_quality: int,
) -> None:
    for crop_size in crop_sizes:
        sat_dst_dir = output_dir / f"satellite_crop{crop_size}"
        drn_dst_dir = output_dir / f"drone_crop{crop_size}"
        sat_dst_dir.mkdir(parents=True, exist_ok=True)
        drn_dst_dir.mkdir(parents=True, exist_ok=True)

        for row in pairs:
            sat_file = row.get("satellite_file", "").strip()
            drn_file = row.get("drone_file", "").strip()
            if not sat_file or not drn_file:
                continue

            sat_src = output_dir / "satellite" / sat_file
            drn_src = output_dir / "drone" / drn_file
            sat_dst = sat_dst_dir / sat_file
            drn_dst = drn_dst_dir / drn_file

            sat_img = Image.open(sat_src).convert("RGB")
            drn_img = Image.open(drn_src).convert("RGB")
            center_recrop(sat_img, crop_size, output_size).save(sat_dst, "JPEG", quality=jpeg_quality)
            center_recrop(drn_img, crop_size, output_size).save(drn_dst, "JPEG", quality=jpeg_quality)

    variants = {
        "origin": {
            "satellite_dir": "satellite",
            "drone_dir": "drone",
        }
    }
    for crop_size in crop_sizes:
        variants[f"crop{crop_size}"] = {
            "crop_size_px": crop_size,
            "output_size_px": output_size,
            "satellite_dir": f"satellite_crop{crop_size}",
            "drone_dir": f"drone_crop{crop_size}",
        }

    (output_dir / "variants.json").write_text(
        json.dumps({"variants": variants}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/SUES-200-512x512"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sues_pair_export"))
    parser.add_argument("--heights", type=str, default="150,200,250,300")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes")

    # Keep current working threshold by default.
    parser.add_argument("--nadir-min-conf", type=float, default=0.30)
    parser.add_argument("--crop-sizes", type=str, default="384,256", help="Center recrop variants to generate")
    parser.add_argument("--skip-recrops", action="store_true", help="Do not generate crop variants")
    parser.add_argument("--recrop-output-size", type=int, default=512)
    parser.add_argument("--recrop-jpeg-quality", type=int, default=95)

    # Build parameters.
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--match-size", type=int, default=128)
    parser.add_argument("--coarse-angle-step", type=int, default=15)
    parser.add_argument("--coarse-fracs", type=str, default="0.7,0.8,0.9,1.0")
    parser.add_argument("--top-k-frames", type=int, default=6)
    parser.add_argument("--fine-angle-radius", type=int, default=20)
    parser.add_argument("--fine-angle-step", type=int, default=5)
    parser.add_argument("--fine-frac-radius", type=float, default=0.12)
    parser.add_argument("--fine-frac-step", type=float, default=0.03)
    parser.add_argument("--min-crop-frac", type=float, default=0.55)
    parser.add_argument("--min-score", type=float, default=-1.0)

    # Preview parameters.
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--preview-rows", type=int, default=8)
    parser.add_argument("--preview-mixed-rows", type=int, default=12)
    parser.add_argument("--preview-seed", type=int, default=42)

    args = parser.parse_args()

    heights = parse_heights(args.heights)
    crop_sizes = parse_int_list(args.crop_sizes)
    this_dir = Path(__file__).resolve().parent
    build_script = this_dir / "build_pairs.py"
    preview_script = this_dir / "sample_preview.py"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        str(build_script),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output_dir),
        "--heights",
        ",".join(heights),
        "--nadir-min-conf",
        str(args.nadir_min_conf),
        "--frame-stride",
        str(args.frame_stride),
        "--match-size",
        str(args.match_size),
        "--coarse-angle-step",
        str(args.coarse_angle_step),
        "--coarse-fracs",
        args.coarse_fracs,
        "--top-k-frames",
        str(args.top_k_frames),
        "--fine-angle-radius",
        str(args.fine_angle_radius),
        "--fine-angle-step",
        str(args.fine_angle_step),
        "--fine-frac-radius",
        str(args.fine_frac_radius),
        "--fine-frac-step",
        str(args.fine_frac_step),
        "--min-crop-frac",
        str(args.min_crop_frac),
        "--min-score",
        str(args.min_score),
    ]
    if args.max_scenes > 0:
        build_cmd.extend(["--max-scenes", str(args.max_scenes)])

    run_cmd(build_cmd)

    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = output_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing pairs.csv after build: {pairs_csv}")

    with pairs_csv.open("r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    if not args.skip_recrops and crop_sizes:
        print(f"[info] generating recrop variants: {crop_sizes}", flush=True)
        materialize_recrops(
            output_dir=output_dir,
            pairs=pairs,
            crop_sizes=crop_sizes,
            output_size=args.recrop_output_size,
            jpeg_quality=args.recrop_jpeg_quality,
        )

    if args.skip_preview:
        print("[done] build+recrops completed, preview skipped")
        return

    valid_pairs = [r for r in pairs if float(r.get("nadir_conf", "0") or 0.0) >= args.nadir_min_conf]
    count_by_height = Counter(str(r.get("height", "")).strip() for r in valid_pairs)

    if not valid_pairs:
        print("[warn] no samples passed nadir filter; skip all previews")
        print(f"[done] pipeline completed: {output_dir}")
        return

    # Mixed preview.
    mixed_out = previews_dir / f"preview_mixed_seed{args.preview_seed}.jpg"
    mixed_cmd = [
        sys.executable,
        str(preview_script),
        "--dataset-dir",
        str(output_dir),
        "--rows",
        str(args.preview_mixed_rows),
        "--min-nadir-conf",
        str(args.nadir_min_conf),
        "--seed",
        str(args.preview_seed),
        "--output",
        str(mixed_out),
    ]
    run_cmd(mixed_cmd)

    # Per-height previews.
    for h in heights:
        if count_by_height.get(h, 0) <= 0:
            print(f"[skip] no kept samples for height={h}, skip preview")
            continue

        out_file = previews_dir / f"preview_H{h}_seed{args.preview_seed}.jpg"
        cmd = [
            sys.executable,
            str(preview_script),
            "--dataset-dir",
            str(output_dir),
            "--rows",
            str(args.preview_rows),
            "--min-nadir-conf",
            str(args.nadir_min_conf),
            "--heights",
            h,
            "--seed",
            str(args.preview_seed),
            "--output",
            str(out_file),
        ]
        run_cmd(cmd)

    print(f"[done] pipeline completed: {output_dir}")


if __name__ == "__main__":
    main()

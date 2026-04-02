#!/usr/bin/env python3
"""Build satellite-drone pairs from SUES-200 with orientation and scale alignment.

Assumptions for SUES-200-512x512:
- Satellite:  data_root/satellite-view/<scene_id>/0.png
- Drone:      data_root/drone_view_512/<scene_id>/<height>/<frame>.jpg

For each (scene_id, height):
1) Search drone frame index + rotation angle + satellite center-crop fraction.
2) Objective is edge-map NCC (coarse -> fine).
3) Export aligned pair:
   - left satellite (center-cropped to matched scale)
   - right drone (rotated to matched orientation)

Output:
- <output_dir>/satellite/{sample_id}.jpg
- <output_dir>/drone/{sample_id}.jpg
- <output_dir>/pairs.csv
- <output_dir>/dataset_info.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class PairRecord:
    sample_id: str
    scene_id: str
    height: str
    drone_frame: int
    drone_rotation_ccw_deg: int
    sat_crop_frac: float
    match_score: float
    nadir_conf: float
    nadir_residual: float
    nadir_grad_norm: float
    nadir_valid_patches: int
    is_nadir: int
    satellite_file: str
    drone_file: str


def parse_csv_str(values: str) -> List[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def parse_csv_floats(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr = arr - float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / std


def edge_map(gray_img: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(gray_img, axis=1, mode="reflect")
    gy = ndimage.sobel(gray_img, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    mag = ndimage.gaussian_filter(mag, sigma=1.0)
    return normalize_map(mag)


def load_gray(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (dy, dx, peak) from phase correlation."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.abs(np.fft.ifft2(cross))

    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    return float(y), float(x), float(corr.max())


def estimate_nadir_confidence(
    sat_rgb: Image.Image,
    drone_rgb: Image.Image,
    eval_size: int,
    grid: int,
    patch_size: int,
    peak_thresh: float,
    residual_w: float,
    grad_w: float,
) -> Tuple[float, float, float, int]:
    """Estimate nadir confidence by local-shift consistency across patches.

    If local shifts are similar over the image, the pair is more likely nadir-like.
    """
    sat = np.asarray(
        sat_rgb.convert("L").resize((eval_size, eval_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    drone = np.asarray(
        drone_rgb.convert("L").resize((eval_size, eval_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0

    grid = max(3, int(grid))
    patch_size = max(24, int(patch_size))
    half = patch_size // 2

    anchors = np.linspace(half + 4, eval_size - half - 4, grid).astype(int)
    rows: List[Tuple[float, float, float, float, float]] = []

    for cy in anchors:
        for cx in anchors:
            y0 = max(0, cy - half)
            y1 = min(eval_size, cy + half)
            x0 = max(0, cx - half)
            x1 = min(eval_size, cx + half)
            if y1 - y0 < 20 or x1 - x0 < 20:
                continue

            sat_p = normalize_map(sat[y0:y1, x0:x1])
            drn_p = normalize_map(drone[y0:y1, x0:x1])
            dy, dx, peak = phase_correlation_shift(sat_p, drn_p)
            rows.append((float(cx), float(cy), dx, dy, peak))

    if len(rows) < 6:
        return 0.0, 0.0, 0.0, 0

    arr = np.asarray(rows, dtype=np.float32)
    arr = arr[arr[:, 4] >= float(peak_thresh)]
    if len(arr) < 6:
        return 0.0, 0.0, 0.0, int(len(arr))

    dx = arr[:, 2]
    dy = arr[:, 3]

    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    residuals = np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)
    residual = float(np.percentile(residuals, 60))

    # Fit linear field: shift_x/y = a*x + b*y + c
    X = np.stack([arr[:, 0], arr[:, 1], np.ones(len(arr), dtype=np.float32)], axis=1)
    bx = np.linalg.lstsq(X, dx, rcond=None)[0]
    by = np.linalg.lstsq(X, dy, rcond=None)[0]
    grad = np.asarray([[bx[0], bx[1]], [by[0], by[1]]], dtype=np.float32)
    grad_norm = float(np.linalg.norm(grad, ord="fro"))

    conf = float(np.exp(-(float(residual_w) * residual + float(grad_w) * grad_norm)))
    conf = float(np.clip(conf, 0.0, 1.0))
    return conf, residual, grad_norm, int(len(arr))


def center_crop_fraction(img: Image.Image, frac: float) -> Image.Image:
    frac = float(np.clip(frac, 0.5, 1.0))
    if frac >= 0.999:
        return img.copy()

    w, h = img.size
    cw = max(8, int(round(w * frac)))
    ch = max(8, int(round(h * frac)))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def rotate_square_reflect(img: Image.Image, ccw_deg: int) -> Image.Image:
    """Rotate without corner black padding by mirrored 3x3 tiling."""
    w, h = img.size
    fh = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fv = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    fhv = fh.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    canvas = Image.new(img.mode, (w * 3, h * 3))
    tiles = [
        (fhv, fv, fhv),
        (fh, img, fh),
        (fhv, fv, fhv),
    ]
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            canvas.paste(tile, (c * w, r * h))

    rotated = canvas.rotate(int(ccw_deg) % 360, resample=Image.Resampling.BICUBIC)
    cx, cy = rotated.size[0] // 2, rotated.size[1] // 2
    half_w, half_h = w // 2, h // 2
    return rotated.crop((cx - half_w, cy - half_h, cx - half_w + w, cy - half_h + h))


def sat_edge_from_image(sat_gray: Image.Image, frac: float, match_size: int) -> np.ndarray:
    sat_c = center_crop_fraction(sat_gray, frac)
    sat_r = sat_c.resize((match_size, match_size), Image.Resampling.BICUBIC)
    return edge_map(np.asarray(sat_r, dtype=np.float32) / 255.0)


def angle_candidates(center: int, radius: int, step: int) -> List[int]:
    vals = [((center + d) % 360) for d in range(-radius, radius + 1, step)]
    ordered = []
    seen = set()
    for v in vals:
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered


def frac_candidates(center: float, radius: float, step: float, min_frac: float) -> List[float]:
    vals: List[float] = []
    cur = center - radius
    while cur <= center + radius + 1e-9:
        vals.append(float(np.clip(cur, min_frac, 1.0)))
        cur += step
    vals.append(float(np.clip(center, min_frac, 1.0)))
    vals = sorted(set(round(v, 4) for v in vals))
    return vals


def search_best_alignment(
    sat_path: Path,
    drone_frame_paths: Sequence[Path],
    match_size: int,
    coarse_angles: Sequence[int],
    coarse_fracs: Sequence[float],
    top_k_frames: int,
    fine_angle_radius: int,
    fine_angle_step: int,
    fine_frac_radius: float,
    fine_frac_step: float,
    min_crop_frac: float,
) -> Tuple[Path, int, float, float]:
    sat_gray = Image.open(sat_path).convert("L")

    coarse_fracs = sorted(set(round(float(np.clip(f, min_crop_frac, 1.0)), 4) for f in coarse_fracs))
    sat_cache: Dict[float, np.ndarray] = {
        frac: sat_edge_from_image(sat_gray, frac, match_size) for frac in coarse_fracs
    }

    drone_edges: Dict[Path, np.ndarray] = {}
    frame_local_best: Dict[Path, Tuple[float, int, float]] = {}

    global_best_score = -1e9
    global_best: Tuple[Path, int, float] | None = None

    for frame_path in drone_frame_paths:
        de = edge_map(load_gray(frame_path, match_size))
        drone_edges[frame_path] = de

        local_best_score = -1e9
        local_best_angle = 0
        local_best_frac = 1.0

        for angle in coarse_angles:
            rot = normalize_map(ndimage.rotate(de, angle, reshape=False, order=1, mode="reflect"))
            for frac, sat_edge in sat_cache.items():
                score = float(np.mean(rot * sat_edge))
                if score > local_best_score:
                    local_best_score = score
                    local_best_angle = int(angle)
                    local_best_frac = float(frac)

        frame_local_best[frame_path] = (local_best_score, local_best_angle, local_best_frac)

        if local_best_score > global_best_score:
            global_best_score = local_best_score
            global_best = (frame_path, local_best_angle, local_best_frac)

    if global_best is None:
        raise RuntimeError(f"No alignment found for satellite={sat_path}")

    sorted_frames = sorted(
        frame_local_best.items(),
        key=lambda kv: kv[1][0],
        reverse=True,
    )

    chosen = [fp for fp, _ in sorted_frames[: max(1, top_k_frames)]]
    best_frame_coarse, best_angle_coarse, best_frac_coarse = global_best

    # Fine candidates: top-k frames plus nearest neighbors in index order.
    idx_lookup = {fp: i for i, fp in enumerate(drone_frame_paths)}
    fine_idx = set()
    for fp in chosen:
        base = idx_lookup[fp]
        for j in range(max(0, base - 1), min(len(drone_frame_paths), base + 2)):
            fine_idx.add(j)
    fine_frame_paths = [drone_frame_paths[i] for i in sorted(fine_idx)]

    fine_angles = angle_candidates(best_angle_coarse, fine_angle_radius, fine_angle_step)
    fine_fracs = frac_candidates(best_frac_coarse, fine_frac_radius, fine_frac_step, min_crop_frac)
    sat_cache_fine = {
        frac: sat_edge_from_image(sat_gray, frac, match_size) for frac in fine_fracs
    }

    best_score = global_best_score
    best_frame = best_frame_coarse
    best_angle = best_angle_coarse
    best_frac = best_frac_coarse

    for frame_path in fine_frame_paths:
        de = drone_edges[frame_path]
        for angle in fine_angles:
            rot = normalize_map(ndimage.rotate(de, angle, reshape=False, order=1, mode="reflect"))
            for frac, sat_edge in sat_cache_fine.items():
                score = float(np.mean(rot * sat_edge))
                if score > best_score:
                    best_score = score
                    best_frame = frame_path
                    best_angle = int(angle)
                    best_frac = float(frac)

    return best_frame, best_angle, best_frac, float(best_score)


def render_aligned_pair(
    sat_path: Path,
    drone_path: Path,
    sat_crop_frac: float,
    drone_ccw_deg: int,
    out_size: int,
) -> Tuple[Image.Image, Image.Image]:
    sat = Image.open(sat_path).convert("RGB")
    drone = Image.open(drone_path).convert("RGB")

    sat_aligned = center_crop_fraction(sat, sat_crop_frac).resize((out_size, out_size), Image.Resampling.LANCZOS)
    drone_aligned = rotate_square_reflect(drone, drone_ccw_deg).resize((out_size, out_size), Image.Resampling.LANCZOS)

    return sat_aligned, drone_aligned


def save_pair_images(
    sat_aligned: Image.Image,
    drone_aligned: Image.Image,
    out_sat_path: Path,
    out_drone_path: Path,
    jpeg_quality: int,
) -> None:
    out_sat_path.parent.mkdir(parents=True, exist_ok=True)
    out_drone_path.parent.mkdir(parents=True, exist_ok=True)

    sat_aligned.save(out_sat_path, "JPEG", quality=jpeg_quality)
    drone_aligned.save(out_drone_path, "JPEG", quality=jpeg_quality)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/SUES-200-512x512"),
        help="SUES root with satellite-view/ and drone_view_512/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sues_pair_export"),
        help="Output directory for aligned pairs",
    )
    parser.add_argument("--heights", type=str, default="150", help="Comma list, e.g. 150,200,250,300")
    parser.add_argument("--scene-ids", type=str, default="", help="Optional comma list of scene ids like 0001,0002")
    parser.add_argument("--max-scenes", type=int, default=0, help="If >0, only process first N scene ids")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every k-th frame in coarse stage")

    parser.add_argument("--match-size", type=int, default=128, help="Search resolution for matching")
    parser.add_argument("--coarse-angle-step", type=int, default=15, help="Coarse rotation step in degrees")
    parser.add_argument(
        "--coarse-fracs",
        type=str,
        default="0.7,0.8,0.9,1.0",
        help="Satellite center-crop fractions used in coarse stage",
    )
    parser.add_argument("--top-k-frames", type=int, default=6, help="Top coarse frames entering fine stage")
    parser.add_argument("--fine-angle-radius", type=int, default=20, help="Fine search angle radius around coarse best")
    parser.add_argument("--fine-angle-step", type=int, default=5, help="Fine search angle step")
    parser.add_argument("--fine-frac-radius", type=float, default=0.12, help="Fine search crop radius around coarse best")
    parser.add_argument("--fine-frac-step", type=float, default=0.03, help="Fine search crop step")
    parser.add_argument("--min-crop-frac", type=float, default=0.55, help="Lower bound for sat crop fraction")

    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-score", type=float, default=-1.0, help="Filter out low-quality matches")
    parser.add_argument("--nadir-min-conf", type=float, default=0.0, help="Keep samples with nadir_conf >= value")
    parser.add_argument("--nadir-eval-size", type=int, default=256, help="Resolution for nadir confidence")
    parser.add_argument("--nadir-grid", type=int, default=4, help="Grid count per side for nadir estimation")
    parser.add_argument("--nadir-patch-size", type=int, default=56, help="Patch size in nadir estimation")
    parser.add_argument("--nadir-peak-thresh", type=float, default=0.045, help="Phase peak threshold")
    parser.add_argument("--nadir-residual-weight", type=float, default=0.06, help="Residual weight in conf")
    parser.add_argument("--nadir-grad-weight", type=float, default=3.0, help="Gradient weight in conf")
    args = parser.parse_args()

    sat_root = args.data_root / "satellite-view"
    drone_root = args.data_root / "drone_view_512"

    if not sat_root.exists() or not drone_root.exists():
        raise FileNotFoundError(f"Missing expected folders under {args.data_root}")

    if args.scene_ids:
        scene_ids = parse_csv_str(args.scene_ids)
    else:
        sat_ids = {p.name for p in sat_root.iterdir() if p.is_dir()}
        drone_ids = {p.name for p in drone_root.iterdir() if p.is_dir()}
        scene_ids = sorted(sat_ids & drone_ids)

    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    heights = parse_csv_str(args.heights)
    coarse_fracs = parse_csv_floats(args.coarse_fracs)
    coarse_angles = list(range(0, 360, max(1, args.coarse_angle_step)))

    out_sat_dir = args.output_dir / "satellite"
    out_drone_dir = args.output_dir / "drone"
    out_sat_dir.mkdir(parents=True, exist_ok=True)
    out_drone_dir.mkdir(parents=True, exist_ok=True)

    total_targets = len(scene_ids) * len(heights)
    done = 0
    kept = 0

    rows: List[PairRecord] = []

    for scene_id in scene_ids:
        sat_path = sat_root / scene_id / "0.png"
        if not sat_path.exists():
            continue

        for height in heights:
            done += 1
            drone_height_dir = drone_root / scene_id / height
            if not drone_height_dir.exists():
                print(f"[skip] scene={scene_id} height={height} (no drone folder)")
                continue

            frame_paths = sorted(drone_height_dir.glob("*.jpg"), key=numeric_stem)
            frame_paths = frame_paths[:: max(1, args.frame_stride)]
            if not frame_paths:
                print(f"[skip] scene={scene_id} height={height} (no frames)")
                continue

            best_frame, best_angle, best_frac, best_score = search_best_alignment(
                sat_path=sat_path,
                drone_frame_paths=frame_paths,
                match_size=args.match_size,
                coarse_angles=coarse_angles,
                coarse_fracs=coarse_fracs,
                top_k_frames=max(1, args.top_k_frames),
                fine_angle_radius=max(0, args.fine_angle_radius),
                fine_angle_step=max(1, args.fine_angle_step),
                fine_frac_radius=max(0.0, args.fine_frac_radius),
                fine_frac_step=max(1e-3, args.fine_frac_step),
                min_crop_frac=float(np.clip(args.min_crop_frac, 0.3, 1.0)),
            )

            if best_score < args.min_score:
                print(
                    f"[drop] scene={scene_id} height={height} score={best_score:.4f} "
                    f"(< min_score={args.min_score})"
                )
                continue

            sample_id = f"{scene_id}_H{height}"
            sat_file = f"{sample_id}.jpg"
            drone_file = f"{sample_id}.jpg"

            sat_aligned, drone_aligned = render_aligned_pair(
                sat_path=sat_path,
                drone_path=best_frame,
                sat_crop_frac=best_frac,
                drone_ccw_deg=best_angle,
                out_size=args.output_size,
            )

            nadir_conf, nadir_residual, nadir_grad_norm, nadir_valid = estimate_nadir_confidence(
                sat_rgb=sat_aligned,
                drone_rgb=drone_aligned,
                eval_size=args.nadir_eval_size,
                grid=args.nadir_grid,
                patch_size=args.nadir_patch_size,
                peak_thresh=args.nadir_peak_thresh,
                residual_w=args.nadir_residual_weight,
                grad_w=args.nadir_grad_weight,
            )
            is_nadir = int(nadir_conf >= args.nadir_min_conf)
            if not is_nadir:
                print(
                    f"[drop] scene={scene_id} height={height} nadir_conf={nadir_conf:.3f} "
                    f"(< nadir_min_conf={args.nadir_min_conf})"
                )
                continue

            save_pair_images(
                sat_aligned=sat_aligned,
                drone_aligned=drone_aligned,
                out_sat_path=out_sat_dir / sat_file,
                out_drone_path=out_drone_dir / drone_file,
                jpeg_quality=args.jpeg_quality,
            )

            frame_idx = numeric_stem(best_frame)
            rows.append(
                PairRecord(
                    sample_id=sample_id,
                    scene_id=scene_id,
                    height=height,
                    drone_frame=frame_idx,
                    drone_rotation_ccw_deg=int(best_angle),
                    sat_crop_frac=float(round(best_frac, 4)),
                    match_score=float(round(best_score, 6)),
                    nadir_conf=float(round(nadir_conf, 6)),
                    nadir_residual=float(round(nadir_residual, 4)),
                    nadir_grad_norm=float(round(nadir_grad_norm, 6)),
                    nadir_valid_patches=int(nadir_valid),
                    is_nadir=int(is_nadir),
                    satellite_file=sat_file,
                    drone_file=drone_file,
                )
            )
            kept += 1

            print(
                f"[{done}/{total_targets}] {sample_id} "
                f"frame={frame_idx} rot={best_angle} frac={best_frac:.3f} score={best_score:.4f} "
                f"nadir={nadir_conf:.3f}"
            )

    csv_path = args.output_dir / "pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(PairRecord(
                sample_id="",
                scene_id="",
                height="",
                drone_frame=0,
                drone_rotation_ccw_deg=0,
                sat_crop_frac=1.0,
                match_score=0.0,
                nadir_conf=0.0,
                nadir_residual=0.0,
                nadir_grad_norm=0.0,
                nadir_valid_patches=0,
                is_nadir=0,
                satellite_file="",
                drone_file="",
            )).keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    by_height: Dict[str, int] = {}
    for h in heights:
        by_height[h] = sum(1 for r in rows if r.height == h)

    info = {
        "name": "SUES-200-512x512 pair export",
        "data_root": str(args.data_root),
        "total_targets": total_targets,
        "kept_pairs": kept,
        "heights": heights,
        "by_height": by_height,
        "output_size": args.output_size,
        "search": {
            "match_size": args.match_size,
            "coarse_angle_step": args.coarse_angle_step,
            "coarse_fracs": coarse_fracs,
            "top_k_frames": args.top_k_frames,
            "fine_angle_radius": args.fine_angle_radius,
            "fine_angle_step": args.fine_angle_step,
            "fine_frac_radius": args.fine_frac_radius,
            "fine_frac_step": args.fine_frac_step,
            "min_crop_frac": args.min_crop_frac,
            "frame_stride": args.frame_stride,
            "min_score": args.min_score,
            "nadir_min_conf": args.nadir_min_conf,
            "nadir_eval_size": args.nadir_eval_size,
            "nadir_grid": args.nadir_grid,
            "nadir_patch_size": args.nadir_patch_size,
            "nadir_peak_thresh": args.nadir_peak_thresh,
            "nadir_residual_weight": args.nadir_residual_weight,
            "nadir_grad_weight": args.nadir_grad_weight,
        },
        "pair_fields": [
            "sample_id",
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
            "satellite_file",
            "drone_file",
        ],
        "orientation_note": "drone_rotation_ccw_deg rotates drone image CCW to align with satellite orientation",
        "range_note": "sat_crop_frac is center-crop ratio applied on satellite before resize",
    }

    info_path = args.output_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {info_path}")
    print(f"Done. kept={kept}/{total_targets}")


if __name__ == "__main__":
    main()

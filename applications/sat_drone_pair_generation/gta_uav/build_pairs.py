#!/usr/bin/env python3
"""Build GTA-UAV near-nadir drone-satellite pair dataset.

Pipeline
--------
  1. Scan GTA-UAV metadata json files
  2. Filter near-nadir drone images and pick the best positive satellite tile
  3. Post-check headings: compare 4 candidate rotations via NCC
     - camera yaw is replaced by the NCC winner for flagged samples
  4. Filter: keep only high-quality pairs (default: heading confirmed)
  5. Export processed 384x384 images + dataset_info.json + pairs.csv

Usage
-----
  python3 build_pairs.py \\
      --data-root   /path/to/GTA-UAV-LR \\
      --output-dir  /path/to/output/gta_pairs

Output layout
-------------
  <output-dir>/
  ├── dataset_info.json     dataset-level statistics and specs
  ├── pairs.csv             per-sample index and metadata
  ├── drone/
  │   └── {sample_id}.jpg   384x384 px, north-up aligned drone crop
  └── satellite/
      └── {sample_id}.jpg   384x384 px, matched north-up satellite crop
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


DEFAULT_META_FILES = [
    "cross-area-drone2sate-train.json",
    "cross-area-drone2sate-test.json",
    "same-area-drone2sate-train.json",
    "same-area-drone2sate-test.json",
]

OUT_SIZE = 384
JPEG_QUALITY = 95
DEFAULT_SAT_SCALE = 0.82
NCC_EDGE_SIZE = 96
NCC_FLAG_THRESH = 0.05


@dataclass
class Pair:
    sample_id: str
    source_meta_file: str
    split: str
    area_mode: str
    drone_img_name: str
    drone_img_path: Path
    satellite_img_name: str
    satellite_img_path: Path
    drone_loc_x: float
    drone_loc_y: float
    satellite_loc_x: Optional[float]
    satellite_loc_y: Optional[float]
    height: float
    cam_roll: float
    cam_pitch: float
    cam_yaw: float
    iou: float
    num_positive_tiles: int
    heading_status: str = "unchecked"
    north_up_rot: float = 0.0
    heading_ncc_best_rotation: Optional[float] = None
    heading_ncc_margin: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GTA-UAV near-nadir pair dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/GTA-UAV-LR"),
        help="Dataset root containing drone/, satellite/, and the 4 metadata json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the exported pair dataset.",
    )
    parser.add_argument(
        "--meta-files",
        nargs="*",
        default=DEFAULT_META_FILES,
        help="Metadata json files to scan, relative to --data-root.",
    )
    parser.add_argument(
        "--roll-threshold",
        type=float,
        default=5.0,
        help="Keep samples with abs(cam_roll + 90) <= threshold.",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=5.0,
        help="Keep samples with abs(cam_pitch) <= threshold.",
    )
    parser.add_argument(
        "--sat-scale",
        type=float,
        default=DEFAULT_SAT_SCALE,
        help="Satellite crop scale relative to the square drone footprint. Default 0.82.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for heading check and image export.",
    )
    parser.add_argument(
        "--skip-heading-check",
        action="store_true",
        help="Skip the 4-way NCC heading post-check.",
    )
    parser.add_argument(
        "--ncc-edge-size",
        type=int,
        default=NCC_EDGE_SIZE,
        help="Edge-map size used by the heading NCC post-check.",
    )
    parser.add_argument(
        "--ncc-flag-thresh",
        type=float,
        default=NCC_FLAG_THRESH,
        help="Minimum NCC margin required to auto-correct the base yaw.",
    )
    parser.add_argument(
        "--keep-statuses",
        nargs="*",
        default=None,
        help="Heading statuses to export. Defaults to confirmed, or unchecked when heading check is skipped.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke runs after pair discovery.",
    )
    parser.add_argument(
        "--pair-offset",
        type=int,
        default=0,
        help="Skip this many discovered pairs before applying --max-pairs. Useful for alternate smoke subsets.",
    )
    return parser.parse_args()


def sanitize_meta_name(meta_name: str) -> str:
    return meta_name.replace(".json", "").replace("-", "_")


def parse_meta_info(meta_name: str) -> Tuple[str, str]:
    split = "train" if "train" in meta_name else "test"
    area_mode = "cross_area" if meta_name.startswith("cross-") else "same_area"
    return split, area_mode


def build_sample_id(meta_name: str, drone_img_name: str) -> str:
    return f"{sanitize_meta_name(meta_name)}__{Path(drone_img_name).stem}"


def pick_best_positive(row: dict) -> Optional[dict]:
    sate_imgs = row.get("pair_pos_sate_img_list", [])
    weights = row.get("pair_pos_sate_weight_list", [])
    sate_locs = row.get("pair_pos_sate_loc_x_y_list", [])
    if not sate_imgs:
        return None

    best_idx = max(range(len(sate_imgs)), key=lambda idx: weights[idx])
    sate_loc = sate_locs[best_idx] if best_idx < len(sate_locs) else [None, None]
    return {
        "satellite_img_name": sate_imgs[best_idx],
        "iou": float(weights[best_idx]),
        "satellite_loc_x": sate_loc[0],
        "satellite_loc_y": sate_loc[1],
        "num_positive_tiles": len(sate_imgs),
    }


def iter_rows(data_root: Path, meta_files: Iterable[str]) -> Iterable[Tuple[str, dict]]:
    for meta_name in meta_files:
        meta_path = data_root / meta_name
        rows = json.loads(meta_path.read_text())
        for row in rows:
            yield meta_name, row


def footprint_size_world(height: float, hfov: float = 74, vfov: float = 59) -> Tuple[float, float]:
    width = 2 * height * math.tan(math.radians(hfov) / 2)
    length = 2 * height * math.tan(math.radians(vfov) / 2)
    return width, length


def parse_tile_name(tile_name: str) -> Tuple[int, int, int, int]:
    zoom, offset, tile_x, tile_y = tile_name.replace(".png", "").split("_")
    return int(zoom), int(offset), int(tile_x), int(tile_y)


def tile_span_world(zoom: int, sate_length: int = 24576) -> float:
    tile_pix = sate_length / (2 ** zoom)
    return tile_pix * 0.45


def stitch_satellite_crop(
    world_x: float,
    world_y: float,
    width_w: float,
    length_w: float,
    zoom: int,
    offset: int,
    sat_dir: Path,
) -> Image.Image:
    tsw = tile_span_world(zoom)

    x_min, x_max = world_x - width_w / 2, world_x + width_w / 2
    y_min, y_max = world_y - length_w / 2, world_y + length_w / 2

    col_f_min, col_f_max = x_min / tsw, x_max / tsw
    row_f_min, row_f_max = y_min / tsw, y_max / tsw

    tx_min, tx_max = int(math.floor(col_f_min)), int(math.floor(col_f_max))
    ty_min, ty_max = int(math.floor(row_f_min)), int(math.floor(row_f_max))

    composite = Image.new("RGB", ((tx_max - tx_min + 1) * 256, (ty_max - ty_min + 1) * 256), (0, 0, 0))
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tile_path = sat_dir / f"{zoom}_{offset}_{tx}_{ty}.png"
            if tile_path.exists():
                composite.paste(Image.open(tile_path).convert("RGB"), ((tx - tx_min) * 256, (ty - ty_min) * 256))

    crop_box = (
        max(0, int(round((col_f_min - tx_min) * 256))),
        max(0, int(round((row_f_min - ty_min) * 256))),
        min(composite.width, int(round((col_f_max - tx_min) * 256))),
        min(composite.height, int(round((row_f_max - ty_min) * 256))),
    )
    return composite.crop(crop_box)


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def mirror_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    fh = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fv = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    fhv = fh.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    canvas = Image.new(img.mode, (w * 3, h * 3))
    for row, imgs in enumerate([(fhv, fv, fhv), (fh, img, fh), (fhv, fv, fhv)]):
        for col, tile in enumerate(imgs):
            canvas.paste(tile, (col * w, row * h))
    return canvas


def rotate_no_border(img: Image.Image, deg: float) -> Image.Image:
    sq = center_crop_square(img)
    side = sq.size[0]
    padded = mirror_pad(sq)
    rotated = padded.rotate(deg, resample=Image.Resampling.BICUBIC)
    cx = padded.size[0] // 2
    half = side // 2
    return rotated.crop((cx - half, cx - half, cx + half, cx + half))


def edge_arr(img: Image.Image, size: int) -> np.ndarray:
    gray = img.convert("L").resize((size, size), Image.Resampling.BICUBIC)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    return np.array(ImageOps.autocontrast(edge), dtype=np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def make_north_up_pair(drone_img: Image.Image, sat_crop: Image.Image, rotation_deg: float, output_size: int) -> Tuple[Image.Image, Image.Image]:
    drone_sq = center_crop_square(drone_img)

    theta = math.radians(abs(rotation_deg) % 180)
    if theta > math.pi / 2:
        theta = math.pi - theta
    scale = math.cos(theta) + math.sin(theta)
    intermediate = int(math.ceil(output_size * scale))
    half = output_size // 2
    cx = intermediate // 2
    crop_box = (cx - half, cx - half, cx + half, cx + half)

    sat_sq = sat_crop.resize((intermediate, intermediate), Image.Resampling.LANCZOS)
    drone_sq = drone_sq.resize((intermediate, intermediate), Image.Resampling.LANCZOS)
    drone_rot = drone_sq.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, expand=False)
    return sat_sq.crop(crop_box), drone_rot.crop(crop_box)


def discover_pairs(args: argparse.Namespace) -> Tuple[List[Pair], Dict[str, int]]:
    drone_root = args.data_root / "drone" / "images"
    satellite_root = args.data_root / "satellite"

    pairs: List[Pair] = []
    stats = {
        "total_rows": 0,
        "nadir_rows": 0,
        "with_positive_tile": 0,
        "missing_drone_files": 0,
        "missing_satellite_files": 0,
    }

    for meta_name, row in iter_rows(args.data_root, args.meta_files):
        stats["total_rows"] += 1
        drone_meta = row["drone_metadata"]
        cam_roll = float(drone_meta["cam_roll"])
        cam_pitch = float(drone_meta["cam_pitch"])
        is_nadir = abs(cam_roll + 90.0) <= args.roll_threshold and abs(cam_pitch) <= args.pitch_threshold
        if not is_nadir:
            continue

        stats["nadir_rows"] += 1
        best_positive = pick_best_positive(row)
        if best_positive is None:
            continue
        stats["with_positive_tile"] += 1

        drone_path = drone_root / row["drone_img_name"]
        satellite_path = satellite_root / best_positive["satellite_img_name"]
        if not drone_path.exists():
            stats["missing_drone_files"] += 1
            continue
        if not satellite_path.exists():
            stats["missing_satellite_files"] += 1
            continue

        split, area_mode = parse_meta_info(meta_name)
        sample_id = build_sample_id(meta_name, row["drone_img_name"])
        pairs.append(
            Pair(
                sample_id=sample_id,
                source_meta_file=meta_name,
                split=split,
                area_mode=area_mode,
                drone_img_name=row["drone_img_name"],
                drone_img_path=drone_path,
                satellite_img_name=best_positive["satellite_img_name"],
                satellite_img_path=satellite_path,
                drone_loc_x=float(row["drone_loc_x_y"][0]),
                drone_loc_y=float(row["drone_loc_x_y"][1]),
                satellite_loc_x=best_positive["satellite_loc_x"],
                satellite_loc_y=best_positive["satellite_loc_y"],
                height=float(drone_meta["height"]),
                cam_roll=cam_roll,
                cam_pitch=cam_pitch,
                cam_yaw=float(drone_meta["cam_yaw"]),
                iou=float(best_positive["iou"]),
                num_positive_tiles=int(best_positive["num_positive_tiles"]),
            )
        )

    if args.pair_offset:
        pairs = pairs[args.pair_offset :]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    return pairs, stats


def heading_check_worker(args: tuple) -> dict:
    pair, sat_scale, edge_size, flag_thresh = args
    try:
        drone_img = Image.open(pair.drone_img_path).convert("RGB")
        zoom, offset, _, _ = parse_tile_name(pair.satellite_img_name)
        _, fl = footprint_size_world(pair.height)
        sat_crop = stitch_satellite_crop(
            pair.drone_loc_x,
            pair.drone_loc_y,
            fl * sat_scale,
            fl * sat_scale,
            zoom,
            offset,
            pair.satellite_img_path.parent,
        )
        sat_edge = edge_arr(sat_crop, edge_size)
        base_rot = pair.cam_yaw
        candidates = [base_rot + i * 90 for i in range(4)]
        scores = {
            rot: ncc(edge_arr(rotate_no_border(drone_img, rot), edge_size), sat_edge)
            for rot in candidates
        }
        best_rot = max(scores, key=scores.__getitem__)
        margin = scores[best_rot] - scores[base_rot]

        if best_rot == base_rot:
            status = "confirmed"
            north_up_rot = base_rot
        elif margin >= flag_thresh:
            status = "corrected"
            north_up_rot = best_rot
        else:
            status = "ambiguous"
            north_up_rot = base_rot

        return {
            "sample_id": pair.sample_id,
            "heading_status": status,
            "north_up_rot": north_up_rot,
            "heading_ncc_best_rotation": best_rot,
            "heading_ncc_margin": margin,
        }
    except Exception as exc:
        return {
            "sample_id": pair.sample_id,
            "heading_status": "error",
            "north_up_rot": pair.cam_yaw,
            "heading_ncc_best_rotation": pair.cam_yaw,
            "heading_ncc_margin": None,
            "heading_error": str(exc),
        }


def run_heading_check(
    pairs: List[Pair],
    sat_scale: float,
    workers: int,
    edge_size: int,
    flag_thresh: float,
) -> Dict[str, int]:
    task_args = [(pair, sat_scale, edge_size, flag_thresh) for pair in pairs]
    counts: Dict[str, int] = {}
    by_id: Dict[str, dict] = {}
    done = 0
    with Pool(workers) as pool:
        for res in pool.imap_unordered(heading_check_worker, task_args, chunksize=16):
            by_id[res["sample_id"]] = res
            done += 1
            if done % 500 == 0 or done == len(pairs):
                print(f"  heading-check [{100 * done / len(pairs):5.1f}%] {done}/{len(pairs)}")

    for pair in pairs:
        res = by_id[pair.sample_id]
        pair.heading_status = res["heading_status"]
        pair.north_up_rot = float(res["north_up_rot"])
        pair.heading_ncc_best_rotation = float(res["heading_ncc_best_rotation"])
        pair.heading_ncc_margin = res["heading_ncc_margin"]
        counts[pair.heading_status] = counts.get(pair.heading_status, 0) + 1
    return counts


def filter_pairs(pairs: List[Pair], keep_statuses: List[str]) -> List[Pair]:
    keep = set(keep_statuses)
    return [pair for pair in pairs if pair.heading_status in keep]


def export_worker(args: tuple) -> str:
    pair, sat_scale, out_dir_s = args
    out_dir = Path(out_dir_s)
    try:
        drone_img = Image.open(pair.drone_img_path).convert("RGB")
        zoom, offset, _, _ = parse_tile_name(pair.satellite_img_name)
        _, fl = footprint_size_world(pair.height)
        sat_crop = stitch_satellite_crop(
            pair.drone_loc_x,
            pair.drone_loc_y,
            fl * sat_scale,
            fl * sat_scale,
            zoom,
            offset,
            pair.satellite_img_path.parent,
        )
        sat_out, drone_out = make_north_up_pair(drone_img, sat_crop, pair.north_up_rot, OUT_SIZE)
        drone_dst = out_dir / "drone" / f"{pair.sample_id}.jpg"
        sat_dst = out_dir / "satellite" / f"{pair.sample_id}.jpg"
        drone_out.save(drone_dst, "JPEG", quality=JPEG_QUALITY, subsampling=0)
        sat_out.save(sat_dst, "JPEG", quality=JPEG_QUALITY, subsampling=0)
        return ""
    except Exception as exc:
        return f"ERROR {pair.sample_id}: {exc}"


def export_images(pairs: List[Pair], out_dir: Path, sat_scale: float, workers: int) -> None:
    (out_dir / "drone").mkdir(parents=True, exist_ok=True)
    (out_dir / "satellite").mkdir(parents=True, exist_ok=True)
    task_args = [(pair, sat_scale, str(out_dir)) for pair in pairs]
    done = 0
    errors = []
    with Pool(workers) as pool:
        for err in pool.imap_unordered(export_worker, task_args, chunksize=16):
            done += 1
            if err:
                errors.append(err)
            if done % 500 == 0 or done == len(task_args):
                print(f"  export [{100 * done / len(task_args):5.1f}%] {done}/{len(task_args)}  errors={len(errors)}")
    if errors:
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)


def write_pairs_csv(pairs: List[Pair], out_dir: Path) -> None:
    fieldnames = [
        "sample_id",
        "split",
        "area_mode",
        "source_meta_file",
        "drone_img_name",
        "satellite_img_name",
        "drone_img_path",
        "satellite_img_path",
        "export_drone_path",
        "export_satellite_path",
        "drone_loc_x",
        "drone_loc_y",
        "satellite_loc_x",
        "satellite_loc_y",
        "height",
        "cam_roll",
        "cam_pitch",
        "cam_yaw",
        "iou",
        "num_positive_tiles",
        "heading_status",
        "north_up_rot",
        "heading_ncc_best_rotation",
        "heading_ncc_margin",
    ]
    with (out_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sample_id": pair.sample_id,
                    "split": pair.split,
                    "area_mode": pair.area_mode,
                    "source_meta_file": pair.source_meta_file,
                    "drone_img_name": pair.drone_img_name,
                    "satellite_img_name": pair.satellite_img_name,
                    "drone_img_path": str(pair.drone_img_path),
                    "satellite_img_path": str(pair.satellite_img_path),
                    "export_drone_path": f"drone/{pair.sample_id}.jpg",
                    "export_satellite_path": f"satellite/{pair.sample_id}.jpg",
                    "drone_loc_x": f"{pair.drone_loc_x:.6f}",
                    "drone_loc_y": f"{pair.drone_loc_y:.6f}",
                    "satellite_loc_x": "" if pair.satellite_loc_x is None else f"{pair.satellite_loc_x:.6f}",
                    "satellite_loc_y": "" if pair.satellite_loc_y is None else f"{pair.satellite_loc_y:.6f}",
                    "height": f"{pair.height:.6f}",
                    "cam_roll": f"{pair.cam_roll:.6f}",
                    "cam_pitch": f"{pair.cam_pitch:.6f}",
                    "cam_yaw": f"{pair.cam_yaw:.6f}",
                    "iou": f"{pair.iou:.6f}",
                    "num_positive_tiles": pair.num_positive_tiles,
                    "heading_status": pair.heading_status,
                    "north_up_rot": f"{pair.north_up_rot:.6f}",
                    "heading_ncc_best_rotation": "" if pair.heading_ncc_best_rotation is None else f"{pair.heading_ncc_best_rotation:.6f}",
                    "heading_ncc_margin": "" if pair.heading_ncc_margin is None else f"{pair.heading_ncc_margin:.6f}",
                }
            )


def write_dataset_info(
    pairs_all: List[Pair],
    pairs_kept: List[Pair],
    out_dir: Path,
    args: argparse.Namespace,
    discover_stats: Dict[str, int],
    heading_counts: Dict[str, int],
) -> None:
    split_counts = {split: sum(1 for pair in pairs_kept if pair.split == split) for split in ["train", "test"]}
    area_counts = {mode: sum(1 for pair in pairs_kept if pair.area_mode == mode) for mode in ["cross_area", "same_area"]}
    avg_height = sum(pair.height for pair in pairs_kept) / len(pairs_kept) if pairs_kept else 0.0
    avg_iou = sum(pair.iou for pair in pairs_kept) / len(pairs_kept) if pairs_kept else 0.0
    info = {
        "name": "GTA-UAV near-nadir pairs",
        "total_discovered": len(pairs_all),
        "total_exported": len(pairs_kept),
        "splits": split_counts,
        "area_modes": area_counts,
        "image_size_px": [OUT_SIZE, OUT_SIZE],
        "image_format": "JPEG",
        "image_quality": JPEG_QUALITY,
        "satellite_crop_scale": args.sat_scale,
        "near_nadir_thresholds": {
            "roll": args.roll_threshold,
            "pitch": args.pitch_threshold,
        },
        "heading_check": {
            "enabled": not args.skip_heading_check,
            "ncc_edge_size": None if args.skip_heading_check else args.ncc_edge_size,
            "ncc_flag_thresh": None if args.skip_heading_check else args.ncc_flag_thresh,
            "status_counts": heading_counts,
            "exported_statuses": args.keep_statuses,
        },
        "stats": discover_stats,
        "pair_offset": args.pair_offset,
        "average_height": round(avg_height, 3),
        "average_iou": round(avg_iou, 4),
        "north_up_rot_field": "north_up_rot",
        "note": (
            "Exported drone and satellite images are already north-up aligned and resized to 384x384. "
            "north_up_rot records the CCW rotation applied to the drone crop during export."
        ),
    }
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.data_root.exists():
        sys.exit(f"data-root not found: {args.data_root}")

    if args.keep_statuses is None:
        args.keep_statuses = ["unchecked"] if args.skip_heading_check else ["confirmed"]

    print("Stage 1/5  Discovering near-nadir pairs ...")
    pairs_all, discover_stats = discover_pairs(args)
    print(
        f"  discovered={len(pairs_all)}  "
        f"nadir_rows={discover_stats['nadir_rows']}  "
        f"with_positive_tile={discover_stats['with_positive_tile']}"
    )

    if not pairs_all:
        sys.exit("No pairs discovered.")

    if args.skip_heading_check:
        print("Stage 2/5  Heading post-check skipped (--skip-heading-check)")
        for pair in pairs_all:
            pair.heading_status = "unchecked"
            pair.north_up_rot = pair.cam_yaw
            pair.heading_ncc_best_rotation = pair.cam_yaw
            pair.heading_ncc_margin = None
        heading_counts = {"unchecked": len(pairs_all)}
    else:
        print(f"Stage 2/5  Heading post-check ({args.workers} workers) ...")
        heading_counts = run_heading_check(
            pairs_all, args.sat_scale, args.workers, args.ncc_edge_size, args.ncc_flag_thresh
        )
        print(f"  status_counts={heading_counts}")

    print("Stage 3/5  Filtering exported pairs ...")
    pairs_kept = filter_pairs(pairs_all, args.keep_statuses)
    print(f"  kept {len(pairs_kept)}/{len(pairs_all)} pairs  statuses={args.keep_statuses}")
    if not pairs_kept:
        sys.exit("No pairs left after filtering.")

    print(f"Stage 4/5  Exporting images -> {args.output_dir}")
    export_images(pairs_kept, args.output_dir, args.sat_scale, args.workers)

    print("Stage 5/5  Writing metadata files ...")
    write_pairs_csv(pairs_kept, args.output_dir)
    write_dataset_info(pairs_all, pairs_kept, args.output_dir, args, discover_stats, heading_counts)

    used_gb = sum(f.stat().st_size for f in args.output_dir.rglob("*.jpg")) / 1e9
    print()
    print("=" * 60)
    print(f"  Output        : {args.output_dir}")
    print(f"  Exported      : {len(pairs_kept)} pairs")
    print(f"  Disk used     : {used_gb:.2f} GB")
    print(f"  dataset_info  : {args.output_dir / 'dataset_info.json'}")
    print(f"  pairs index   : {args.output_dir / 'pairs.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

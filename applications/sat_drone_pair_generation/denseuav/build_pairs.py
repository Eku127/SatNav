#!/usr/bin/env python3
"""Build DenseUAV drone-satellite pair dataset.

Pipeline
--------
  1. Parse GPS metadata from Dense_GPS_*.txt
  2. Estimate flight heading per sample from GPS trajectory
  3. Match drone (H90.JPG) ↔ satellite (H90.tif) by sample ID
  4. Post-check headings: compare 4 candidate rotations via NCC
     – GPS rotation is replaced by the NCC winner for flagged samples
  5. Filter: keep only high-quality pairs (GPS conf ≥ 0.8, NCC ok/fixed)
  6. Export processed images + dataset_info.json + pairs.csv

Usage
-----
  python3 build_pairs.py \\
      --dataset-root /path/to/DenseUAV/DenseUAV \\
      --output-dir   /path/to/output/denseuav

Output layout
-------------
  <output-dir>/
  ├── dataset_info.json     dataset-level statistics and specs
  ├── pairs.csv             per-sample index (sample_id, split, lon, lat, north_up_rot)
  ├── drone/
  │   └── {sample_id}.jpg   512×512 px, center-cropped square
  └── satellite/
      └── {sample_id}.jpg   512×512 px, scale-corrected to match drone coverage

dataset_info.json fields
------------------------
  name, total, splits {train, test}
  image_size_px, image_format, image_quality
  drone_gsd_cm_per_px, satellite_gsd_cm_per_px
  ground_coverage_m    (approx. ground side length visible in both images)
  north_up_rot_field   (column in pairs.csv to apply for north-up alignment)

pairs.csv columns
-----------------
  sample_id    6-digit zero-padded identifier
  split        train | test
  lon / lat    WGS-84 frame centre
  north_up_rot PIL CCW rotation (°) to produce a north-up aligned image
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OUT_SIZE        = 512   # output pixel size (both drone and satellite)
JPEG_QUALITY    = 95

GPS_CONF_MIN    = 0.8   # minimum heading confidence for "usable"
NCC_EDGE_SIZE   = 96    # edge map resolution for NCC comparison
NCC_FLAG_THRESH = 0.03  # NCC margin to flag a GPS heading as wrong

# Per-altitude config.
# drone_cov  : ground coverage of the 1080×1080 center crop, relative to GPS range_m
#              = altitude_m / 80  (H80 is the GPS reference altitude)
# sat_cov    : satellite tif ground coverage relative to GPS range_m
#              = CROP_SIZE_PX / 640  (H80 crop = 640 px at full resolution)
# sat_crop_frac : fraction to center-crop the 512×512 satellite so its visible
#                 ground footprint matches the drone center crop
#                 = drone_cov / sat_cov
ALTITUDE_CONFIG = {
    "H80":  {"drone": "H80.JPG",  "sat": "H80.tif",  "sat_crop_frac": 1.000000},  # 1.0/1.0
    "H90":  {"drone": "H90.JPG",  "sat": "H90.tif",  "sat_crop_frac": 0.937500},  # 1.125/1.2
    "H100": {"drone": "H100.JPG", "sat": "H100.tif", "sat_crop_frac": 0.892857},  # 1.25/1.4
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – GPS parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPSInfo:
    lon: float
    lat: float
    range_m: float


def load_gps_map(gps_file: Path) -> Dict[str, GPSInfo]:
    """Parse Dense_GPS_*.txt into {sample_id: GPSInfo}."""
    mapping: Dict[str, GPSInfo] = {}
    for raw in gps_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 4:
            continue
        rel, lon_tok, lat_tok, range_tok = parts
        if not (rel.endswith("/H80.tif") or rel.endswith("/H80_old.tif")):
            continue
        elems = rel.split("/")
        if len(elems) < 4:
            continue
        sample_id = elems[2]
        try:
            lon     = float(lon_tok[1:]) if lon_tok.startswith("E") else float(lon_tok)
            lat     = float(lat_tok[1:]) if lat_tok.startswith("N") else float(lat_tok)
            range_m = float(range_tok)
        except ValueError:
            continue
        if sample_id not in mapping or rel.endswith("/H80.tif"):
            mapping[sample_id] = GPSInfo(lon=lon, lat=lat, range_m=range_m)
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – GPS heading estimation
# ─────────────────────────────────────────────────────────────────────────────

def _dist_m(lo1: float, la1: float, lo2: float, la2: float) -> float:
    dlat = (la2 - la1) * 111320.0
    dlon = (lo2 - lo1) * 111320.0 * math.cos(math.radians((la1 + la2) / 2.0))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _step_bearing(lo1: float, la1: float, lo2: float, la2: float) -> Optional[float]:
    """Bearing from point 1 to point 2 (degrees CW from N). None if step < 0.3 m."""
    dl = (la2 - la1) * 111320.0
    do = (lo2 - lo1) * 111320.0 * math.cos(math.radians((la1 + la2) / 2.0))
    if (dl * dl + do * do) ** 0.5 < 0.3:
        return None
    return math.degrees(math.atan2(do, dl)) % 360.0


def _angular_diff(a: float, b: float) -> float:
    """Signed angular difference a − b, wrapped to (−180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _build_window(
    entries: List[Tuple[str, float, float]],
    idx: int,
    ref: float,
    half_window: int,
    jump_m: float,
    bear_thresh: float,
) -> List[Tuple[float, float]]:
    """Grow a GPS window around entries[idx] constrained to bearing ≈ ref."""
    n = len(entries)
    _, lo0, la0 = entries[idx]
    win: List[Tuple[float, float]] = [(lo0, la0)]

    prev_lo, prev_la = lo0, la0
    for j in range(idx - 1, max(-1, idx - half_window - 1), -1):
        _, lo, la = entries[j]
        if _dist_m(lo, la, prev_lo, prev_la) >= jump_m:
            break
        b = _step_bearing(lo, la, prev_lo, prev_la)
        if b is not None and abs(_angular_diff(b, ref)) > bear_thresh:
            break
        win.insert(0, (lo, la))
        prev_lo, prev_la = lo, la

    prev_lo, prev_la = lo0, la0
    for j in range(idx + 1, min(n, idx + half_window + 1)):
        _, lo, la = entries[j]
        if _dist_m(prev_lo, prev_la, lo, la) >= jump_m:
            break
        b = _step_bearing(prev_lo, prev_la, lo, la)
        if b is not None and abs(_angular_diff(b, ref)) > bear_thresh:
            break
        win.append((lo, la))
        prev_lo, prev_la = lo, la

    return win


def compute_gps_headings(
    gps_map: Dict[str, GPSInfo],
    half_window: int = 6,
    jump_m: float = 35.0,
    bear_thresh: float = 50.0,
) -> Dict[str, Tuple[float, float]]:
    """Return {sample_id: (heading_deg, confidence)} from GPS trajectory.

    Strategy: for each sample, collect candidate headings from neighbouring
    steps, build a bearing-constrained window for each, keep the longest
    (= the flight-leg direction).  Turn steps produce short windows and are
    thus not mistaken for the main heading.
    """
    entries = sorted(
        ((sid, g.lon, g.lat) for sid, g in gps_map.items()),
        key=lambda x: x[0],
    )
    n = len(entries)
    results: Dict[str, Tuple[float, float]] = {}

    for idx, (sid, lo0, la0) in enumerate(entries):
        # Collect unique candidate reference headings from ±1, ±2 neighbours
        refs: List[float] = []
        for step in [-1, 1, -2, 2]:
            j = idx + step
            if not (0 <= j < n):
                continue
            _, lo_j, la_j = entries[j]
            b = (_step_bearing(lo_j, la_j, lo0, la0) if step < 0
                 else _step_bearing(lo0, la0, lo_j, la_j))
            if b is None:
                continue
            if _dist_m(lo0, la0, lo_j, la_j) >= jump_m * abs(step):
                continue
            if not any(abs(_angular_diff(b, r)) < 20.0 for r in refs):
                refs.append(b)

        if not refs:
            results[sid] = (0.0, 0.0)
            continue

        # Build window for each ref; keep the longest
        best: List[Tuple[float, float]] = [(lo0, la0)]
        for ref in refs:
            w = _build_window(entries, idx, ref, half_window, jump_m, bear_thresh)
            if len(w) > len(best):
                best = w

        if len(best) < 2:
            best = _build_window(entries, idx, refs[0], half_window, jump_m, bear_thresh)
            if len(best) < 2:
                results[sid] = (round(refs[0], 1), 0.3)
                continue

        # Heading = first→last of best window
        lons = [p[0] for p in best]
        lats = [p[1] for p in best]
        mean_lat = sum(lats) / len(lats)
        dlat = (lats[-1] - lats[0]) * 111320.0
        dlon = (lons[-1] - lons[0]) * 111320.0 * math.cos(math.radians(mean_lat))
        heading = math.degrees(math.atan2(dlon, dlat)) % 360.0

        # Confidence = 1 − RMS(step bearing deviation) / 5°
        step_bearings = [
            b for b in (
                _step_bearing(lons[i - 1], lats[i - 1], lons[i], lats[i])
                for i in range(1, len(best))
            )
            if b is not None
        ]
        if step_bearings:
            diffs = [_angular_diff(b, heading) for b in step_bearings]
            rms = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
            conf = max(0.0, min(1.0, 1.0 - rms / 5.0))
        else:
            conf = 0.5

        results[sid] = (round(heading, 1), round(conf, 3))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Pair discovery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pair:
    sample_id: str        # "{original_id}_{altitude}", e.g. "000000_H90"
    split: str
    altitude: str         # "H80" | "H90" | "H100"
    sat_crop_frac: float  # per-altitude satellite scale-correction fraction
    drone_path: Path
    satellite_path: Path
    lon: float
    lat: float
    range_m: float
    gps_heading_deg: float
    gps_heading_conf: float
    pil_rotation_deg: int   # updated to final_pil_rot after NCC stage
    ncc_confirmed: int = 0
    ncc_flagged: int = 0


def discover_pairs(
    dataset_root: Path,
    gps_map: Dict[str, GPSInfo],
    heading_map: Dict[str, Tuple[float, float]],
    split: str,
) -> List[Pair]:
    if split == "train":
        drone_base = dataset_root / "train" / "drone"
        sat_base   = dataset_root / "train" / "satellite"
    else:
        drone_base = dataset_root / "test" / "query_drone"
        sat_base   = dataset_root / "test" / "gallery_satellite"

    pairs: List[Pair] = []
    for sample_dir in sorted(drone_base.iterdir()):
        if not sample_dir.is_dir():
            continue
        orig_id = sample_dir.name
        if orig_id not in gps_map:
            continue
        gps = gps_map[orig_id]
        heading, conf = heading_map.get(orig_id, (0.0, 0.0))
        pil_rot = int((360.0 - heading) % 360)

        for alt, cfg in ALTITUDE_CONFIG.items():
            drone_path = sample_dir / cfg["drone"]
            sat_path   = sat_base / orig_id / cfg["sat"]
            if not drone_path.exists() or not sat_path.exists():
                continue
            pairs.append(Pair(
                sample_id        = f"{orig_id}_{alt}",
                split            = split,
                altitude         = alt,
                sat_crop_frac    = cfg["sat_crop_frac"],
                drone_path       = drone_path,
                satellite_path   = sat_path,
                lon              = gps.lon,
                lat              = gps.lat,
                range_m          = gps.range_m,
                gps_heading_deg  = heading,
                gps_heading_conf = conf,
                pil_rotation_deg = pil_rot,
            ))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – NCC post-check
# ─────────────────────────────────────────────────────────────────────────────

def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def _mirror_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    fh  = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fv  = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    fhv = fh.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    c   = Image.new(img.mode, (w * 3, h * 3))
    for row, imgs in enumerate([
        (fhv, fv, fhv),
        (fh,  img, fh),
        (fhv, fv, fhv),
    ]):
        for col, tile in enumerate(imgs):
            c.paste(tile, (col * w, row * h))
    return c


def _rotate_no_border(img: Image.Image, deg: int) -> Image.Image:
    sq    = _center_crop_square(img)
    side  = sq.size[0]
    pad   = _mirror_pad(sq)
    rot   = pad.rotate(deg, resample=Image.Resampling.BICUBIC)
    cx    = pad.size[0] // 2
    half  = side // 2
    return rot.crop((cx - half, cx - half, cx + half, cx + half))


def _sat_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    sc = round(min(w, h) * frac)
    return img.crop(((w - sc) // 2, (h - sc) // 2, (w + sc) // 2, (h + sc) // 2))


def _edge_arr(img: Image.Image, size: int) -> np.ndarray:
    gray = img.convert("L").resize((size, size), Image.Resampling.BICUBIC)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    arr  = np.array(ImageOps.autocontrast(edge), dtype=np.float32)
    return arr


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def _ncc_check_worker(args: tuple) -> dict:
    """Multiprocessing worker: run 4-rotation NCC for one pair."""
    pil_rot, drone_path_s, sat_path_s, sat_crop_frac, edge_size, flag_thresh = args
    drone    = Image.open(drone_path_s).convert("RGB")
    sat      = Image.open(sat_path_s).convert("RGB")
    sat_edge = _edge_arr(_sat_crop(sat, sat_crop_frac), edge_size)

    candidates = [(pil_rot + i * 90) % 360 for i in range(4)]
    scores = {
        rot: _ncc(_edge_arr(_rotate_no_border(drone, rot), edge_size), sat_edge)
        for rot in candidates
    }
    best_rot   = max(scores, key=scores.__getitem__)
    sorted_s   = sorted(scores.values(), reverse=True)
    confirmed  = best_rot == pil_rot
    margin     = scores[best_rot] - scores[pil_rot]
    flagged    = (not confirmed) and (margin >= flag_thresh)

    return {
        "best_rot":  best_rot,
        "confirmed": int(confirmed),
        "flagged":   int(flagged),
    }


def run_ncc_check(pairs: List[Pair], workers: int) -> None:
    """In-place update pairs: set ncc_confirmed, ncc_flagged, pil_rotation_deg."""
    task_args = [
        (p.pil_rotation_deg, str(p.drone_path), str(p.satellite_path),
         p.sat_crop_frac, NCC_EDGE_SIZE, NCC_FLAG_THRESH)
        for p in pairs
    ]
    with Pool(workers) as pool:
        results = pool.map(_ncc_check_worker, task_args, chunksize=32)

    for pair, res in zip(pairs, results):
        pair.ncc_confirmed    = res["confirmed"]
        pair.ncc_flagged      = res["flagged"]
        if res["flagged"]:
            pair.pil_rotation_deg = res["best_rot"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – Quality filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_usable(pairs: List[Pair]) -> List[Pair]:
    """Keep pairs where GPS conf ≥ GPS_CONF_MIN AND (NCC confirmed OR flagged+fixed)."""
    return [
        p for p in pairs
        if p.gps_heading_conf >= GPS_CONF_MIN
        and (p.ncc_confirmed or p.ncc_flagged)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Image export
# ─────────────────────────────────────────────────────────────────────────────

def _export_worker(args: tuple) -> str:
    drone_src_s, sat_src_s, sat_crop_frac, drone_dst_s, sat_dst_s = args
    try:
        # Drone: center-crop square → resize
        img = Image.open(drone_src_s).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if img.size[0] != OUT_SIZE:
            img = img.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
        img.save(drone_dst_s, "JPEG", quality=JPEG_QUALITY, subsampling=0)

        # Satellite: scale-correct → resize
        img = Image.open(sat_src_s).convert("RGB")
        w, h = img.size
        sc = round(min(w, h) * sat_crop_frac)
        img = img.crop(((w - sc) // 2, (h - sc) // 2, (w + sc) // 2, (h + sc) // 2))
        if img.size[0] != OUT_SIZE:
            img = img.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
        img.save(sat_dst_s, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    except Exception as exc:
        return f"ERROR {drone_src_s}: {exc}"
    return ""


def export_images(pairs: List[Pair], out_dir: Path, workers: int) -> None:
    (out_dir / "drone").mkdir(parents=True, exist_ok=True)
    (out_dir / "satellite").mkdir(parents=True, exist_ok=True)

    task_args = [
        (str(p.drone_path), str(p.satellite_path), p.sat_crop_frac,
         str(out_dir / "drone"     / f"{p.sample_id}.jpg"),
         str(out_dir / "satellite" / f"{p.sample_id}.jpg"))
        for p in pairs
    ]
    errors = []
    done = 0
    n = len(task_args)
    with Pool(workers) as pool:
        for err in pool.imap_unordered(_export_worker, task_args, chunksize=32):
            done += 1
            if err:
                errors.append(err)
            if done % 500 == 0 or done == n:
                print(f"  [{100*done/n:5.1f}%] {done}/{n}  errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 – dataset_info.json + pairs.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(pairs: List[Pair], out_dir: Path) -> None:
    n_train = sum(1 for p in pairs if p.split == "train")
    n_test  = sum(1 for p in pairs if p.split == "test")

    # Per-altitude counts and GSD
    avg_range = sum(p.range_m for p in pairs) / len(pairs)
    alt_counts = {}
    alt_gsd    = {}
    for alt, cfg in ALTITUDE_CONFIG.items():
        alt_m = int(alt[1:])                              # 80 / 90 / 100
        drone_cov = avg_range * alt_m / 80               # ground coverage (m)
        alt_counts[alt] = sum(1 for p in pairs if p.altitude == alt)
        alt_gsd[alt] = {
            "drone_gsd_cm_per_px": round(drone_cov / OUT_SIZE * 100, 1),
            "satellite_gsd_cm_per_px": round((avg_range * (int(alt[1:])/80*1.2)) / OUT_SIZE * 100 / cfg["sat_crop_frac"], 1),
            "ground_coverage_m": round(drone_cov),
        }

    # dataset_info.json
    info = {
        "name":           "DenseUAV",
        "total":          len(pairs),
        "splits":         {"train": n_train, "test": n_test},
        "by_altitude":    alt_counts,
        "image_size_px":  [OUT_SIZE, OUT_SIZE],
        "image_format":   "JPEG",
        "image_quality":  JPEG_QUALITY,
        "altitudes":      alt_gsd,
        "north_up_rot_field": "north_up_rot",
        "note": (
            "sample_id format: {original_id}_{altitude}, e.g. 000000_H90. "
            "Apply north_up_rot (PIL CCW degrees) to either image for north-up view."
        ),
    }
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # pairs.csv – minimal per-sample index
    with (out_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "altitude", "split", "lon", "lat", "north_up_rot"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({
                "sample_id":    p.sample_id,
                "altitude":     p.altitude,
                "split":        p.split,
                "lon":          f"{p.lon:.8f}",
                "lat":          f"{p.lat:.8f}",
                "north_up_rot": p.pil_rotation_deg,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DenseUAV drone-satellite pair dataset."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Path to the raw DenseUAV dataset root "
             "(the folder containing train/, test/, Dense_GPS_*.txt)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory (will be created if absent)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel workers for NCC check and image export (default: 16)",
    )
    parser.add_argument(
        "--skip-ncc", action="store_true",
        help="Skip NCC post-check (faster but no heading verification)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir      = args.output_dir.resolve()

    if not dataset_root.exists():
        sys.exit(f"dataset-root not found: {dataset_root}")

    # ── Stage 1: GPS ─────────────────────────────────────────────────────────
    print("Stage 1/5  Loading GPS metadata …")
    gps_train = load_gps_map(dataset_root / "Dense_GPS_train.txt")
    gps_test  = load_gps_map(dataset_root / "Dense_GPS_test.txt")
    print(f"  train={len(gps_train)}  test={len(gps_test)}")

    # ── Stage 2: Headings ────────────────────────────────────────────────────
    print("Stage 2/5  Computing GPS headings …")
    head_train = compute_gps_headings(gps_train)
    head_test  = compute_gps_headings(gps_test)

    # ── Stage 3: Pair discovery ──────────────────────────────────────────────
    print("Stage 3/5  Discovering pairs …")
    pairs = (
        discover_pairs(dataset_root, gps_train, head_train, "train") +
        discover_pairs(dataset_root, gps_test,  head_test,  "test")
    )
    print(f"  found {len(pairs)} pairs  "
          f"(train={sum(1 for p in pairs if p.split=='train')}, "
          f"test={sum(1 for p in pairs if p.split=='test')})")

    # ── Stage 4: NCC post-check ──────────────────────────────────────────────
    if args.skip_ncc:
        print("Stage 4/5  NCC post-check skipped (--skip-ncc)")
        for p in pairs:
            p.ncc_confirmed = 1   # trust GPS when skipping NCC
    else:
        print(f"Stage 4/5  NCC heading post-check ({args.workers} workers) …")
        run_ncc_check(pairs, args.workers)
        n_confirmed = sum(p.ncc_confirmed for p in pairs)
        n_flagged   = sum(p.ncc_flagged   for p in pairs)
        print(f"  confirmed={n_confirmed}  flagged+corrected={n_flagged}  "
              f"soft-diff={len(pairs)-n_confirmed-n_flagged}")

    # ── Stage 5: Filter ──────────────────────────────────────────────────────
    print("Stage 5/5  Filtering usable pairs …")
    usable = filter_usable(pairs)
    print(f"  kept {len(usable)}/{len(pairs)} pairs  "
          f"(GPS conf≥{GPS_CONF_MIN}, NCC confirmed or corrected)")

    # ── Stage 6: Export images ───────────────────────────────────────────────
    print(f"\nExporting images → {out_dir}  ({args.workers} workers)")
    print(f"  output size: {OUT_SIZE}×{OUT_SIZE} px  JPEG q={JPEG_QUALITY}")
    export_images(usable, out_dir, args.workers)

    # ── Stage 7: dataset_info.json + pairs.csv ───────────────────────────────
    write_outputs(usable, out_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_train = sum(1 for p in usable if p.split == "train")
    n_test  = sum(1 for p in usable if p.split == "test")
    used_gb = sum(f.stat().st_size for f in out_dir.rglob("*.jpg")) / 1e9

    print()
    print("=" * 60)
    print(f"  Output        : {out_dir}")
    print(f"  Pairs         : {len(usable)}  (train={n_train}, test={n_test})")
    print(f"  Disk used     : {used_gb:.2f} GB")
    print(f"  dataset_info  : {out_dir / 'dataset_info.json'}")
    print(f"  pairs index   : {out_dir / 'pairs.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

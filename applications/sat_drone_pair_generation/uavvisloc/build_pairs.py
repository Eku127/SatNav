#!/usr/bin/env python3
"""Build approximate north-up drone-satellite pairs from UAV-VisLoc.

This exporter uses the metadata available in UAV-VisLoc:
- per-image lat/lon/height
- optional pose fields Omega/Kappa/Phi1/Phi2
- per-sequence satellite raster corner coordinates

Unlike DenseUAV, UAV-VisLoc does not provide an explicit per-image ground
coverage range. This script therefore uses a transparent heuristic:

    crop_side_m = clamp(height_m * crop_height_scale, min_crop_m, max_crop_m)

The exported dataset is useful for qualitative inspection and for bootstrap
experiments, but the scale alignment remains approximate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

OUT_SIZE = 512
JPEG_QUALITY = 95
PAD_RGB = (0, 0, 0)


@dataclass
class DroneRow:
    seq_id: str
    index: int
    filename: str
    image_path: Path
    lat: float
    lon: float
    height: float
    omega: Optional[float]
    kappa: Optional[float]
    phi1: Optional[float]
    phi2: Optional[float]


@dataclass
class Pair:
    sample_id: str
    seq_id: str
    image_name: str
    source_drone_path: Path
    source_satellite_desc: str
    lat: float
    lon: float
    height_m: float
    omega: Optional[float]
    kappa: Optional[float]
    phi1: Optional[float]
    heading_deg: float
    heading_source: str
    north_up_rot: float
    satellite_px_x: float
    satellite_px_y: float
    satellite_mpp_x: float
    satellite_mpp_y: float
    crop_side_m: float
    satellite_crop_mult: float
    satellite_crop_side_m: float
    satellite_extra_rotation_deg: float
    satellite_up_shift_m: float
    export_drone_path: str
    export_satellite_path: str


@dataclass
class Candidate:
    sample_id: str
    seq_id: str
    image_name: str
    source_drone_path: Path
    source_satellite_desc: str
    lat: float
    lon: float
    height_m: float
    omega: Optional[float]
    kappa: Optional[float]
    phi1: Optional[float]
    heading_deg: float
    heading_source: str
    north_up_rot: float
    satellite_px_x: float
    satellite_px_y: float
    satellite_mpp_x: float
    satellite_mpp_y: float
    crop_side_m: float
    satellite_crop_mult: float
    satellite_crop_side_m: float
    satellite_extra_rotation_deg: float
    satellite_up_shift_m: float


@dataclass
class SatelliteBounds:
    lt_lat: float
    lt_lon: float
    rb_lat: float
    rb_lon: float
    region: str


@dataclass
class Tile:
    path: Path
    x0: int
    y0: int
    width: int
    height: int


class SatelliteSource:
    def __init__(self, seq_dir: Path, seq_id: str) -> None:
        self.seq_id = seq_id
        sat_files = sorted(seq_dir.glob("satellite*.tif"))
        if not sat_files:
            raise FileNotFoundError(f"No satellite tif found in {seq_dir}")

        self.description: str
        self.size: Tuple[int, int]
        self.mode = "single"
        self.single_path: Optional[Path] = None
        self.tiles: List[Tile] = []

        if len(sat_files) == 1:
            self.single_path = sat_files[0]
            with Image.open(self.single_path) as im:
                self.size = im.size
            self.description = self.single_path.name
            return

        self.mode = "mosaic"
        tile_specs = []
        row_heights: Dict[int, int] = {}
        col_widths: Dict[int, int] = {}
        for path in sat_files:
            stem = path.stem
            suffix = stem.split("_", 1)[1]
            row_idx = int(suffix.split("-")[0])
            col_idx = int(suffix.split("-")[1])
            with Image.open(path) as im:
                width, height = im.size
            row_heights[row_idx] = max(row_heights.get(row_idx, 0), height)
            col_widths[col_idx] = max(col_widths.get(col_idx, 0), width)
            tile_specs.append((row_idx, col_idx, path, width, height))

        x_offsets: Dict[int, int] = {}
        y_offsets: Dict[int, int] = {}
        acc = 0
        for col_idx in sorted(col_widths):
            x_offsets[col_idx] = acc
            acc += col_widths[col_idx]
        total_width = acc

        acc = 0
        for row_idx in sorted(row_heights):
            y_offsets[row_idx] = acc
            acc += row_heights[row_idx]
        total_height = acc

        self.size = (total_width, total_height)
        self.description = ",".join(path.name for path in sat_files)
        for row_idx, col_idx, path, width, height in tile_specs:
            self.tiles.append(
                Tile(
                    path=path,
                    x0=x_offsets[col_idx],
                    y0=y_offsets[row_idx],
                    width=width,
                    height=height,
                )
            )

    def crop(self, box: Tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = box
        out_w = max(1, x2 - x1)
        out_h = max(1, y2 - y1)
        canvas = Image.new("RGB", (out_w, out_h), PAD_RGB)

        if self.mode == "single":
            assert self.single_path is not None
            with Image.open(self.single_path) as im:
                paste_intersection(canvas, im, box, (0, 0))
            return canvas

        for tile in self.tiles:
            tx1 = tile.x0
            ty1 = tile.y0
            tx2 = tile.x0 + tile.width
            ty2 = tile.y0 + tile.height
            ix1 = max(x1, tx1)
            iy1 = max(y1, ty1)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue

            with Image.open(tile.path) as im:
                local = im.crop((ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1)).convert("RGB")
            canvas.paste(local, (ix1 - x1, iy1 - y1))

        return canvas


def paste_intersection(
    canvas: Image.Image,
    src: Image.Image,
    global_box: Tuple[int, int, int, int],
    src_origin: Tuple[int, int],
) -> None:
    x1, y1, x2, y2 = global_box
    sx0, sy0 = src_origin
    sx1 = max(0, x1 - sx0)
    sy1 = max(0, y1 - sy0)
    sx2 = min(src.width, x2 - sx0)
    sy2 = min(src.height, y2 - sy0)
    if sx1 >= sx2 or sy1 >= sy2:
        return
    region = src.crop((sx1, sy1, sx2, sy2)).convert("RGB")
    canvas.paste(region, (sx1 + sx0 - x1, sy1 + sy0 - y1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UAV-VisLoc sat-drone pairs.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--sat-bounds-csv",
        type=Path,
        default=Path("satellite_ coordinates_range.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional subset of sequences, e.g. 01 02 03",
    )
    parser.add_argument(
        "--exclude-sequences",
        nargs="*",
        default=None,
        help="Optional sequences to skip.",
    )
    parser.add_argument("--roll-threshold", type=float, default=7.0)
    parser.add_argument("--pitch-threshold", type=float, default=7.0)
    parser.add_argument(
        "--allow-missing-pose",
        action="store_true",
        help="Keep rows without Omega/Kappa/Phi1 by falling back to GPS heading.",
    )
    parser.add_argument(
        "--crop-height-scale",
        type=float,
        default=0.7,
        help="Approximate crop_side_m = height * scale before clamping.",
    )
    parser.add_argument("--min-crop-side-m", type=float, default=256.0)
    parser.add_argument("--max-crop-side-m", type=float, default=640.0)
    parser.add_argument(
        "--sat-crop-mult",
        type=float,
        default=0.86,
        help="Satellite crop-side multiplier relative to crop_side_m.",
    )
    parser.add_argument(
        "--sat-extra-rotation",
        type=float,
        default=-4.0,
        help="Extra CCW rotation applied to satellite crops after north-up extraction.",
    )
    parser.add_argument(
        "--sat-up-shift-m",
        type=float,
        default=0.0,
        help="Move visible satellite content upward by this many meters.",
    )
    parser.add_argument("--output-size", type=int, default=OUT_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument(
        "--max-pairs-per-seq",
        type=int,
        default=None,
        help="Optional cap per sequence before global offset/max-pairs.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-offset", type=int, default=0)
    return parser.parse_args()


def load_satellite_bounds(csv_path: Path) -> Dict[str, SatelliteBounds]:
    bounds: Dict[str, SatelliteBounds] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            mapname = row["mapname"]
            seq_id = mapname.replace("satellite", "").replace(".tif", "")
            bounds[seq_id] = SatelliteBounds(
                lt_lat=float(row["LT_lat_map"]),
                lt_lon=float(row["LT_lon_map"]),
                rb_lat=float(row["RB_lat_map"]),
                rb_lon=float(row["RB_lon_map"]),
                region=row["region"],
            )
    return bounds


def read_rows(seq_dir: Path, seq_id: str) -> List[DroneRow]:
    rows: List[DroneRow] = []
    csv_path = seq_dir / f"{seq_id}.csv"
    with csv_path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            filename = row["filename"]
            rows.append(
                DroneRow(
                    seq_id=seq_id,
                    index=idx,
                    filename=filename,
                    image_path=seq_dir / "drone" / filename,
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    height=float(row["height"]),
                    omega=parse_optional_float(row.get("Omega")),
                    kappa=parse_optional_float(row.get("Kappa")),
                    phi1=parse_optional_float(row.get("Phi1")),
                    phi2=parse_optional_float(row.get("Phi2")),
                )
            )
    return rows


def parse_optional_float(token: Optional[str]) -> Optional[float]:
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def compute_track_headings(rows: Sequence[DroneRow]) -> List[Optional[float]]:
    headings: List[Optional[float]] = []
    n = len(rows)
    for idx in range(n):
        heading = first_non_none(
            search_heading(rows, idx, -1, 1),
            search_heading(rows, idx, -2, 2),
            search_heading(rows, idx, -3, 3),
            search_heading(rows, idx, -1, 2),
            search_heading(rows, idx, -2, 1),
        )
        headings.append(heading)
    return headings


def first_non_none(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def search_heading(
    rows: Sequence[DroneRow],
    idx: int,
    left_delta: int,
    right_delta: int,
) -> Optional[float]:
    left = idx + left_delta
    right = idx + right_delta
    if left < 0 or right >= len(rows):
        return None
    return bearing_deg(rows[left].lat, rows[left].lon, rows[right].lat, rows[right].lon)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    dlat_m = (lat2 - lat1) * 111320.0
    mean_lat = (lat1 + lat2) * 0.5
    dlon_m = (lon2 - lon1) * 111320.0 * math.cos(math.radians(mean_lat))
    if dlat_m * dlat_m + dlon_m * dlon_m < 1.0:
        return None
    return math.degrees(math.atan2(dlon_m, dlat_m)) % 360.0


def keep_row(row: DroneRow, args: argparse.Namespace) -> bool:
    if row.omega is None or row.kappa is None:
        return args.allow_missing_pose
    return abs(row.omega) <= args.pitch_threshold and abs(row.kappa) <= args.roll_threshold


def infer_heading(row: DroneRow, track_heading: Optional[float]) -> Tuple[Optional[float], str]:
    if row.phi1 is not None:
        return row.phi1 % 360.0, "phi1"
    if track_heading is not None:
        return track_heading, "gps_track"
    return None, "missing"


def make_sample_id(seq_id: str, filename: str) -> str:
    return f"{seq_id}_{Path(filename).stem}"


def crop_side_m(height_m: float, args: argparse.Namespace) -> float:
    value = height_m * args.crop_height_scale
    return max(args.min_crop_side_m, min(args.max_crop_side_m, value))


def latlon_to_pixel(
    lat: float,
    lon: float,
    bounds: SatelliteBounds,
    size: Tuple[int, int],
) -> Tuple[float, float]:
    width, height = size
    x = (lon - bounds.lt_lon) / (bounds.rb_lon - bounds.lt_lon) * width
    y = (lat - bounds.lt_lat) / (bounds.rb_lat - bounds.lt_lat) * height
    return x, y


def bounds_contains(lat: float, lon: float, bounds: SatelliteBounds) -> bool:
    return bounds.rb_lat <= lat <= bounds.lt_lat and bounds.lt_lon <= lon <= bounds.rb_lon


def satellite_mpp(bounds: SatelliteBounds, size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = size
    mean_lat = (bounds.lt_lat + bounds.rb_lat) * 0.5
    width_m = abs(bounds.rb_lon - bounds.lt_lon) * 111320.0 * math.cos(math.radians(mean_lat))
    height_m = abs(bounds.lt_lat - bounds.rb_lat) * 111320.0
    return width_m / width, height_m / height


def center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def valid_center_crop_side(side: int, deg: float) -> int:
    theta = math.radians(abs(deg) % 180.0)
    if theta > math.pi / 2:
        theta = math.pi - theta
    if theta > math.pi / 4:
        theta = math.pi / 2 - theta

    scale = math.cos(theta) + math.sin(theta)
    crop_side = int(math.floor(side / scale))
    return max(1, min(side, crop_side))


def rotate_square_valid(img: Image.Image, deg: float) -> Image.Image:
    square = center_crop_square(img)
    rotated = square.rotate(deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=PAD_RGB)
    crop_side = valid_center_crop_side(square.width, deg)
    left = (rotated.width - crop_side) // 2
    top = (rotated.height - crop_side) // 2
    return rotated.crop((left, top, left + crop_side, top + crop_side))


def discover_candidates(args: argparse.Namespace) -> Tuple[List[Candidate], Dict[str, int]]:
    bounds_map = load_satellite_bounds(args.sat_bounds_csv)
    seq_ids = sorted(p.name for p in args.data_root.iterdir() if p.is_dir())
    if args.sequences:
        wanted = {item.zfill(2) for item in args.sequences}
        seq_ids = [seq for seq in seq_ids if seq in wanted]
    if args.exclude_sequences:
        excluded = {item.zfill(2) for item in args.exclude_sequences}
        seq_ids = [seq for seq in seq_ids if seq not in excluded]

    candidates: List[Candidate] = []
    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "skipped_pose_filter": 0,
        "skipped_missing_heading": 0,
        "skipped_outside_satellite": 0,
    }

    for seq_id in seq_ids:
        seq_dir = args.data_root / seq_id
        bounds = bounds_map.get(seq_id)
        if bounds is None:
            continue
        sat_source = SatelliteSource(seq_dir, seq_id)
        sat_mpp_x, sat_mpp_y = satellite_mpp(bounds, sat_source.size)
        rows = read_rows(seq_dir, seq_id)
        track_headings = compute_track_headings(rows)

        for row, track_heading in zip(rows, track_headings):
            stats["rows_total"] += 1
            if not keep_row(row, args):
                stats["skipped_pose_filter"] += 1
                continue

            heading, heading_source = infer_heading(row, track_heading)
            if heading is None:
                stats["skipped_missing_heading"] += 1
                continue
            if not bounds_contains(row.lat, row.lon, bounds):
                stats["skipped_outside_satellite"] += 1
                continue

            north_up_rot = (360.0 - heading) % 360.0
            center_x, center_y = latlon_to_pixel(row.lat, row.lon, bounds, sat_source.size)
            side_m = crop_side_m(row.height, args)
            sat_side_m = side_m * args.sat_crop_mult
            side_px_x = max(1, int(round(sat_side_m / sat_mpp_x)))
            side_px_y = max(1, int(round(sat_side_m / sat_mpp_y)))
            sat_box = (
                int(round(center_x - side_px_x / 2)),
                int(round(center_y - side_px_y / 2)),
                int(round(center_x + side_px_x / 2)),
                int(round(center_y + side_px_y / 2)),
            )

            sample_id = make_sample_id(seq_id, row.filename)
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    seq_id=seq_id,
                    image_name=row.filename,
                    source_drone_path=row.image_path,
                    source_satellite_desc=sat_source.description,
                    lat=row.lat,
                    lon=row.lon,
                    height_m=row.height,
                    omega=row.omega,
                    kappa=row.kappa,
                    phi1=row.phi1,
                    heading_deg=heading,
                    heading_source=heading_source,
                    north_up_rot=north_up_rot,
                    satellite_px_x=center_x,
                    satellite_px_y=center_y,
                    satellite_mpp_x=sat_mpp_x,
                    satellite_mpp_y=sat_mpp_y,
                    crop_side_m=side_m,
                    satellite_crop_mult=args.sat_crop_mult,
                    satellite_crop_side_m=sat_side_m,
                    satellite_extra_rotation_deg=args.sat_extra_rotation,
                    satellite_up_shift_m=args.sat_up_shift_m,
                )
            )
            stats["rows_kept"] += 1

    if args.max_pairs_per_seq is not None:
        per_seq_counts: Dict[str, int] = {}
        filtered: List[Candidate] = []
        for cand in candidates:
            count = per_seq_counts.get(cand.seq_id, 0)
            if count >= args.max_pairs_per_seq:
                continue
            per_seq_counts[cand.seq_id] = count + 1
            filtered.append(cand)
        candidates = filtered

    if args.pair_offset:
        candidates = candidates[args.pair_offset :]
    if args.max_pairs is not None:
        candidates = candidates[: args.max_pairs]
    return candidates, stats


def export_selected_pairs(args: argparse.Namespace, candidates: Sequence[Candidate]) -> List[Pair]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drone_out = args.output_dir / "drone"
    sat_out = args.output_dir / "satellite"
    drone_out.mkdir(parents=True, exist_ok=True)
    sat_out.mkdir(parents=True, exist_ok=True)

    sat_cache: Dict[str, SatelliteSource] = {}
    pairs: List[Pair] = []

    for cand in candidates:
        sat_source = sat_cache.get(cand.seq_id)
        if sat_source is None:
            sat_source = SatelliteSource(args.data_root / cand.seq_id, cand.seq_id)
            sat_cache[cand.seq_id] = sat_source

        side_px_x = max(1, int(round(cand.satellite_crop_side_m / cand.satellite_mpp_x)))
        side_px_y = max(1, int(round(cand.satellite_crop_side_m / cand.satellite_mpp_y)))
        center_y_shifted = cand.satellite_px_y + (cand.satellite_up_shift_m / cand.satellite_mpp_y)
        sat_box = (
            int(round(cand.satellite_px_x - side_px_x / 2)),
            int(round(center_y_shifted - side_px_y / 2)),
            int(round(cand.satellite_px_x + side_px_x / 2)),
            int(round(center_y_shifted + side_px_y / 2)),
        )

        with Image.open(cand.source_drone_path) as drone_im:
            drone_rgb = drone_im.convert("RGB")
            drone_rot = rotate_square_valid(drone_rgb, cand.north_up_rot)
            drone_out_im = drone_rot.resize((args.output_size, args.output_size), Image.Resampling.LANCZOS)

        sat_crop = sat_source.crop(sat_box)
        sat_out_im = ImageOps.fit(
            sat_crop,
            (args.output_size, args.output_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        if abs(cand.satellite_extra_rotation_deg) > 1e-6:
            sat_out_im = rotate_square_valid(sat_out_im, cand.satellite_extra_rotation_deg).resize(
                (args.output_size, args.output_size),
                Image.Resampling.LANCZOS,
            )

        export_drone_rel = f"drone/{cand.sample_id}.jpg"
        export_sat_rel = f"satellite/{cand.sample_id}.jpg"
        drone_out_im.save(args.output_dir / export_drone_rel, "JPEG", quality=args.jpeg_quality)
        sat_out_im.save(args.output_dir / export_sat_rel, "JPEG", quality=args.jpeg_quality)
        pairs.append(
            Pair(
                sample_id=cand.sample_id,
                seq_id=cand.seq_id,
                image_name=cand.image_name,
                source_drone_path=cand.source_drone_path,
                source_satellite_desc=cand.source_satellite_desc,
                lat=cand.lat,
                lon=cand.lon,
                height_m=cand.height_m,
                omega=cand.omega,
                kappa=cand.kappa,
                phi1=cand.phi1,
                heading_deg=cand.heading_deg,
                heading_source=cand.heading_source,
                north_up_rot=cand.north_up_rot,
                satellite_px_x=cand.satellite_px_x,
                satellite_px_y=cand.satellite_px_y,
                satellite_mpp_x=cand.satellite_mpp_x,
                satellite_mpp_y=cand.satellite_mpp_y,
                crop_side_m=cand.crop_side_m,
                satellite_crop_mult=cand.satellite_crop_mult,
                satellite_crop_side_m=cand.satellite_crop_side_m,
                satellite_extra_rotation_deg=cand.satellite_extra_rotation_deg,
                satellite_up_shift_m=cand.satellite_up_shift_m,
                export_drone_path=export_drone_rel,
                export_satellite_path=export_sat_rel,
            )
        )

    return pairs


def write_pairs_csv(path: Path, pairs: Iterable[Pair]) -> None:
    fieldnames = [
        "sample_id",
        "seq_id",
        "image_name",
        "lat",
        "lon",
        "height_m",
        "omega",
        "kappa",
        "phi1",
        "heading_deg",
        "heading_source",
        "north_up_rot",
        "satellite_px_x",
        "satellite_px_y",
        "satellite_mpp_x",
        "satellite_mpp_y",
        "crop_side_m",
        "satellite_crop_mult",
        "satellite_crop_side_m",
        "satellite_extra_rotation_deg",
        "satellite_up_shift_m",
        "source_drone_path",
        "source_satellite_desc",
        "export_drone_path",
        "export_satellite_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sample_id": pair.sample_id,
                    "seq_id": pair.seq_id,
                    "image_name": pair.image_name,
                    "lat": f"{pair.lat:.8f}",
                    "lon": f"{pair.lon:.8f}",
                    "height_m": f"{pair.height_m:.2f}",
                    "omega": "" if pair.omega is None else f"{pair.omega:.6f}",
                    "kappa": "" if pair.kappa is None else f"{pair.kappa:.6f}",
                    "phi1": "" if pair.phi1 is None else f"{pair.phi1:.6f}",
                    "heading_deg": f"{pair.heading_deg:.3f}",
                    "heading_source": pair.heading_source,
                    "north_up_rot": f"{pair.north_up_rot:.3f}",
                    "satellite_px_x": f"{pair.satellite_px_x:.3f}",
                    "satellite_px_y": f"{pair.satellite_px_y:.3f}",
                    "satellite_mpp_x": f"{pair.satellite_mpp_x:.6f}",
                    "satellite_mpp_y": f"{pair.satellite_mpp_y:.6f}",
                    "crop_side_m": f"{pair.crop_side_m:.2f}",
                    "satellite_crop_mult": f"{pair.satellite_crop_mult:.3f}",
                    "satellite_crop_side_m": f"{pair.satellite_crop_side_m:.2f}",
                    "satellite_extra_rotation_deg": f"{pair.satellite_extra_rotation_deg:.3f}",
                    "satellite_up_shift_m": f"{pair.satellite_up_shift_m:.2f}",
                    "source_drone_path": str(pair.source_drone_path),
                    "source_satellite_desc": pair.source_satellite_desc,
                    "export_drone_path": pair.export_drone_path,
                    "export_satellite_path": pair.export_satellite_path,
                }
            )


def write_dataset_info(
    path: Path,
    args: argparse.Namespace,
    pairs: Sequence[Pair],
    stats: Dict[str, int],
) -> None:
    per_seq: Dict[str, int] = {}
    for pair in pairs:
        per_seq[pair.seq_id] = per_seq.get(pair.seq_id, 0) + 1
    payload = {
        "name": "UAV-VisLoc",
        "total": len(pairs),
        "per_sequence": per_seq,
        "image_size_px": [args.output_size, args.output_size],
        "image_format": "JPEG",
        "image_quality": args.jpeg_quality,
        "north_up_rot_field": "north_up_rot",
        "heuristic_crop_model": {
            "formula": "crop_side_m = clamp(height_m * crop_height_scale, min_crop_side_m, max_crop_side_m)",
            "crop_height_scale": args.crop_height_scale,
            "min_crop_side_m": args.min_crop_side_m,
            "max_crop_side_m": args.max_crop_side_m,
            "satellite_crop_mult": args.sat_crop_mult,
            "satellite_crop_formula": "satellite_crop_side_m = crop_side_m * sat_crop_mult",
            "satellite_extra_rotation_deg": args.sat_extra_rotation,
            "satellite_up_shift_m": args.sat_up_shift_m,
        },
        "filters": {
            "roll_threshold_deg": args.roll_threshold,
            "pitch_threshold_deg": args.pitch_threshold,
            "allow_missing_pose": args.allow_missing_pose,
            "max_pairs_per_seq": args.max_pairs_per_seq,
        },
        "stats_before_limit": stats,
        "notes": [
            "Scale alignment is approximate because UAV-VisLoc does not provide per-image ground footprint metadata.",
            "Satellite crops are north-up by construction from the satellite rasters, with optional extra CCW rotation.",
            "Drone images are center-cropped to square, rotated CCW by north_up_rot, then center-cropped to the largest valid inner square.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    candidates, stats = discover_candidates(args)
    pairs = export_selected_pairs(args, candidates)
    write_pairs_csv(args.output_dir / "pairs.csv", pairs)
    write_dataset_info(args.output_dir / "dataset_info.json", args, pairs, stats)
    print(f"Exported {len(pairs)} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()

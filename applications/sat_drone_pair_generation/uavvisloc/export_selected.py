#!/usr/bin/env python3
"""Export the currently selected high-quality UAV-VisLoc sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

try:
    from .build_pairs import discover_candidates, export_selected_pairs, write_pairs_csv
except ImportError:  # pragma: no cover - keep direct script execution working
    from build_pairs import discover_candidates, export_selected_pairs, write_pairs_csv


SEQ_CONFIGS = [
    {"seq_id": "01", "sat_crop_mult": 0.86, "sat_extra_rotation": -4.0, "sat_up_shift_m": 15.0},
    {"seq_id": "02", "sat_crop_mult": 0.86, "sat_extra_rotation": -4.0, "sat_up_shift_m": 0.0},
    {"seq_id": "03", "sat_crop_mult": 0.86, "sat_extra_rotation": -4.0, "sat_up_shift_m": -15.0},
    {"seq_id": "04", "sat_crop_mult": 0.86, "sat_extra_rotation": -4.0, "sat_up_shift_m": 0.0},
    {"seq_id": "06", "sat_crop_mult": 0.36, "sat_extra_rotation": -4.0, "sat_up_shift_m": 0.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sat-bounds-csv", type=Path, default=Path("satellite_ coordinates_range.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roll-threshold", type=float, default=7.0)
    parser.add_argument("--pitch-threshold", type=float, default=7.0)
    parser.add_argument("--allow-missing-pose", action="store_true")
    parser.add_argument("--crop-height-scale", type=float, default=0.7)
    parser.add_argument("--min-crop-side-m", type=float, default=256.0)
    parser.add_argument("--max-crop-side-m", type=float, default=640.0)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def make_seq_args(base: argparse.Namespace, cfg: Dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=base.data_root,
        sat_bounds_csv=base.sat_bounds_csv,
        output_dir=base.output_dir,
        sequences=[cfg["seq_id"]],
        exclude_sequences=None,
        roll_threshold=base.roll_threshold,
        pitch_threshold=base.pitch_threshold,
        allow_missing_pose=base.allow_missing_pose,
        crop_height_scale=base.crop_height_scale,
        min_crop_side_m=base.min_crop_side_m,
        max_crop_side_m=base.max_crop_side_m,
        sat_crop_mult=cfg["sat_crop_mult"],
        sat_extra_rotation=cfg["sat_extra_rotation"],
        sat_up_shift_m=cfg["sat_up_shift_m"],
        output_size=base.output_size,
        jpeg_quality=base.jpeg_quality,
        max_pairs_per_seq=None,
        max_pairs=None,
        pair_offset=0,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    merged_stats = Counter()
    exported_cfgs: List[Dict[str, float]] = []

    for cfg in SEQ_CONFIGS:
        seq_args = make_seq_args(args, cfg)
        candidates, stats = discover_candidates(seq_args)
        pairs = export_selected_pairs(seq_args, candidates)
        all_pairs.extend(pairs)
        merged_stats.update(stats)
        exported_cfgs.append(dict(cfg, total=len(pairs)))
        print(f"[{cfg['seq_id']}] exported {len(pairs)} pairs")

    write_pairs_csv(args.output_dir / "pairs.csv", all_pairs)

    per_seq = Counter(pair.seq_id for pair in all_pairs)
    payload = {
        "name": "UAV-VisLoc",
        "total": len(all_pairs),
        "per_sequence": dict(sorted(per_seq.items())),
        "image_size_px": [args.output_size, args.output_size],
        "image_format": "JPEG",
        "image_quality": args.jpeg_quality,
        "north_up_rot_field": "north_up_rot",
        "selected_sequence_configs": exported_cfgs,
        "filters": {
            "roll_threshold_deg": args.roll_threshold,
            "pitch_threshold_deg": args.pitch_threshold,
            "allow_missing_pose": args.allow_missing_pose,
        },
        "stats_before_export": dict(merged_stats),
        "notes": [
            "This exporter uses the currently selected high-quality sequences only: 01, 02, 03, 04, 06.",
            "Sequence-specific satellite scale/rotation/vertical-shift parameters are applied.",
            "For an additional tighter view, run center_recrop_pairs.py with --crop-size 384 after export.",
        ],
    }
    (args.output_dir / "dataset_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(f"Exported {len(all_pairs)} total pairs to {args.output_dir}")


if __name__ == "__main__":
    main()

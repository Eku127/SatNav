#!/usr/bin/env python3
"""Create a deterministic subset from offline trajectory annotations."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def scene_key(annotation: dict) -> str:
    scene = annotation.get("_scene_id", "")
    if not scene:
        video = annotation.get("video", "")
        parts = Path(video).parts
        if len(parts) >= 2:
            return parts[1].split("_satnav_")[0]
        return ""
    return Path(scene).name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an offline smoke subset.")
    parser.add_argument("--src", required=True, help="Source annotations.json path")
    parser.add_argument("--dst", required=True, help="Output subset JSON path")
    parser.add_argument("--count", type=int, default=128, help="Max annotations to keep")
    parser.add_argument(
        "--per-scene",
        type=int,
        default=2,
        help="Max annotations to keep per scene during the first pass",
    )
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    kept = []
    scene_counts = defaultdict(int)
    seen_ids = set()

    for annotation in annotations:
        if len(kept) >= args.count:
            break
        scene = scene_key(annotation)
        if scene_counts[scene] >= args.per_scene:
            continue
        kept.append(annotation)
        scene_counts[scene] += 1
        seen_ids.add(annotation.get("id"))

    if len(kept) < args.count:
        for annotation in annotations:
            if len(kept) >= args.count:
                break
            if annotation.get("id") in seen_ids:
                continue
            kept.append(annotation)
            seen_ids.add(annotation.get("id"))

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    print(
        f"Wrote {len(kept)} offline trajectories from {args.src} to {args.dst} "
        f"(per_scene={args.per_scene})"
    )


if __name__ == "__main__":
    main()

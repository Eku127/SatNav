#!/usr/bin/env python3
"""Create a deterministic small subset from a SatNav dataset JSON file."""

import argparse
import json
import os
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small deterministic subset for SatNav smoke tests."
    )
    parser.add_argument("--src", required=True, help="Source dataset JSON path")
    parser.add_argument("--dst", required=True, help="Output subset JSON path")
    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="Maximum number of episodes to keep",
    )
    parser.add_argument(
        "--per-scene",
        type=int,
        default=2,
        help="Maximum number of episodes to keep per scene during the first pass",
    )
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    kept = []
    scene_counts = defaultdict(int)

    for episode in episodes:
        if len(kept) >= args.count:
            break
        scene_id = episode.get("scene_id", "")
        if scene_counts[scene_id] >= args.per_scene:
            continue
        kept.append(episode)
        scene_counts[scene_id] += 1

    if len(kept) < args.count:
        seen_ids = {ep.get("episode_id") for ep in kept}
        for episode in episodes:
            if len(kept) >= args.count:
                break
            if episode.get("episode_id") in seen_ids:
                continue
            kept.append(episode)
            seen_ids.add(episode.get("episode_id"))

    out = dict(data)
    out["episodes"] = kept

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(
        f"Wrote {len(kept)} episodes from {args.src} to {args.dst} "
        f"(per_scene={args.per_scene})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate SatNav episode data and scene coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from satnav.dataset import SatNavDataset, SceneResolver


def _file_identity(path: Path) -> Mapping[str, Any]:
    """Return a portable file identity without retaining a host path."""
    return {
        "id": path.name,
        "size": path.stat().st_size,
    }


def _directory_identity(path: Path) -> Mapping[str, str]:
    """Identify a configured directory by its portable logical name only."""
    return {"id": path.name or "scenes"}


def _portable_episode_key(value: Any) -> str:
    """Remove a legacy scene path from a stable-key diagnostic, if present."""
    key = str(value)
    parts = key.split("::")
    if len(parts) == 3:
        parts[1] = SceneResolver.logical_scene_id(parts[1])
        return "::".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _normalize_source(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Match the loader's longstanding string-ID compatibility contract."""
    normalized = dict(row)
    normalized["episode_id"] = str(normalized.get("episode_id", ""))
    normalized["trajectory_id"] = str(normalized.get("trajectory_id", ""))
    return normalized


def _parse_split(value: str) -> Tuple[str, Path]:
    split, separator, path = value.partition("=")
    if not separator or not split or not path:
        raise argparse.ArgumentTypeError("--split must be NAME=/path/to/file.json")
    return split, Path(path)


def validate_split(
    split: str,
    path: Path,
    scenes_dir: Path,
    expected_count: int,
) -> Mapping[str, Any]:
    errors: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    source_rows = source.get("episodes") if isinstance(source, dict) else None
    if not isinstance(source_rows, list):
        raise ValueError(f"{path} must contain an episodes list")

    dataset = SatNavDataset(
        {
            "SPLIT": split,
            "DATA_PATH": str(path),
            "SCENES_DIR": str(scenes_dir),
            "CONTENT_SCENES": ["*"],
            "EPISODES_ALLOWED": ["*"],
        }
    )
    if len(dataset.episodes) != expected_count:
        errors.append(
            f"loader count mismatch: expected {expected_count}, got {len(dataset.episodes)}"
        )
    if len(source_rows) != expected_count:
        errors.append(
            f"source count mismatch: expected {expected_count}, got {len(source_rows)}"
        )

    keys = []
    field_counts: Counter = Counter()
    mismatch_examples = []
    absolute_scene_ids = []
    missing_scene_assets = []
    runtime_path_leaks = []
    for index, (raw, episode) in enumerate(zip(source_rows, dataset.episodes)):
        keys.append(episode.episode_key)
        serialized = episode.to_dict()
        field_counts.update(serialized.keys())
        expected = _normalize_source(raw)
        if serialized != expected and len(mismatch_examples) < 20:
            differing = sorted(
                key
                for key in set(serialized).union(expected)
                if serialized.get(key) != expected.get(key)
            )
            mismatch_examples.append({"index": index, "fields": differing})
        if os.path.isabs(str(episode.scene_id)):
            absolute_scene_ids.append(index)
        if episode.scene_path:
            scene_path = Path(episode.scene_path)
            candidates = (scene_path, Path(str(scene_path) + ".tif"))
            if not any(candidate.is_file() for candidate in candidates):
                missing_scene_assets.append(
                    {
                        "index": index,
                        "scene_id": SceneResolver.logical_scene_id(
                            episode.scene_id
                        ),
                    }
                )
        if episode.scene_path and episode.scene_path in json.dumps(serialized):
            runtime_path_leaks.append(index)

    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if mismatch_examples:
        errors.append(f"{len(mismatch_examples)}+ loader round-trip mismatches")
    if duplicate_keys:
        errors.append(f"{len(duplicate_keys)} duplicate stable episode keys")
    if absolute_scene_ids:
        errors.append(f"{len(absolute_scene_ids)} absolute logical scene IDs")
    if missing_scene_assets:
        errors.append(f"{len(missing_scene_assets)} missing scene assets")
    if runtime_path_leaks:
        errors.append(f"{len(runtime_path_leaks)} serialized runtime path leaks")

    key_digest = hashlib.sha256()
    for key in sorted(keys):
        encoded = key.encode("utf-8")
        key_digest.update(len(encoded).to_bytes(8, "big"))
        key_digest.update(encoded)
    return {
        "status": "passed" if not errors else "failed",
        "artifact": _file_identity(path),
        "source_count": len(source_rows),
        "loader_count": len(dataset.episodes),
        "unique_key_count": len(set(keys)),
        "sorted_key_digest": key_digest.hexdigest(),
        "field_presence_counts": dict(sorted(field_counts.items())),
        "mismatch_examples": mismatch_examples,
        "duplicate_keys": [
            _portable_episode_key(key) for key in duplicate_keys[:20]
        ],
        "absolute_scene_id_indices": absolute_scene_ids[:20],
        "missing_scene_assets": missing_scene_assets[:20],
        "runtime_path_leak_indices": runtime_path_leaks[:20],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=_parse_split, action="append", required=True)
    parser.add_argument("--expected", action="append", required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. No report is written by default.",
    )
    args = parser.parse_args()

    expected = {}
    for value in args.expected:
        name, separator, count = value.partition("=")
        if not separator or not name or not count:
            raise ValueError("--expected must be NAME=COUNT")
        expected[name] = int(count)

    reports = {}
    for split, path in args.split:
        if split not in expected:
            raise ValueError(f"no --expected facts for split {split}")
        reports[split] = validate_split(split, path, args.scenes_dir, expected[split])
    status = "passed" if all(
        report["status"] == "passed" for report in reports.values()
    ) else "failed"
    if args.report is not None:
        payload = {
            "schema_version": 2,
            "status": status,
            "scene_dir": _directory_identity(args.scenes_dir),
            "splits": reports,
        }
        _atomic_json(args.report, payload)

    for split, report in reports.items():
        print(
            f"{split}: {report['status']} "
            f"({report['loader_count']} episodes)"
        )
        for error in report["errors"]:
            print(f"  - {error}")

    if args.report is not None:
        print(f"Report: {args.report}")
    if status == "passed":
        print("SatNav data configuration is complete.")
    else:
        print("SatNav data configuration is incomplete.")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

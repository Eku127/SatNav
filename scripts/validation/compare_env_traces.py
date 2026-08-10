#!/usr/bin/env python3
"""Compare deterministic environment traces and classify intended metadata changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, List, Mapping


def _file_identity(path: Path) -> Mapping[str, Any]:
    """Return a portable identity for a trace input."""
    payload = path.read_bytes()
    return {
        "id": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _logical_scene_id(value: Any) -> str:
    normalized = str(value or "").strip().replace("\\", "/").rstrip("/")
    return Path(normalized.rsplit("/", 1)[-1]).stem if normalized else ""


def _portable_report_value(value: Any) -> Any:
    """Keep diagnostics useful without copying machine paths into a report."""
    if isinstance(value, str):
        normalized = value.strip().replace("\\", "/")
        is_windows_absolute = len(normalized) >= 3 and (
            normalized[1:3] == ":/" or normalized.startswith("//")
        )
        is_scene_reference = normalized.startswith("data/scene_datasets/") or (
            "/" in normalized
            and Path(normalized).suffix.lower() in {".tif", ".tiff"}
        )
        if os.path.isabs(normalized) or is_windows_absolute or is_scene_reference:
            return {
                "id": Path(normalized.rstrip("/")).name or "filesystem-root"
            }
        return value
    if isinstance(value, Mapping):
        return {
            _portable_report_key(key): _portable_report_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_report_value(item) for item in value]
    return value


def _portable_report_key(value: Any) -> str:
    portable = _portable_report_value(str(value))
    if isinstance(portable, Mapping):
        return f"artifact:{portable['id']}"
    return str(portable)


def _portable_episode_key(trace: Mapping[str, Any]) -> Any:
    key = trace["episode"].get("episode_key")
    if key is None:
        return None
    return "::".join(
        (
            str(trace.get("source", {}).get("split", "")),
            _logical_scene_id(trace["episode"].get("scene_id")),
            str(trace["episode"].get("episode_id", "")),
        )
    )


def _compare(
    before: Any,
    after: Any,
    path: str,
    differences: List[Mapping[str, Any]],
    tolerance: float,
) -> None:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if not math.isclose(float(before), float(after), abs_tol=tolerance, rel_tol=0.0):
            differences.append(
                {
                    "path": path,
                    "before": _portable_report_value(before),
                    "after": _portable_report_value(after),
                }
            )
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before).union(after)):
            portable_key = _portable_report_key(key)
            child = f"{path}.{portable_key}" if path else portable_key
            if key not in before:
                differences.append(
                    {
                        "path": child,
                        "before": "<missing>",
                        "after": _portable_report_value(after[key]),
                    }
                )
            elif key not in after:
                differences.append(
                    {
                        "path": child,
                        "before": _portable_report_value(before[key]),
                        "after": "<missing>",
                    }
                )
            else:
                _compare(before[key], after[key], child, differences, tolerance)
        return
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            differences.append(
                {"path": f"{path}.length", "before": len(before), "after": len(after)}
            )
            return
        for index, (left, right) in enumerate(zip(before, after)):
            _compare(left, right, f"{path}[{index}]", differences, tolerance)
        return
    if before != after:
        differences.append(
            {
                "path": path,
                "before": _portable_report_value(before),
                "after": _portable_report_value(after),
            }
        )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    differences: List[Mapping[str, Any]] = []

    # Runtime behavior is the non-negotiable comparison surface.  New public
    # info fields and logical identity are classified separately below.
    behavior_before = {
        "actions_requested": before["actions_requested"],
        "steps": [
            {
                "step": row["step"],
                "action": row["action"],
                "observation": row["observation"],
                "agent_state": row["agent_state"],
                "metrics": row["metrics"],
                "done": row["done"],
            }
            for row in before["steps"]
        ],
    }
    behavior_after = {
        "actions_requested": after["actions_requested"],
        "steps": [
            {
                "step": row["step"],
                "action": row["action"],
                "observation": row["observation"],
                "agent_state": row["agent_state"],
                "metrics": row["metrics"],
                "done": row["done"],
            }
            for row in after["steps"]
        ],
    }
    _compare(
        behavior_before,
        behavior_after,
        "behavior",
        differences,
        args.metric_tolerance,
    )

    before_scene = str(before["episode"]["scene_id"])
    after_scene = str(after["episode"]["scene_id"])
    identity_changes = []
    if before_scene != after_scene:
        identity_changes.append(
            {
                "field": "episode.scene_id",
                "before": _logical_scene_id(before_scene),
                "after": _logical_scene_id(after_scene),
                "classification": "expected_change",
                "reason": "scene_id is now a logical benchmark ID, not a local path",
            }
        )
    if before["episode"].get("episode_key") != after["episode"].get("episode_key"):
        identity_changes.append(
            {
                "field": "episode.episode_key",
                "before": _portable_episode_key(before),
                "after": _portable_episode_key(after),
                "classification": "expected_change",
                "reason": "stable split::scene_id::episode_id identity was added",
            }
        )

    report = {
        "schema_version": 2,
        "status": "passed" if not differences else "failed",
        "behavior_equal": not differences,
        "metric_abs_tolerance": args.metric_tolerance,
        "unexpected_differences": differences,
        "identity_changes": identity_changes,
        "before": _file_identity(args.before),
        "after": _file_identity(args.after),
    }
    _atomic_json(args.output, report)
    print(
        f"status={report['status']} unexpected={len(differences)} "
        f"expected={len(identity_changes)} output={args.output}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

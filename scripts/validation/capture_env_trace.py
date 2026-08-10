#!/usr/bin/env python3
"""Capture a deterministic SatNav reset/step trace for before/after comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf

from satnav.core import Env
from satnav.dataset import SatNavDataset, SceneResolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> Mapping[str, Any]:
    """Return a portable, content-bound identity for an input artifact."""
    return {
        "id": path.name,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _repository_identity(repo: Path) -> Mapping[str, str]:
    """Identify source code without serializing its machine-local checkout."""
    return {
        "id": repo.resolve().name or "repository",
        "commit": _commit(repo),
    }


def _portable_path_value(value: str) -> Any:
    """Replace an incidental absolute path with a basename-only identity."""
    normalized = value.strip().replace("\\", "/")
    is_windows_absolute = len(normalized) >= 3 and (
        normalized[1:3] == ":/" or normalized.startswith("//")
    )
    is_scene_reference = normalized.startswith("data/scene_datasets/") or (
        "/" in normalized
        and Path(normalized).suffix.lower() in {".tif", ".tiff"}
    )
    if os.path.isabs(normalized) or is_windows_absolute or is_scene_reference:
        return {"id": Path(normalized.rstrip("/")).name or "filesystem-root"}
    return value


def _portable_mapping_key(value: Any) -> str:
    portable = _portable_path_value(str(value))
    if isinstance(portable, Mapping):
        return f"artifact:{portable['id']}"
    return str(portable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return _portable_path_value(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            _portable_mapping_key(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "position") and hasattr(value, "rotation"):
        return {
            "position": _jsonable(value.position),
            "rotation": _jsonable(value.rotation),
        }
    representation = repr(value)
    return {
        "type": type(value).__name__,
        "repr_sha256": hashlib.sha256(representation.encode("utf-8")).hexdigest(),
    }


def _observation_fingerprint(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for name, value in observation.items():
        portable_name = _portable_mapping_key(name)
        if isinstance(value, np.ndarray):
            result[portable_name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        else:
            result[portable_name] = _jsonable(value)
    return result


def _commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _agent_state(env: Any) -> Any:
    """Use the public contract, with a compatibility fallback for before code."""
    try:
        return env.agent_state
    except AttributeError:
        return env._task._sim.get_agent_state()  # before-refactor capture only


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sorted-index", type=int, default=0)
    parser.add_argument(
        "--actions",
        default="MOVE_FORWARD,TURN_RIGHT,MOVE_FORWARD,STOP",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.DATASET.DATA_PATH = str(args.data_path)
    config.DATASET.SCENES_DIR = str(args.scenes_dir)
    config.DATASET.SPLIT = args.split
    measurements = list(config.TASK.MEASUREMENTS)
    config.TASK.MEASUREMENTS = [
        item for item in measurements if str(item).upper() != "TOP_DOWN_MAP"
    ]

    dataset = SatNavDataset(config.DATASET)
    ordered = sorted(
        dataset.episodes,
        key=lambda episode: (
            str(episode.scene_id),
            str(episode.episode_id),
            str(getattr(episode, "trajectory_id", "")),
        ),
    )
    if args.sorted_index < 0 or args.sorted_index >= len(ordered):
        raise IndexError(
            f"sorted index {args.sorted_index} is outside {len(ordered)} episodes"
        )
    episode = ordered[args.sorted_index]
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]

    env = Env(config, dataset=dataset)
    try:
        observation = env.reset_to_episode(episode)
        steps = [
            {
                "step": 0,
                "action": None,
                "observation": _observation_fingerprint(observation),
                "agent_state": _jsonable(_agent_state(env)),
                "metrics": _jsonable(env.get_metrics()),
                "done": False,
                "last_step_info": None,
            }
        ]
        for index, action in enumerate(actions, start=1):
            observation, done, info = env.step(action)
            steps.append(
                {
                    "step": index,
                    "action": action,
                    "observation": _observation_fingerprint(observation),
                    "agent_state": _jsonable(_agent_state(env)),
                    "metrics": _jsonable(env.get_metrics()),
                    "done": bool(done),
                    "last_step_info": _jsonable(
                        getattr(env, "last_step_info", info)
                    ),
                }
            )
            if done:
                break

        payload = {
            "schema_version": 2,
            "source": {
                "repo": _repository_identity(args.repo),
                "config": _file_identity(args.config),
                "dataset": _file_identity(args.data_path),
                "split": args.split,
                "sorted_index": args.sorted_index,
            },
            "episode": {
                "episode_id": str(episode.episode_id),
                "trajectory_id": str(getattr(episode, "trajectory_id", "")),
                "scene_id": SceneResolver.logical_scene_id(episode.scene_id),
                "episode_key": (
                    None
                    if getattr(episode, "episode_key", None) is None
                    else "::".join(
                        (
                            args.split,
                            SceneResolver.logical_scene_id(episode.scene_id),
                            str(episode.episode_id),
                        )
                    )
                ),
            },
            "actions_requested": actions,
            "steps": steps,
        }
        _atomic_write(args.output, payload)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        print(f"trace_sha256={hashlib.sha256(canonical).hexdigest()}")
        print(f"steps={len(steps)} output={args.output}")
        return 0
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())

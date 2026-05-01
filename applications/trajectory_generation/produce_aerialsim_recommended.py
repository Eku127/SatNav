#!/usr/bin/env python3
"""Production pipeline for generating AerialSim trajectories from recommended groups.

This pipeline is designed for reliability rather than throughput:
- expand recommended trajectory groups into all member episodes
- generate in small batches so each batch gets a fresh browser process
- validate on-disk outputs after every batch
- retry only still-incomplete episodes/trajectories
- emit manifests for completed / incomplete production status
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from applications.trajectory_generation.aerial_quality import passes_aerial_quality_thresholds
from applications.trajectory_generation.utils import format_episode_dirname
from satnav.core.config import load_config
from satnav.dataset.satnav_dataset import SatNavDataset


DATASET_NAME = "satnav"


def _normalize_scene_id(scene_id: str) -> str:
    return scene_id.split("/")[-1] if isinstance(scene_id, str) and "/" in scene_id else scene_id


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _chunked(items: Sequence[Dict], chunk_size: int) -> Iterable[Sequence[Dict]]:
    if chunk_size < 1:
        chunk_size = 1
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _load_recommended_rows(path: Path) -> List[Dict]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("selected"), list):
            return payload["selected"]
        if isinstance(payload.get("ranked"), list):
            return payload["ranked"]
        if isinstance(payload.get("items"), list):
            return payload["items"]
        raise ValueError(f"Unsupported recommended JSON object schema: {path}")
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported recommended JSON schema: {path}")
    return payload


def _passes_thresholds(
    row: Dict,
    min_score: Optional[float],
    min_avg_std: float,
    max_avg_black_frac: float,
    min_avg_height_range: float,
) -> bool:
    return passes_aerial_quality_thresholds(
        row,
        min_score=min_score,
        min_avg_std=min_avg_std,
        max_avg_black_frac=max_avg_black_frac,
        min_avg_height_range=min_avg_height_range,
    )


def _build_targets(
    recommended_path: Path,
    groups_path: Path,
    max_trajectories: int,
    min_score: Optional[float],
    min_avg_std: float,
    max_avg_black_frac: float,
    min_avg_height_range: float,
) -> Tuple[List[Dict], List[str]]:
    recommended_rows = _load_recommended_rows(recommended_path)
    groups = _read_json(groups_path)
    if not isinstance(groups, list):
        raise ValueError(f"trajectory_groups_full must be a list: {groups_path}")

    group_index = {}
    for row in groups:
        key = row.get("trajectory_group_key")
        if isinstance(key, str) and key:
            group_index[key] = row

    selected: List[Dict] = []
    missing: List[str] = []
    seen = set()
    for row in recommended_rows:
        key = row.get("trajectory_group_key")
        if not isinstance(key, str) or not key or key in seen:
            continue
        if not _passes_thresholds(
            row,
            min_score=min_score,
            min_avg_std=min_avg_std,
            max_avg_black_frac=max_avg_black_frac,
            min_avg_height_range=min_avg_height_range,
        ):
            continue
        group = group_index.get(key)
        if group is None:
            missing.append(key)
            continue
        episodes = sorted(
            group.get("episodes", []),
            key=lambda item: int(item.get("episode_index", -1)),
        )
        episode_indices = [int(item["episode_index"]) for item in episodes]
        if not episode_indices:
            missing.append(key)
            continue
        selected.append(
            {
                "trajectory_group_key": key,
                "scene_id": _normalize_scene_id(str(group.get("scene_id", row.get("scene_id", "")))),
                "trajectory_id": str(row.get("trajectory_id", group.get("representative", {}).get("trajectory_id", ""))),
                "trajectory_type": str(row.get("trajectory_type", group.get("representative", {}).get("trajectory_type", ""))),
                "score": float(row.get("score", 0.0)),
                "episode_count": int(group.get("episode_count", len(episode_indices))),
                "path_point_count": int(group.get("path_point_count", row.get("path_point_count", 0))),
                "representative_episode_index": int(
                    row.get("representative_episode_index", group.get("representative", {}).get("episode_index", episode_indices[0]))
                ),
                "episode_indices": episode_indices,
                "episodes": episodes,
            }
        )
        seen.add(key)
        if max_trajectories > 0 and len(selected) >= max_trajectories:
            break
    return selected, missing


def _load_episode_metadata(config_path: Path, episode_indices: Sequence[int]) -> Dict[int, Dict]:
    config = load_config(config_path)
    dataset = SatNavDataset(config.DATASET)
    meta: Dict[int, Dict] = {}
    for episode_idx in sorted(set(int(idx) for idx in episode_indices)):
        episode = dataset.episodes[episode_idx]
        meta[episode_idx] = {
            "scene_id": _normalize_scene_id(str(episode.scene_id)),
            "episode_id": str(episode.episode_id),
            "trajectory_id": str(getattr(episode, "trajectory_id", "")),
            "trajectory_type": str(getattr(episode, "trajectory_type", "")),
        }
    return meta


def _is_summary_entry_complete(output_dir: Path, entry: Dict) -> bool:
    try:
        episode_idx = int(entry["id"])
        steps = int(entry["steps"])
        video_rel = entry["video"]
        actions = entry["actions"]
    except (KeyError, TypeError, ValueError):
        return False

    if episode_idx < 0 or steps < 0 or not isinstance(video_rel, str) or not isinstance(actions, list):
        return False
    if len(actions) != steps + 1:
        return False

    rgb_dir = output_dir / video_rel / "rgb"
    if not rgb_dir.is_dir():
        return False

    jpgs = sorted(path.name for path in rgb_dir.iterdir() if path.suffix.lower() == ".jpg")
    expected = steps + 1
    if len(jpgs) != expected:
        return False
    for idx, filename in enumerate(jpgs, start=1):
        if filename != f"{idx:03d}.jpg":
            return False
    return True


def _load_valid_summary_entries(output_dir: Path, target_episode_ids: Optional[set] = None) -> Dict[int, Dict]:
    summary_path = output_dir / "summary.json"
    valid: Dict[int, Dict] = {}
    if not summary_path.exists():
        return valid

    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                episode_idx = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            if target_episode_ids is not None and episode_idx not in target_episode_ids:
                continue
            if _is_summary_entry_complete(output_dir, entry):
                valid[episode_idx] = entry
    return valid


def _cleanup_episode_artifacts(output_dir: Path, scene_id: str, episode_idx: int) -> None:
    episode_dirname = format_episode_dirname(_normalize_scene_id(scene_id), DATASET_NAME, int(episode_idx))
    episode_dir = output_dir / "images" / episode_dirname
    if episode_dir.exists():
        shutil.rmtree(episode_dir)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _default_runtime_roots(output_dir: Path) -> Tuple[Path, Path]:
    digest = hashlib.sha1(str(output_dir).encode("utf-8")).hexdigest()[:10]
    base = Path("/mnt/data3/jiangjiajun/tmp/sa") / digest
    return base / "r", base / "t"


def _cleanup_stale_runtime_roots(runtime_root: Path, tmp_root: Path) -> None:
    for root in (runtime_root, tmp_root):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.name.startswith("round_") and child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


def _prepare_batch_runtime_dir(
    runtime_root: Path,
    tmp_root: Path,
    round_idx: int,
    batch_idx: int,
) -> Dict[str, Path]:
    batch_runtime_dir = runtime_root / f"round_{round_idx:02d}_batch_{batch_idx:03d}"
    batch_tmp_dir = tmp_root / f"round_{round_idx:02d}_batch_{batch_idx:03d}"
    _reset_directory(batch_runtime_dir)
    _reset_directory(batch_tmp_dir)
    batch_tmp_dir.chmod(0o700)

    layout = {
        "root": batch_runtime_dir,
        "tmp": batch_tmp_dir,
        "xdg_config": runtime_root / "xdg-config",
        "aerialsim": batch_runtime_dir / "aerialsim",
        "chrome_user_data": batch_runtime_dir / "chrome-user-data",
        "chrome_disk_cache": batch_runtime_dir / "chrome-disk-cache",
        "chrome_crash_dumps": batch_runtime_dir / "chrome-crash-dumps",
    }
    for path in layout.values():
        if path in {batch_runtime_dir, batch_tmp_dir}:
            continue
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _cleanup_batch_runtime_dir(runtime_layout: Dict[str, Path]) -> None:
    for key in ("root", "tmp"):
        path = runtime_layout.get(key)
        if path and path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _load_attempt_counters(path: Path) -> Tuple[Dict[int, int], Dict[str, int]]:
    episode_attempts: Dict[int, int] = {}
    trajectory_attempts: Dict[str, int] = {}
    if not path.exists():
        return episode_attempts, trajectory_attempts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for episode_idx in row.get("episode_indices", []):
                episode_idx = int(episode_idx)
                episode_attempts[episode_idx] = max(episode_attempts.get(episode_idx, 0), int(row.get("episode_attempt", 0)))
            for group_key in row.get("trajectory_group_keys", []):
                group_key = str(group_key)
                trajectory_attempts[group_key] = max(trajectory_attempts.get(group_key, 0), int(row.get("trajectory_attempt", 0)))
    return episode_attempts, trajectory_attempts


def _compute_reports(
    targets: Sequence[Dict],
    valid_entries: Dict[int, Dict],
    episode_attempts: Dict[int, int],
    max_episode_attempts: int,
) -> Tuple[List[Dict], List[Dict]]:
    reports: List[Dict] = []
    pending_targets: List[Dict] = []
    for target in targets:
        episode_indices = [int(idx) for idx in target["episode_indices"]]
        completed_episode_indices = [idx for idx in episode_indices if idx in valid_entries]
        missing_episode_indices = [idx for idx in episode_indices if idx not in valid_entries]
        exhausted_episode_indices = [
            idx for idx in missing_episode_indices
            if episode_attempts.get(idx, 0) >= max_episode_attempts
        ]
        if not missing_episode_indices:
            status = "completed"
        elif len(exhausted_episode_indices) == len(missing_episode_indices):
            status = "failed"
        else:
            status = "pending"
            pending_targets.append(target)
        reports.append(
            {
                **target,
                "status": status,
                "completed_episode_indices": completed_episode_indices,
                "missing_episode_indices": missing_episode_indices,
                "exhausted_episode_indices": exhausted_episode_indices,
            }
        )
    return reports, pending_targets


def _build_episode_reports(
    targets: Sequence[Dict],
    episode_meta: Dict[int, Dict],
    valid_entries: Dict[int, Dict],
    episode_attempts: Dict[int, int],
    max_episode_attempts: int,
) -> List[Dict]:
    ownership = {}
    for target in targets:
        for episode_idx in target["episode_indices"]:
            ownership[int(episode_idx)] = target["trajectory_group_key"]

    reports = []
    for episode_idx in sorted(ownership):
        valid_entry = valid_entries.get(episode_idx)
        status = "completed" if valid_entry is not None else (
            "failed" if episode_attempts.get(episode_idx, 0) >= max_episode_attempts else "pending"
        )
        reports.append(
            {
                "episode_index": episode_idx,
                "trajectory_group_key": ownership[episode_idx],
                "scene_id": episode_meta[episode_idx]["scene_id"],
                "episode_id": episode_meta[episode_idx]["episode_id"],
                "attempts": episode_attempts.get(episode_idx, 0),
                "status": status,
                "video": valid_entry.get("video") if valid_entry else None,
                "steps": valid_entry.get("steps") if valid_entry else None,
            }
        )
    return reports


def _run_batch(
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    batch_indices_path: Path,
    log_path: Path,
    landmark_success: float,
    runtime_layout: Dict[str, Path],
) -> Tuple[int, float]:
    cmd = [
        "xvfb-run",
        "-a",
        sys.executable,
        "-m",
        "applications.trajectory_generation.generate",
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--landmark_success",
        str(landmark_success),
        "--episode_indices_file",
        str(batch_indices_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["TMPDIR"] = str(runtime_layout["tmp"])
    env["TMP"] = str(runtime_layout["tmp"])
    env["TEMP"] = str(runtime_layout["tmp"])
    env["XDG_RUNTIME_DIR"] = str(runtime_layout["tmp"])
    env["XDG_CONFIG_HOME"] = str(runtime_layout["xdg_config"])
    env["SATNAV_AERIALSIM_RUNTIME_DIR"] = str(runtime_layout["aerialsim"])
    env["SATNAV_AERIALSIM_CHROME_USER_DATA_DIR"] = str(runtime_layout["chrome_user_data"])
    env["SATNAV_AERIALSIM_CHROME_DISK_CACHE_DIR"] = str(runtime_layout["chrome_disk_cache"])
    env["SATNAV_AERIALSIM_CHROME_CRASH_DUMPS_DIR"] = str(runtime_layout["chrome_crash_dumps"])
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable production pipeline for recommended AerialSim trajectories")
    parser.add_argument("--config", type=str, default="configs/satnav_task.yaml")
    parser.add_argument("--recommended_json", type=str, default="output/aerialsim_train_traj_quality_eval_full/recommended_top.json")
    parser.add_argument("--trajectory_groups_json", type=str, default="output/preselect_0404_train_trajectory/trajectory_groups_full.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_trajectories", type=int, default=-1)
    parser.add_argument("--trajectories_per_batch", type=int, default=3)
    parser.add_argument("--episodes_per_batch", type=int, default=1)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--max_episode_attempts", type=int, default=4)
    parser.add_argument("--landmark_success", type=float, default=2.0)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--min_avg_std", type=float, default=45.0)
    parser.add_argument("--max_avg_black_frac", type=float, default=0.002)
    parser.add_argument("--min_avg_height_range", type=float, default=6.0)
    parser.add_argument("--runtime_root", type=str, default=None)
    parser.add_argument("--tmp_root", type=str, default=None)
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    recommended_path = (REPO_ROOT / args.recommended_json) if not Path(args.recommended_json).is_absolute() else Path(args.recommended_json)
    groups_path = (REPO_ROOT / args.trajectory_groups_json) if not Path(args.trajectory_groups_json).is_absolute() else Path(args.trajectory_groups_json)
    output_dir = Path(args.output_dir)
    pipeline_dir = output_dir / "pipeline"
    batches_dir = pipeline_dir / "batches"
    logs_dir = pipeline_dir / "logs"
    attempts_path = pipeline_dir / "attempts.jsonl"
    default_runtime_root, default_tmp_root = _default_runtime_roots(output_dir)
    runtime_root = Path(args.runtime_root) if args.runtime_root else default_runtime_root
    tmp_root = Path(args.tmp_root) if args.tmp_root else default_tmp_root
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_runtime_roots(runtime_root, tmp_root)

    targets, missing_group_keys = _build_targets(
        recommended_path=recommended_path,
        groups_path=groups_path,
        max_trajectories=args.max_trajectories,
        min_score=args.min_score,
        min_avg_std=args.min_avg_std,
        max_avg_black_frac=args.max_avg_black_frac,
        min_avg_height_range=args.min_avg_height_range,
    )
    if not targets:
        raise RuntimeError("No target trajectories selected for production.")

    episode_indices = sorted({idx for target in targets for idx in target["episode_indices"]})
    episode_meta = _load_episode_metadata(config_path, episode_indices)
    _write_json(
        pipeline_dir / "selection_manifest.json",
        {
            "config": {
                "config": str(config_path),
                "recommended_json": str(recommended_path),
                "trajectory_groups_json": str(groups_path),
                "max_trajectories": args.max_trajectories,
                "trajectories_per_batch": args.trajectories_per_batch,
                "episodes_per_batch": args.episodes_per_batch,
                "max_rounds": args.max_rounds,
                "max_episode_attempts": args.max_episode_attempts,
                "min_score": args.min_score,
                "min_avg_std": args.min_avg_std,
                "max_avg_black_frac": args.max_avg_black_frac,
                "min_avg_height_range": args.min_avg_height_range,
                "runtime_root": str(runtime_root),
                "tmp_root": str(tmp_root),
            },
            "summary": {
                "selected_trajectory_count": len(targets),
                "selected_episode_count": len(episode_indices),
                "missing_group_key_count": len(missing_group_keys),
            },
            "missing_group_keys": missing_group_keys,
            "targets": targets,
        },
    )
    _write_json(pipeline_dir / "selected_episode_indices.json", {"episode_indices": episode_indices})

    episode_attempts, trajectory_attempts = _load_attempt_counters(attempts_path)
    valid_entries = _load_valid_summary_entries(output_dir, target_episode_ids=set(episode_indices))

    round_idx = 1
    while round_idx <= args.max_rounds:
        trajectory_reports, pending_targets = _compute_reports(
            targets=targets,
            valid_entries=valid_entries,
            episode_attempts=episode_attempts,
            max_episode_attempts=args.max_episode_attempts,
        )
        completed_before = sum(1 for row in trajectory_reports if row["status"] == "completed")
        runnable_targets = []
        for target in pending_targets:
            if any(
                episode_idx not in valid_entries and episode_attempts.get(episode_idx, 0) < args.max_episode_attempts
                for episode_idx in target["episode_indices"]
            ):
                runnable_targets.append(target)

        if not runnable_targets:
            break

        episode_owner = {}
        for target in targets:
            for episode_idx in target["episode_indices"]:
                episode_owner[int(episode_idx)] = target["trajectory_group_key"]

        for batch_idx, batch_targets in enumerate(_chunked(runnable_targets, args.trajectories_per_batch), start=1):
            batch_episode_indices: List[int] = []
            for target in batch_targets:
                for episode_idx in target["episode_indices"]:
                    episode_idx = int(episode_idx)
                    if episode_idx in valid_entries:
                        continue
                    if episode_attempts.get(episode_idx, 0) >= args.max_episode_attempts:
                        continue
                    batch_episode_indices.append(episode_idx)
            batch_episode_indices = sorted(set(batch_episode_indices))
            if not batch_episode_indices:
                continue

            for run_idx, run_episode_indices in enumerate(_chunked(batch_episode_indices, args.episodes_per_batch), start=1):
                run_episode_indices = sorted(int(idx) for idx in run_episode_indices)
                run_group_keys = sorted({episode_owner[idx] for idx in run_episode_indices})
                current_episode_attempt = max(episode_attempts.get(idx, 0) for idx in run_episode_indices) + 1
                current_trajectory_attempt = max(trajectory_attempts.get(key, 0) for key in run_group_keys) + 1
                for episode_idx in run_episode_indices:
                    episode_attempts[episode_idx] = current_episode_attempt
                    _cleanup_episode_artifacts(output_dir, episode_meta[episode_idx]["scene_id"], episode_idx)
                for group_key in run_group_keys:
                    trajectory_attempts[group_key] = current_trajectory_attempt

                batch_spec = {
                    "round": round_idx,
                    "batch": batch_idx,
                    "run": run_idx,
                    "trajectory_group_keys": run_group_keys,
                    "episode_indices": run_episode_indices,
                }
                batch_indices_path = batches_dir / f"round_{round_idx:02d}_batch_{batch_idx:03d}_run_{run_idx:03d}.json"
                _write_json(batch_indices_path, batch_spec)
                log_path = logs_dir / f"round_{round_idx:02d}_batch_{batch_idx:03d}_run_{run_idx:03d}.log"
                runtime_layout = _prepare_batch_runtime_dir(runtime_root, tmp_root, round_idx, batch_idx * 1000 + run_idx)
                try:
                    return_code, elapsed_sec = _run_batch(
                        repo_root=REPO_ROOT,
                        config_path=config_path,
                        output_dir=output_dir,
                        batch_indices_path=batch_indices_path,
                        log_path=log_path,
                        landmark_success=args.landmark_success,
                        runtime_layout=runtime_layout,
                    )
                finally:
                    _cleanup_batch_runtime_dir(runtime_layout)
                valid_entries = _load_valid_summary_entries(output_dir, target_episode_ids=set(episode_indices))
                completed_episodes = [idx for idx in run_episode_indices if idx in valid_entries]
                missing_episodes = [idx for idx in run_episode_indices if idx not in valid_entries]
                _append_jsonl(
                    attempts_path,
                    {
                        "round": round_idx,
                        "batch": batch_idx,
                        "run": run_idx,
                        "trajectory_group_keys": run_group_keys,
                        "episode_indices": run_episode_indices,
                        "episode_attempt": current_episode_attempt,
                        "trajectory_attempt": current_trajectory_attempt,
                        "return_code": return_code,
                        "elapsed_sec": round(elapsed_sec, 2),
                        "completed_episode_indices": completed_episodes,
                        "missing_episode_indices": missing_episodes,
                        "log_path": str(log_path),
                    },
                )

        valid_entries = _load_valid_summary_entries(output_dir, target_episode_ids=set(episode_indices))
        trajectory_reports, _ = _compute_reports(
            targets=targets,
            valid_entries=valid_entries,
            episode_attempts=episode_attempts,
            max_episode_attempts=args.max_episode_attempts,
        )
        completed_after = sum(1 for row in trajectory_reports if row["status"] == "completed")
        if completed_after == completed_before and completed_after < len(targets) and round_idx >= args.max_rounds:
            break
        round_idx += 1

    valid_entries = _load_valid_summary_entries(output_dir, target_episode_ids=set(episode_indices))
    trajectory_reports, _ = _compute_reports(
        targets=targets,
        valid_entries=valid_entries,
        episode_attempts=episode_attempts,
        max_episode_attempts=args.max_episode_attempts,
    )
    episode_reports = _build_episode_reports(
        targets=targets,
        episode_meta=episode_meta,
        valid_entries=valid_entries,
        episode_attempts=episode_attempts,
        max_episode_attempts=args.max_episode_attempts,
    )
    selected_annotations = []
    target_episode_set = set(episode_indices)
    for idx in sorted(valid_entries):
        if idx not in target_episode_set:
            continue
        entry = valid_entries[idx]
        selected_annotations.append(
            {
                "id": int(entry["id"]),
                "trajectory_id": str(entry.get("trajectory_id", "")),
                "steps": int(entry["steps"]),
                "video": entry["video"],
                "instructions": entry.get("instructions", []),
                "actions": entry.get("actions", []),
                "scene_id": entry.get("scene_id"),
                "episode_id": entry.get("episode_id"),
            }
        )

    completed_trajectory_count = sum(1 for row in trajectory_reports if row["status"] == "completed")
    failed_trajectory_count = sum(1 for row in trajectory_reports if row["status"] == "failed")
    pending_trajectory_count = sum(1 for row in trajectory_reports if row["status"] == "pending")
    completed_episode_count = sum(1 for row in episode_reports if row["status"] == "completed")
    failed_episode_count = sum(1 for row in episode_reports if row["status"] == "failed")
    pending_episode_count = sum(1 for row in episode_reports if row["status"] == "pending")

    final_report = {
        "config": {
            "config": str(config_path),
            "recommended_json": str(recommended_path),
            "trajectory_groups_json": str(groups_path),
            "output_dir": str(output_dir),
            "max_rounds": args.max_rounds,
            "max_episode_attempts": args.max_episode_attempts,
            "trajectories_per_batch": args.trajectories_per_batch,
        },
        "summary": {
            "selected_trajectory_count": len(targets),
            "selected_episode_count": len(episode_indices),
            "completed_trajectory_count": completed_trajectory_count,
            "failed_trajectory_count": failed_trajectory_count,
            "pending_trajectory_count": pending_trajectory_count,
            "completed_episode_count": completed_episode_count,
            "failed_episode_count": failed_episode_count,
            "pending_episode_count": pending_episode_count,
            "all_trajectories_completed": completed_trajectory_count == len(targets),
        },
        "trajectories": trajectory_reports,
        "episodes": episode_reports,
    }

    _write_json(output_dir / "production_manifest.json", final_report)
    _write_json(output_dir / "annotations.json", selected_annotations)
    _write_json(pipeline_dir / "final_report.json", final_report)

    print(json.dumps(final_report["summary"], ensure_ascii=False, indent=2))
    if pending_trajectory_count > 0 or failed_trajectory_count > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

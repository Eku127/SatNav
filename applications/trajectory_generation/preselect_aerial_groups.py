#!/usr/bin/env python3
"""Preselect AerialSim trajectory groups and select high-quality ones.

This script is designed for long-running full-dataset screening and supports
resume by reading existing JSONL results in output_dir.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from applications.trajectory_generation.aerial_quality import (
    SCORE_FORMULA,
    compute_aerial_quality_score_from_row,
    format_aerial_quality_threshold_rule,
    passes_aerial_quality_thresholds,
)
from applications.trajectory_generation.runner import SatNavTrajectoryRunner


def load_preselect_items(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["preselect_list"]


def _load_jsonl_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_unique_outputs(
    results_jsonl: Path,
    failures_jsonl: Path,
) -> Tuple[List[Dict], List[Dict]]:
    # Success records win over failures for the same trajectory_group_key.
    success_by_key: Dict[str, Dict] = {}
    success_order: List[str] = []
    for row in _load_jsonl_rows(results_jsonl):
        key = row.get("trajectory_group_key")
        if not isinstance(key, str) or not key:
            continue
        if key not in success_by_key:
            success_order.append(key)
        success_by_key[key] = row

    failure_by_key: Dict[str, Dict] = {}
    failure_order: List[str] = []
    for row in _load_jsonl_rows(failures_jsonl):
        key = row.get("trajectory_group_key")
        if not isinstance(key, str) or not key or key in success_by_key:
            continue
        if key not in failure_by_key:
            failure_order.append(key)
        failure_by_key[key] = row

    successes = [success_by_key[key] for key in success_order]
    failures = [failure_by_key[key] for key in failure_order]
    return successes, failures


def load_done_keys(results_jsonl: Path, failures_jsonl: Path) -> Set[str]:
    successes, failures = load_unique_outputs(results_jsonl, failures_jsonl)
    done = {
        row["trajectory_group_key"]
        for row in successes + failures
        if isinstance(row.get("trajectory_group_key"), str) and row["trajectory_group_key"]
    }
    return done


def evaluate_one(
    runner: SatNavTrajectoryRunner,
    entry: Dict,
    samples_per_episode: int,
) -> Dict:
    ep_idx = int(entry["representative_episode_index"])
    ep = runner.dataset.episodes[ep_idx]
    metrics = runner._collect_episode_preselect_metrics(
        episode_idx=ep_idx,
        episode=ep,
        samples_per_episode=samples_per_episode,
    )
    if metrics is None:
        raise RuntimeError("no_metrics")

    return {
        **entry,
        **metrics,
        "score": compute_aerial_quality_score_from_row(metrics),
    }


def aggregate_outputs(
    output_dir: Path,
    min_std: float,
    max_black: float,
    min_height: float,
    min_score: Optional[float],
    source_preselect: str,
    config_path: str,
    samples_per_episode: int,
    total_candidates: int,
    elapsed_sec: float,
) -> None:
    results_jsonl = output_dir / "results.jsonl"
    failures_jsonl = output_dir / "failures.jsonl"

    successes, failures = load_unique_outputs(results_jsonl, failures_jsonl)

    successes.sort(key=lambda x: x["score"], reverse=True)

    recommended = []
    for row in successes:
        if passes_aerial_quality_thresholds(
            row,
            min_score=min_score,
            min_avg_std=min_std,
            max_avg_black_frac=max_black,
            min_avg_height_range=min_height,
        ):
            recommended.append(row)

    summary = {
        "source_preselect": source_preselect,
        "evaluated_count": len(successes) + len(failures),
        "success_count": len(successes),
        "failed_count": len(failures),
        "recommended_count": len(recommended),
        "total_candidates": total_candidates,
        "total_elapsed_sec": round(elapsed_sec, 2),
        "config": {
            "aerial_config": config_path,
            "samples_per_episode": samples_per_episode,
            "score": SCORE_FORMULA,
            "recommend_rule": format_aerial_quality_threshold_rule(
                min_avg_std=min_std,
                max_avg_black_frac=max_black,
                min_avg_height_range=min_height,
                min_score=min_score,
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluated_results.json").write_text(
        json.dumps(successes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recommended_top.json").write_text(
        json.dumps(recommended[:2000], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failed.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preselect AerialSim train trajectory groups with resume",
    )
    parser.add_argument(
        "--preselect_json",
        type=str,
        default="output/preselect_0404_train_trajectory/preselect_trajectory_list.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/satnav_task.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/aerialsim_train_traj_quality_eval_full",
    )
    parser.add_argument("--samples_per_episode", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="exclusive; -1 means all")
    parser.add_argument("--min_std", type=float, default=45.0)
    parser.add_argument("--max_black", type=float, default=0.002)
    parser.add_argument("--min_height", type=float, default=6.0)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--landmark_success", type=float, default=2.0)
    parser.add_argument("--flush_every", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    failures_jsonl = output_dir / "failures.jsonl"

    items = load_preselect_items(Path(args.preselect_json))
    total_candidates = len(items)

    end_idx = total_candidates if args.end < 0 else min(args.end, total_candidates)
    start_idx = max(0, min(args.start, end_idx))
    slice_items = items[start_idx:end_idx]

    done_keys = load_done_keys(results_jsonl, failures_jsonl)
    print(f"total_candidates={total_candidates}")
    print(f"eval_range=[{start_idx}, {end_idx}) count={len(slice_items)}")
    print(f"already_done={len(done_keys)}")

    runner = SatNavTrajectoryRunner(
        config_path=args.config,
        output_path=str(output_dir / "tmp_run"),
        landmark_success=args.landmark_success,
    )

    t_start = time.time()
    succ = 0
    fail = 0
    for idx, entry in enumerate(slice_items, start=1):
        key = entry["trajectory_group_key"]
        if key in done_keys:
            continue

        t0 = time.time()
        try:
            row = evaluate_one(
                runner=runner,
                entry=entry,
                samples_per_episode=args.samples_per_episode,
            )
            row["elapsed_sec"] = round(time.time() - t0, 2)
            with results_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done_keys.add(key)
            succ += 1
            print(
                f"[{idx}/{len(slice_items)}] ok score={row['score']:.2f} "
                f"std={row['avg_std']:.2f} h={row['avg_height_range']:.2f} "
                f"black={row['avg_black_frac']:.4f} key={key}"
            )
        except Exception as e:
            row = {
                "trajectory_group_key": key,
                "representative_episode_index": entry.get("representative_episode_index"),
                "reason": str(e),
                "elapsed_sec": round(time.time() - t0, 2),
            }
            with failures_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done_keys.add(key)
            fail += 1
            print(f"[{idx}/{len(slice_items)}] fail key={key} err={e}")

        if args.flush_every > 0 and (idx % args.flush_every == 0):
            aggregate_outputs(
                output_dir=output_dir,
                min_std=args.min_std,
                max_black=args.max_black,
                min_height=args.min_height,
                min_score=args.min_score,
                source_preselect=args.preselect_json,
                config_path=args.config,
                samples_per_episode=args.samples_per_episode,
                total_candidates=total_candidates,
                elapsed_sec=time.time() - t_start,
            )

    aggregate_outputs(
        output_dir=output_dir,
        min_std=args.min_std,
        max_black=args.max_black,
        min_height=args.min_height,
        min_score=args.min_score,
        source_preselect=args.preselect_json,
        config_path=args.config,
        samples_per_episode=args.samples_per_episode,
        total_candidates=total_candidates,
        elapsed_sec=time.time() - t_start,
    )
    print(f"done success={succ} fail={fail} elapsed={round(time.time() - t_start, 2)}s")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""Unified Random/ReferenceFollower/Seq2Seq/CMA evaluation command."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from baselines.classic.common.config import load_classic_config, resolve_path
from baselines.classic.factory import build_classic_adapter, normalize_method
from satnav.core.env import Env
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.evaluation import (
    BenchmarkManifest,
    EvaluationConfig,
    Evaluator,
    ManifestMismatchError,
    build_episode_plan,
    load_benchmark_manifest,
    payload_digest,
    referenced_scene_identity,
)


DEFAULT_CONFIGS = {
    "random": "configs/baselines/random_agent.yaml",
    "reference_follower": "configs/baselines/reference_follower.yaml",
    "seq2seq": "configs/baselines/seq2seq_eval.yaml",
    "cma": "configs/baselines/cma_eval.yaml",
}

METRIC_NAMES = {
    "DISTANCE_TO_GOAL": "distance_to_goal",
    "SUCCESS": "success",
    "ORACLE_SUCCESS": "oracle_success",
    "SPL": "spl",
    "PATH_LENGTH": "path_length",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("random", "reference", "reference_follower", "seq2seq", "cma"),
    )
    parser.add_argument("--config", help="experiment YAML; defaults by method")
    parser.add_argument("--benchmark", help="tracked benchmark JSON contract")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vocab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-stop-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fail-on-episode-error", action="store_true")
    parser.add_argument("--no-action-trace", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable OmegaConf override",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve config and exit before creating a dataset/model",
    )
    return parser


def _configured_split(config: DictConfig, requested: Optional[str]) -> str:
    split = requested or OmegaConf.select(config, "EVAL.SPLIT", default=None)
    split = split or OmegaConf.select(config, "DATASET.SPLIT", default=None)
    if not split:
        raise ValueError("split is missing; set --split or DATASET.SPLIT")
    split = str(split)
    OmegaConf.update(config, "EVAL.SPLIT", split, merge=False)
    OmegaConf.update(config, "DATASET.SPLIT", split, merge=False)
    return split


def _dynamic_benchmark(
    config: DictConfig, *, split: str, max_steps: int
) -> BenchmarkManifest:
    measurements = OmegaConf.select(config, "TASK.MEASUREMENTS", default=()) or ()
    required_metrics = tuple(
        METRIC_NAMES[str(name).upper()]
        for name in measurements
        if str(name).upper() in METRIC_NAMES
    )
    action_space = tuple(
        str(action)
        for action in (
            OmegaConf.select(config, "TASK.POSSIBLE_ACTIONS", default=()) or ()
        )
    )
    rgb = OmegaConf.select(config, "SIMULATOR.RGB_SENSOR", default={}) or {}
    observation = {
        "rgb": {
            "width": int(OmegaConf.select(rgb, "WIDTH", default=0)),
            "height": int(OmegaConf.select(rgb, "HEIGHT", default=0)),
            "hfov": float(OmegaConf.select(rgb, "HFOV", default=0)),
        }
    }
    success_threshold = OmegaConf.select(
        config, "TASK.SUCCESS_DISTANCE", default=None
    )
    if success_threshold is not None and not isinstance(
        success_threshold, (str, int, float)
    ):
        success_threshold = OmegaConf.to_container(
            success_threshold, resolve=True
        )
    dataset_version = str(
        OmegaConf.select(config, "DATASET.VERSION", default="unspecified")
    )
    return BenchmarkManifest(
        benchmark_id=f"satnav-{dataset_version}-{split}-{max_steps}-step",
        dataset_version=dataset_version,
        split=split,
        kind="smoke",
        action_space=action_space,
        forward_step_size=float(
            OmegaConf.select(config, "SIMULATOR.FORWARD_STEP_SIZE", default=0)
        ),
        turn_angle=float(
            OmegaConf.select(config, "SIMULATOR.TURN_ANGLE", default=0)
        ),
        observation=observation,
        max_episode_steps=max_steps,
        success_threshold=success_threshold,
        required_metrics=required_metrics,
        metadata={"source": "resolved-config"},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path_id(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _artifact_identity(path: Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    return {
        "id": resolved.name,
        "sha256": _file_sha256(resolved),
        "size": resolved.stat().st_size,
    }


def _directory_identity(path: Path) -> Mapping[str, Any]:
    """Return a portable, exact identity for every regular file below ``path``."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    entries = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"artifact directory contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        entries.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "sha256": _file_sha256(candidate),
                "size": candidate.stat().st_size,
            }
        )
    if not entries:
        raise ValueError(f"artifact directory contains no files: {root}")
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "digest": payload_digest({"files": entries}),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
    }


def _resolve_dataset_paths(config: DictConfig, split: str) -> Path:
    configured = str(OmegaConf.select(config, "DATASET.DATA_PATH"))
    dataset_path = resolve_path(configured.format(split=split))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset artifact does not exist: {dataset_path}")
    OmegaConf.update(config, "DATASET.DATA_PATH", str(dataset_path), merge=False)
    scenes = OmegaConf.select(config, "DATASET.SCENES_DIR", default=None)
    if scenes is not None:
        OmegaConf.update(
            config,
            "DATASET.SCENES_DIR",
            str(resolve_path(str(scenes))),
            merge=False,
        )
    return dataset_path


def _resolved_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _path_free_config(value: Any, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _path_free_config(item, str(name))
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_path_free_config(item) for item in value]
    upper = key.upper()
    if isinstance(value, str) and (
        upper.endswith(("_PATH", "_DIR", "_FILE", "_ROOT", "_FOLDER"))
        or upper in {"RESULTS_DIR", "VIDEO_DIR"}
    ):
        return {"path_role": upper.lower()}
    return value


def _benchmark_config_contract(config: DictConfig) -> Mapping[str, Any]:
    return {
        "action_space": list(
            OmegaConf.select(config, "TASK.POSSIBLE_ACTIONS", default=()) or ()
        ),
        "forward_step_size": float(
            OmegaConf.select(config, "SIMULATOR.FORWARD_STEP_SIZE")
        ),
        "turn_angle": float(OmegaConf.select(config, "SIMULATOR.TURN_ANGLE")),
        "observation": {
            "rgb": {
                "width": int(
                    OmegaConf.select(config, "SIMULATOR.RGB_SENSOR.WIDTH")
                ),
                "height": int(
                    OmegaConf.select(config, "SIMULATOR.RGB_SENSOR.HEIGHT")
                ),
                "hfov": float(
                    OmegaConf.select(config, "SIMULATOR.RGB_SENSOR.HFOV")
                ),
            }
        },
        "max_episode_steps": int(
            OmegaConf.select(config, "ENVIRONMENT.MAX_EPISODE_STEPS")
        ),
        "success_threshold": _resolved_value(
            OmegaConf.select(config, "TASK.SUCCESS_DISTANCE", default=None)
        ),
        "required_metrics": [
            METRIC_NAMES.get(str(name).upper(), str(name).lower())
            for name in (
                OmegaConf.select(config, "TASK.MEASUREMENTS", default=()) or ()
            )
            if str(name).upper() in METRIC_NAMES
        ],
    }


def _validate_benchmark_config(
    config: DictConfig, benchmark: BenchmarkManifest
) -> None:
    contract = _benchmark_config_contract(config)
    if benchmark.action_space and tuple(contract["action_space"]) != tuple(
        benchmark.action_space
    ):
        raise ManifestMismatchError("task action space does not match benchmark")
    for name in ("forward_step_size", "turn_angle"):
        declared = getattr(benchmark, name)
        if declared is not None and float(contract[name]) != float(declared):
            raise ManifestMismatchError(f"{name} does not match benchmark")
    declared_rgb = (
        benchmark.observation.get("rgb", {}) if benchmark.observation else {}
    )
    actual_rgb = contract["observation"]["rgb"]
    for name, expected in declared_rgb.items():
        if name in actual_rgb and float(actual_rgb[name]) != float(expected):
            raise ManifestMismatchError(f"rgb {name} does not match benchmark")
    actual_metrics = set(contract["required_metrics"])
    missing_metrics = [
        name for name in benchmark.required_metrics if name not in actual_metrics
    ]
    if missing_metrics:
        raise ManifestMismatchError(
            f"task config is missing benchmark metrics: {missing_metrics}"
        )
    declared_threshold = benchmark.success_threshold
    actual_threshold = contract["success_threshold"]
    if isinstance(declared_threshold, Mapping):
        if not isinstance(actual_threshold, Mapping):
            raise ManifestMismatchError("success threshold does not match benchmark")
        for name, expected in declared_threshold.items():
            if name not in actual_threshold or float(actual_threshold[name]) != float(
                expected
            ):
                raise ManifestMismatchError(
                    f"success threshold {name} does not match benchmark"
                )
    elif declared_threshold is not None:
        comparable = (
            actual_threshold.get("DEFAULT")
            if isinstance(actual_threshold, Mapping)
            else actual_threshold
        )
        if comparable is None or float(comparable) != float(declared_threshold):
            raise ManifestMismatchError("success threshold does not match benchmark")


def _verify_dataset_artifact(
    dataset_path: Path, benchmark: BenchmarkManifest
) -> Mapping[str, Any]:
    identity = _artifact_identity(dataset_path)
    expected = benchmark.dataset_digest
    if expected is not None and identity["sha256"] != str(expected).lower():
        raise ManifestMismatchError(
            "Benchmark dataset_digest does not match the loaded episode artifact"
        )
    return identity


def _run_metadata(
    *,
    method: str,
    config_path: Path,
    config: DictConfig,
    benchmark_path: Optional[Path],
    episode_identity: Mapping[str, Any],
    min_stop_steps: int,
    scene_identity: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    task_contract = _benchmark_config_contract(config)
    semantic_config = _path_free_config(
        OmegaConf.to_container(config, resolve=True)
    )
    scenes_dir = OmegaConf.select(config, "DATASET.SCENES_DIR", default=None)
    if scenes_dir is None:
        raise ValueError("DATASET.SCENES_DIR is required for reproducible evaluation")
    return {
        "entrypoint": "baselines.classic.evaluate",
        "method": method,
        "config": {
            "id": _portable_path_id(config_path),
            "contract_digest": payload_digest({"task_contract": task_contract}),
            "semantic_digest": payload_digest({"config": semantic_config}),
        },
        "benchmark": (
            "dynamic" if benchmark_path is None else Path(benchmark_path).name
        ),
        "episodes": dict(episode_identity),
        "scenes": (
            dict(scene_identity)
            if scene_identity is not None
            else _directory_identity(Path(str(scenes_dir)))
        ),
        "min_stop_steps": int(min_stop_steps),
        "task_contract": task_contract,
    }


def _resolved_output_dir(
    config: DictConfig, method: str, split: str, requested: Optional[Path]
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    configured = OmegaConf.select(config, "RESULTS_DIR", default=None)
    if configured:
        return resolve_path(str(configured))
    return resolve_path(f"output/baselines/classic/{method}/{split}")


def _print_resolved(
    *,
    method: str,
    config_path: Path,
    split: str,
    config: DictConfig,
    benchmark_path: Optional[Path],
) -> None:
    payload = {
        "method": method,
        "config_path": str(config_path),
        "benchmark_path": None if benchmark_path is None else str(benchmark_path),
        "split": split,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_from_args(args: argparse.Namespace) -> Mapping[str, Any]:
    method = normalize_method(args.method)
    config_path = resolve_path(args.config or DEFAULT_CONFIGS[method])
    config = load_classic_config(config_path, overrides=tuple(args.overrides))
    split = _configured_split(config, args.split)
    benchmark_path = (
        None if args.benchmark is None else resolve_path(args.benchmark)
    )
    if args.print_config:
        _print_resolved(
            method=method,
            config_path=config_path,
            split=split,
            config=config,
            benchmark_path=benchmark_path,
        )
        return {"status": "config_only"}

    if benchmark_path is not None:
        benchmark = load_benchmark_manifest(benchmark_path)
        benchmark_max_steps = benchmark.max_episode_steps
        max_steps = int(
            args.max_steps
            if args.max_steps is not None
            else benchmark_max_steps
            if benchmark_max_steps is not None
            else OmegaConf.select(
                config, "ENVIRONMENT.MAX_EPISODE_STEPS", default=500
            )
        )
        if (
            benchmark_max_steps is not None
            and max_steps != int(benchmark_max_steps)
        ):
            raise ValueError(
                "--max-steps contradicts the benchmark contract: "
                f"{max_steps} != {benchmark_max_steps}"
            )
    else:
        max_steps = int(
            args.max_steps
            if args.max_steps is not None
            else OmegaConf.select(
                config, "ENVIRONMENT.MAX_EPISODE_STEPS", default=500
            )
        )
        benchmark = _dynamic_benchmark(config, split=split, max_steps=max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    OmegaConf.update(
        config, "ENVIRONMENT.MAX_EPISODE_STEPS", max_steps, merge=False
    )

    dataset_path = _resolve_dataset_paths(config, split)
    _validate_benchmark_config(config, benchmark)
    episode_identity = _verify_dataset_artifact(dataset_path, benchmark)
    dataset = SatNavDataset(config.DATASET)
    selected_plan = build_episode_plan(
        dataset.episodes,
        split=split,
        offset=args.offset,
        limit=args.limit,
    )
    scene_identity = referenced_scene_identity(
        Path(str(config.DATASET.SCENES_DIR)),
        (item.episode for item in selected_plan.selected),
    )
    environment = Env(config, dataset=dataset, cycle=False)
    try:
        bundle = build_classic_adapter(
            method,
            config,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            device=args.device,
            deterministic=True,
            min_stop_steps=args.min_stop_steps,
        )
    except Exception:
        environment.close()
        raise

    output_dir = _resolved_output_dir(
        bundle.config, method, split, args.output_dir
    )
    evaluator = Evaluator(
        environment=environment,
        policy=bundle.adapter,
        config=EvaluationConfig(
            output_dir=output_dir,
            split=split,
            policy_id=bundle.policy_id,
            offset=args.offset,
            limit=args.limit,
            rank=args.rank,
            world_size=args.world_size,
            base_seed=args.seed,
            max_steps=max_steps,
            resume=args.resume,
            fail_fast=args.fail_fast,
            fail_on_episode_error=args.fail_on_episode_error,
            capture_action_trace=not args.no_action_trace,
            aggregate_single_rank=True,
            policy_metadata=bundle.policy_metadata,
            run_metadata=_run_metadata(
                method=method,
                config_path=config_path,
                config=bundle.config,
                benchmark_path=benchmark_path,
                episode_identity=episode_identity,
                min_stop_steps=args.min_stop_steps,
                scene_identity=scene_identity,
            ),
        ),
        benchmark=benchmark,
    )
    summary = evaluator.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_from_args(args)
    except Exception as error:
        print(f"classic evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

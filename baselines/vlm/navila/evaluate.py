"""Evaluate NaVILA through SatNav's generic rollout/result contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from satnav.evaluation import (
    EvaluationConfig,
    Evaluator,
    ManifestMismatchError,
    load_benchmark_manifest,
    referenced_scene_identity,
)

from baselines.vlm.navila.adapter import NaVILAPolicyAdapter
from baselines.vlm.navila.artifacts import file_identity, navila_model_identity
from baselines.vlm.navila.bootstrap import (
    NAVILA_REVISION,
    bootstrap_navila,
    source_revision,
)


BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]
DEFAULT_TASK_CONFIG = BASELINE_DIR / "configs" / "satnav_task.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NaVILA with SatNav PolicyAdapter + JSONL/resume manifests"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--navila-repo", type=Path)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument(
        "--split", default="val_seen", choices=("val_seen", "val_unseen", "test")
    )
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fail-on-episode-error", action="store_true")
    parser.add_argument("--no-action-trace", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--action-format", default="compact", choices=("compact", "sentence")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16",))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    return fallback if value is None else int(value)


def _default_benchmark(split: str) -> Path:
    return SATNAV_ROOT / "configs" / "benchmark" / f"satnav_v0_1_{split}.json"


def _resolved_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_benchmark_config(config: Any, benchmark: Any) -> None:
    from omegaconf import OmegaConf

    actions = tuple(OmegaConf.select(config, "TASK.POSSIBLE_ACTIONS", default=()))
    if benchmark.action_space and actions != tuple(benchmark.action_space):
        raise ValueError(
            f"task action space {actions} != benchmark {benchmark.action_space}"
        )
    for key, declared in (
        ("SIMULATOR.FORWARD_STEP_SIZE", benchmark.forward_step_size),
        ("SIMULATOR.TURN_ANGLE", benchmark.turn_angle),
    ):
        if declared is not None and float(OmegaConf.select(config, key)) != float(
            declared
        ):
            raise ValueError(f"{key} does not match benchmark {declared}")
    rgb = benchmark.observation.get("rgb", {}) if benchmark.observation else {}
    for config_key, manifest_key in (
        ("SIMULATOR.RGB_SENSOR.WIDTH", "width"),
        ("SIMULATOR.RGB_SENSOR.HEIGHT", "height"),
        ("SIMULATOR.RGB_SENSOR.HFOV", "hfov"),
    ):
        if manifest_key in rgb and float(OmegaConf.select(config, config_key)) != float(
            rgb[manifest_key]
        ):
            raise ValueError(f"{config_key} does not match benchmark")
    measurements = {
        str(value).lower()
        for value in OmegaConf.select(config, "TASK.MEASUREMENTS", default=())
    }
    missing = [
        name for name in benchmark.required_metrics if name.lower() not in measurements
    ]
    if missing:
        raise ValueError(f"task config is missing benchmark measures: {missing}")
    if isinstance(benchmark.success_threshold, Mapping):
        thresholds = OmegaConf.to_container(
            OmegaConf.select(config, "TASK.SUCCESS_DISTANCE"), resolve=True
        )
        for name, expected in benchmark.success_threshold.items():
            if name not in thresholds or float(thresholds[name]) != float(expected):
                raise ValueError(f"TASK.SUCCESS_DISTANCE.{name} mismatch")


def _build_environment(args: argparse.Namespace, benchmark: Any, max_steps: int):
    from omegaconf import OmegaConf
    from satnav.core.env import Env
    from satnav.dataset.satnav_dataset import SatNavDataset

    task_path = Path(args.task_config).expanduser().resolve()
    config = OmegaConf.load(task_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = args.split
    episodes = (
        args.episodes
        or os.environ.get("SATNAV_NAVILA_EVAL_EPISODES")
        or config.DATASET.DATA_PATH
    )
    scenes = (
        args.scenes_dir
        or os.environ.get("SATNAV_NAVILA_SCENES_DIR")
        or config.DATASET.SCENES_DIR
    )
    config.DATASET.DATA_PATH = str(
        _resolved_path(str(episodes).format(split=args.split), SATNAV_ROOT)
    )
    config.DATASET.SCENES_DIR = str(_resolved_path(scenes, SATNAV_ROOT))
    config.ENVIRONMENT.MAX_EPISODE_STEPS = int(max_steps)
    OmegaConf.set_struct(config, True)
    _validate_benchmark_config(config, benchmark)
    dataset = SatNavDataset(config.DATASET)
    return config, dataset, Env(config, dataset=dataset, cycle=False)


def _policy_metadata(args: argparse.Namespace) -> Mapping[str, Any]:
    model = Path(args.model_path).expanduser().resolve()
    identity = navila_model_identity(model)
    return {
        "checkpoint_id": args.checkpoint_id or model.name,
        "checkpoint_digest": identity["digest"],
        "checkpoint_identity": identity,
        "upstream_revision": NAVILA_REVISION,
        "num_frames": args.num_frames,
        "max_new_tokens": args.max_new_tokens,
        "action_format": args.action_format,
        "dtype": args.dtype,
        "generation": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "use_cache": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rank = args.rank if args.rank is not None else _env_int("RANK", 0)
    world_size = (
        args.world_size if args.world_size is not None else _env_int("WORLD_SIZE", 1)
    )
    local_rank = (
        args.local_rank if args.local_rank is not None else _env_int("LOCAL_RANK", rank)
    )
    benchmark_path = Path(
        args.benchmark_manifest or _default_benchmark(args.split)
    ).resolve()
    benchmark = load_benchmark_manifest(benchmark_path)
    max_steps = args.max_steps or benchmark.max_episode_steps
    if max_steps is None or int(max_steps) <= 0:
        raise ValueError("max steps must be positive")
    config, dataset, environment = _build_environment(args, benchmark, int(max_steps))
    episode_identity = file_identity(Path(str(config.DATASET.DATA_PATH)))
    if (
        benchmark.dataset_digest is not None
        and episode_identity["sha256"] != benchmark.dataset_digest
    ):
        environment.close()
        raise ManifestMismatchError(
            "Benchmark dataset_digest does not match loaded episodes"
        )
    metadata = _policy_metadata(args)
    output = args.output_dir or (
        SATNAV_ROOT
        / "output"
        / "baselines"
        / "vlm"
        / "navila"
        / str(metadata["checkpoint_id"])
        / args.split
    )
    output = Path(output).expanduser().resolve()
    from satnav.evaluation import build_episode_plan

    complete_plan = build_episode_plan(dataset.episodes, split=args.split)
    selected_plan = build_episode_plan(
        dataset.episodes,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    benchmark.resolved(
        split=args.split,
        episode_count=len(complete_plan.selected),
        episode_digest=complete_plan.selected_digest,
        max_episode_steps=int(max_steps),
    )
    if args.dry_run:
        plan = build_episode_plan(
            dataset.episodes,
            split=args.split,
            offset=args.offset,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
        )
        environment.close()
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dataset_episode_count": len(dataset.episodes),
                    "selected_episode_count": len(plan.selected),
                    "rank_episode_count": len(plan.shard),
                    "selected_episode_digest": plan.selected_digest,
                    "checkpoint_digest": metadata["checkpoint_digest"],
                    "output_dir": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    scene_identity = referenced_scene_identity(
        Path(str(config.DATASET.SCENES_DIR)),
        (item.episode for item in selected_plan.selected),
    )

    repo = bootstrap_navila(
        args.navila_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    metadata = dict(metadata, upstream_revision=source_revision(repo) or "unknown")
    device = args.device
    if device == "cuda":
        device = f"cuda:{local_rank}"
    policy = NaVILAPolicyAdapter.from_pretrained(
        args.model_path,
        navila_repo=repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
        device=device,
        dtype=args.dtype,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        action_format=args.action_format,
    )
    evaluator = Evaluator(
        environment=environment,
        policy=policy,
        config=EvaluationConfig(
            output_dir=output,
            split=args.split,
            policy_id=f"navila:{metadata['checkpoint_id']}",
            offset=args.offset,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
            base_seed=args.base_seed,
            max_steps=int(max_steps),
            resume=args.resume,
            fail_fast=args.fail_fast,
            fail_on_episode_error=args.fail_on_episode_error,
            capture_action_trace=not args.no_action_trace,
            aggregate_single_rank=True,
            policy_metadata=metadata,
            run_metadata={
                "baseline": "navila",
                "benchmark": benchmark_path.name,
                "task_config": file_identity(Path(args.task_config)),
                "episodes": episode_identity,
                "scenes": scene_identity,
            },
        ),
        benchmark=benchmark,
    )
    print(json.dumps(evaluator.run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

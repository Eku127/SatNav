"""Evaluate StreamVLN through SatNav's generic rollout/result contract."""

from __future__ import annotations

import argparse
import hashlib
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

from baselines.vlm.streamvln.adapter import StreamVLNPolicyAdapter
from baselines.vlm.streamvln.bootstrap import (
    STREAMVLN_REVISION,
    bootstrap_streamvln,
    source_revision,
)


BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]
DEFAULT_TASK_CONFIG = BASELINE_DIR / "configs" / "satnav_task.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate StreamVLN with SatNav PolicyAdapter + JSONL/resume manifests"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--vision-tower")
    parser.add_argument("--vision-tower-digest")
    parser.add_argument("--streamvln-repo", type=Path)
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
    parser.add_argument("--checkpoint-digest")
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
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths/config/benchmark and dataset identity without loading weights",
    )
    return parser


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    return fallback if value is None else int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, option: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{option} must be a 64-character SHA-256")
    return digest


def _artifact_manifest(root: Path, paths: Sequence[Path]) -> Mapping[str, Any]:
    """Hash named local artifacts and return a path-independent identity.

    The aggregate digest commits to each relative filename, byte length, and
    content digest.  Keeping the per-file entries in the run manifest makes a
    checkpoint identity independently auditable without recording machine
    paths.
    """

    root = Path(root).expanduser().resolve()
    entries = []
    logical_paths = {
        Path(os.path.abspath(str(Path(value).expanduser()))) for value in paths
    }
    for path in sorted(logical_paths):
        if not path.is_file():
            raise FileNotFoundError(f"identity artifact not found: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"identity artifact is outside {root}: {path}") from error
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not entries:
        raise ValueError(f"no identity artifacts found below {root}")
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "digest": hashlib.sha256(encoded).hexdigest(),
        "algorithm": "sha256-file-manifest-v1",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "files": entries,
    }


def _checkpoint_artifacts(model_path: Path) -> Sequence[Path]:
    """Resolve every model/config artifact consumed by ``from_pretrained``."""

    model_path = Path(model_path).expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")

    artifacts = {config_path}
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        artifacts.add(generation_config)

    index_paths = tuple(
        path
        for path in (
            model_path / "model.safetensors.index.json",
            model_path / "pytorch_model.bin.index.json",
        )
        if path.is_file()
    )
    weight_paths = set()
    for index_path in index_paths:
        artifacts.add(index_path)
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"checkpoint index has no weight_map: {index_path}")
        for filename in weight_map.values():
            candidate = Path(os.path.abspath(str(model_path / str(filename))))
            try:
                candidate.relative_to(model_path)
            except ValueError as error:
                raise ValueError(
                    f"checkpoint index references a path outside {model_path}: {filename}"
                ) from error
            weight_paths.add(candidate)

    if not index_paths:
        for pattern in (
            "*.safetensors",
            "pytorch_model*.bin",
            "adapter_model*.bin",
            "mm_projector*.bin",
            "non_lora_trainables*.bin",
        ):
            weight_paths.update(model_path.glob(pattern))
    if not weight_paths:
        raise FileNotFoundError(f"no checkpoint weight artifacts found in {model_path}")
    artifacts.update(weight_paths)
    return tuple(artifacts)


def _tokenizer_artifacts(tokenizer_path: Path) -> Sequence[Path]:
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    config_path = tokenizer_path / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"tokenizer config not found: {config_path}")
    names = (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
    )
    return tuple(tokenizer_path / name for name in names if (tokenizer_path / name).is_file())


def _vision_tower_artifacts(tower_path: Path) -> Sequence[Path]:
    """Return the complete local snapshot consumed by the vision stack.

    Unlike a language-model checkpoint, a vision snapshot also contains image
    processor configuration.  Hash every regular file so pixel preprocessing
    changes cannot silently pass resume validation.
    """

    tower_path = Path(tower_path).expanduser().resolve()
    artifacts = tuple(path for path in tower_path.rglob("*") if path.is_file())
    if not artifacts:
        raise FileNotFoundError(f"no vision-tower artifacts found in {tower_path}")
    return artifacts


def _checkpoint_metadata(args: argparse.Namespace) -> Mapping[str, Any]:
    model_path = Path(args.model_path).expanduser().resolve()
    checkpoint_id = args.checkpoint_id or model_path.name
    if args.checkpoint_digest:
        checkpoint_digest = _validated_sha256(
            args.checkpoint_digest, "--checkpoint-digest"
        )
        checkpoint_identity: Mapping[str, Any] = {
            "digest": checkpoint_digest,
            "algorithm": "user-supplied-sha256",
            "scope": "complete-checkpoint",
        }
    else:
        checkpoint_identity = dict(
            _artifact_manifest(model_path, _checkpoint_artifacts(model_path)),
            scope="from-pretrained-model-and-config-artifacts",
        )
    metadata = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_digest": checkpoint_identity["digest"],
        "checkpoint_identity": checkpoint_identity,
        "upstream_revision": STREAMVLN_REVISION,
        "num_frames": args.num_frames,
        "num_history": args.num_history,
        "num_future_steps": args.num_future_steps,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "generation": {"do_sample": False, "num_beams": 1, "use_cache": True},
    }
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    tokenizer_identity = _artifact_manifest(
        tokenizer_path, _tokenizer_artifacts(tokenizer_path)
    )
    metadata["tokenizer_id"] = tokenizer_path.name
    metadata["tokenizer_identity"] = tokenizer_identity
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    configured_tower_value = args.vision_tower or model_config.get(
        "mm_vision_tower", model_config.get("vision_tower")
    )
    if configured_tower_value:
        configured_tower = Path(str(configured_tower_value)).expanduser()
        metadata["vision_tower_id"] = (
            configured_tower.name
            if configured_tower.is_absolute() or configured_tower.exists()
            else str(configured_tower_value)
        )
        if args.vision_tower_digest:
            tower_identity: Mapping[str, Any] = {
                "digest": _validated_sha256(
                    args.vision_tower_digest, "--vision-tower-digest"
                ),
                "algorithm": "user-supplied-sha256",
                "scope": "complete-vision-tower",
            }
        elif configured_tower.is_dir():
            tower_root = configured_tower.resolve()
            tower_identity = dict(
                _artifact_manifest(tower_root, _vision_tower_artifacts(tower_root)),
                scope="complete-local-vision-tower-snapshot",
            )
        else:
            raise ValueError(
                "Vision-tower resume identity requires a local --vision-tower "
                "directory or --vision-tower-digest"
            )
        metadata["vision_tower_identity"] = tower_identity
    return metadata


def _resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    model_path = Path(args.model_path).expanduser().resolve()
    candidates = []
    if args.tokenizer_path is not None:
        candidates.append(Path(args.tokenizer_path).expanduser())
    elif (model_path / "tokenizer_config.json").is_file():
        candidates.append(model_path)
    else:
        configured = os.environ.get("STREAMVLN_TOKENIZER_PATH")
        if configured:
            candidates.append(Path(configured).expanduser())
        model_root = os.environ.get("STREAMVLN_MODEL_ROOT")
        if model_root:
            candidates.append(Path(model_root).expanduser() / "LLaVA-Video-7B-Qwen2")
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "tokenizer_config.json").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates) or "none"
    raise FileNotFoundError(
        "No tokenizer_config.json found. Pass --tokenizer-path or set "
        f"STREAMVLN_TOKENIZER_PATH. Checked: {rendered}"
    )


def _run_metadata(
    args: argparse.Namespace,
    benchmark_path: Path,
    episode_identity: Optional[Mapping[str, Any]] = None,
    scene_identity: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    task_config = Path(args.task_config).expanduser().resolve()
    metadata = {
        "baseline": "streamvln",
        "task_config": {
            "id": task_config.name,
            "sha256": _file_sha256(task_config),
        },
        "benchmark": Path(benchmark_path).name,
    }
    if episode_identity is not None:
        metadata["episodes"] = dict(episode_identity)
    if scene_identity is not None:
        metadata["scenes"] = dict(scene_identity)
    return metadata


def _episode_artifact_identity(
    episode_path: Path, expected_digest: Optional[str]
) -> Mapping[str, Any]:
    episode_path = Path(episode_path).expanduser().resolve()
    actual_digest = _file_sha256(episode_path)
    if expected_digest is not None and actual_digest != str(expected_digest).lower():
        raise ManifestMismatchError(
            "Benchmark dataset_digest does not match the loaded episode artifact"
        )
    return {
        "id": episode_path.name,
        "sha256": actual_digest,
        "size": episode_path.stat().st_size,
    }


def _default_benchmark(split: str) -> Path:
    return SATNAV_ROOT / "configs" / "benchmark" / f"satnav_v0_1_{split}.json"


def _resolved_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_benchmark_config(config: Any, benchmark: Any) -> None:
    """Validate declarations the generic manifest intentionally does not inspect."""

    from omegaconf import OmegaConf

    actions = tuple(OmegaConf.select(config, "TASK.POSSIBLE_ACTIONS", default=()))
    if benchmark.action_space and actions != tuple(benchmark.action_space):
        raise ValueError(
            f"task action space {actions} != benchmark {benchmark.action_space}"
        )
    checks = (
        ("SIMULATOR.FORWARD_STEP_SIZE", benchmark.forward_step_size),
        ("SIMULATOR.TURN_ANGLE", benchmark.turn_angle),
    )
    for key, declared in checks:
        if declared is None:
            continue
        actual = float(OmegaConf.select(config, key))
        if actual != float(declared):
            raise ValueError(f"{key}={actual} != benchmark {declared}")
    rgb = benchmark.observation.get("rgb", {}) if benchmark.observation else {}
    for config_key, manifest_key in (
        ("SIMULATOR.RGB_SENSOR.WIDTH", "width"),
        ("SIMULATOR.RGB_SENSOR.HEIGHT", "height"),
        ("SIMULATOR.RGB_SENSOR.HFOV", "hfov"),
    ):
        if manifest_key in rgb:
            actual = float(OmegaConf.select(config, config_key))
            if actual != float(rgb[manifest_key]):
                raise ValueError(
                    f"{config_key}={actual} != benchmark {rgb[manifest_key]}"
                )
    measurements = {
        str(value).lower()
        for value in OmegaConf.select(config, "TASK.MEASUREMENTS", default=())
    }
    missing = [name for name in benchmark.required_metrics if name.lower() not in measurements]
    if missing:
        raise ValueError(f"task config is missing benchmark measures: {missing}")
    if benchmark.success_threshold is not None:
        actual_thresholds = OmegaConf.to_container(
            OmegaConf.select(config, "TASK.SUCCESS_DISTANCE"), resolve=True
        )
        declared_thresholds = benchmark.success_threshold
        if isinstance(declared_thresholds, Mapping):
            for name, value in declared_thresholds.items():
                if name not in actual_thresholds or float(actual_thresholds[name]) != float(value):
                    raise ValueError(
                        f"TASK.SUCCESS_DISTANCE.{name} does not match benchmark {value}"
                    )


def _build_environment(args: argparse.Namespace, benchmark: Any, max_steps: int):
    from omegaconf import OmegaConf
    from satnav.core.env import Env
    from satnav.dataset.satnav_dataset import SatNavDataset

    task_path = Path(args.task_config).expanduser().resolve()
    config = OmegaConf.load(task_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = args.split
    configured_episodes = args.episodes or os.environ.get(
        "SATNAV_STREAMVLN_EVAL_EPISODES"
    ) or config.DATASET.DATA_PATH
    configured_scenes = args.scenes_dir or os.environ.get(
        "SATNAV_STREAMVLN_SCENES_DIR"
    ) or config.DATASET.SCENES_DIR
    episode_path = str(configured_episodes).format(split=args.split)
    config.DATASET.DATA_PATH = str(_resolved_path(episode_path, SATNAV_ROOT))
    config.DATASET.SCENES_DIR = str(_resolved_path(configured_scenes, SATNAV_ROOT))
    config.ENVIRONMENT.MAX_EPISODE_STEPS = int(max_steps)
    OmegaConf.set_struct(config, True)
    _validate_benchmark_config(config, benchmark)

    dataset = SatNavDataset(config.DATASET)
    environment = Env(config, dataset=dataset, cycle=False)
    return config, dataset, environment


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rank = args.rank if args.rank is not None else _env_int("RANK", 0)
    world_size = (
        args.world_size if args.world_size is not None else _env_int("WORLD_SIZE", 1)
    )
    local_rank = (
        args.local_rank if args.local_rank is not None else _env_int("LOCAL_RANK", rank)
    )
    benchmark_path = args.benchmark_manifest or _default_benchmark(args.split)
    benchmark = load_benchmark_manifest(benchmark_path)
    max_steps = (
        args.max_steps if args.max_steps is not None else benchmark.max_episode_steps
    )
    if max_steps is None or int(max_steps) <= 0:
        raise ValueError("max steps must be positive in CLI or benchmark manifest")

    config, dataset, environment = _build_environment(args, benchmark, int(max_steps))
    episode_identity = _episode_artifact_identity(
        Path(str(config.DATASET.DATA_PATH)), benchmark.dataset_digest
    )
    args.tokenizer_path = _resolve_tokenizer_path(args)
    metadata = _checkpoint_metadata(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            SATNAV_ROOT
            / "output"
            / "baselines"
            / "vlm"
            / "streamvln"
            / str(metadata["checkpoint_id"])
            / args.split
        )
    output_dir = Path(output_dir).expanduser().resolve()

    from satnav.evaluation import build_episode_plan

    dataset_plan = build_episode_plan(dataset.episodes, split=args.split)
    selected_plan = build_episode_plan(
        dataset.episodes,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    benchmark.resolved(
        split=args.split,
        episode_count=len(dataset_plan.selected),
        episode_digest=dataset_plan.selected_digest,
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
                    "output_dir": str(output_dir),
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

    streamvln_repo = bootstrap_streamvln(
        args.streamvln_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    revision = source_revision(streamvln_repo)
    metadata = dict(metadata, upstream_revision=revision or "unknown")
    device = args.device
    if device == "cuda":
        device = f"cuda:{local_rank}"
    policy = StreamVLNPolicyAdapter.from_pretrained(
        args.model_path,
        tokenizer_path=args.tokenizer_path,
        vision_tower=args.vision_tower,
        streamvln_repo=streamvln_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
        device=device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        num_frames=args.num_frames,
        num_history=args.num_history,
        num_future_steps=args.num_future_steps,
        max_new_tokens=args.max_new_tokens,
    )
    evaluator = Evaluator(
        environment=environment,
        policy=policy,
        config=EvaluationConfig(
            output_dir=output_dir,
            split=args.split,
            policy_id=f"streamvln:{metadata['checkpoint_id']}",
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
            run_metadata=_run_metadata(
                args, benchmark_path, episode_identity, scene_identity
            ),
        ),
        benchmark=benchmark,
    )
    result = evaluator.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

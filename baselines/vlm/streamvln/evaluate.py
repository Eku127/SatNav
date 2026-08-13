"""Evaluate StreamVLN through SatNav's generic rollout/result contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from satnav.evaluation import EvaluationConfig, Evaluator, build_episode_plan

from baselines.vlm.streamvln.adapter import StreamVLNPolicyAdapter
from baselines.vlm.streamvln.bootstrap import bootstrap_streamvln


BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]
DEFAULT_TASK_CONFIG = BASELINE_DIR / "configs" / "satnav_task.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate StreamVLN with SatNav PolicyAdapter and resumable JSONL"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--vision-tower")
    parser.add_argument("--streamvln-repo", type=Path)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
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
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    return fallback if value is None else int(value)


def _resolved_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


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


def _build_environment(args: argparse.Namespace):
    from omegaconf import OmegaConf
    from satnav.core.env import Env
    from satnav.dataset.satnav_dataset import SatNavDataset

    config = OmegaConf.load(Path(args.task_config).expanduser().resolve())
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = args.split
    episodes = (
        args.episodes
        or os.environ.get("SATNAV_STREAMVLN_EVAL_EPISODES")
        or config.DATASET.DATA_PATH
    )
    scenes = (
        args.scenes_dir
        or os.environ.get("SATNAV_STREAMVLN_SCENES_DIR")
        or config.DATASET.SCENES_DIR
    )
    config.DATASET.DATA_PATH = str(
        _resolved_path(str(episodes).format(split=args.split), SATNAV_ROOT)
    )
    config.DATASET.SCENES_DIR = str(_resolved_path(scenes, SATNAV_ROOT))
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else OmegaConf.select(config, "ENVIRONMENT.MAX_EPISODE_STEPS", default=500)
    )
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    config.ENVIRONMENT.MAX_EPISODE_STEPS = max_steps
    OmegaConf.set_struct(config, True)
    dataset = SatNavDataset(config.DATASET)
    return dataset, Env(config, dataset=dataset, cycle=False), max_steps


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rank = args.rank if args.rank is not None else _env_int("RANK", 0)
    world_size = (
        args.world_size if args.world_size is not None else _env_int("WORLD_SIZE", 1)
    )
    local_rank = (
        args.local_rank if args.local_rank is not None else _env_int("LOCAL_RANK", rank)
    )
    args.tokenizer_path = _resolve_tokenizer_path(args)
    dataset, environment, max_steps = _build_environment(args)
    checkpoint_id = args.checkpoint_id or Path(args.model_path).name
    output = Path(
        args.output_dir
        or SATNAV_ROOT
        / "output"
        / "baselines"
        / "vlm"
        / "streamvln"
        / checkpoint_id
        / args.split
    ).expanduser().resolve()
    plan = build_episode_plan(
        dataset.episodes,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        rank=rank,
        world_size=world_size,
    )
    if args.dry_run:
        environment.close()
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dataset_episode_count": len(dataset.episodes),
                    "selected_episode_count": len(plan.selected),
                    "rank_episode_count": len(plan.shard),
                    "output_dir": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    streamvln_repo = bootstrap_streamvln(
        args.streamvln_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    device = f"cuda:{local_rank}" if args.device == "cuda" else args.device
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
            output_dir=output,
            split=args.split,
            policy_id=f"streamvln:{checkpoint_id}",
            offset=args.offset,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
            base_seed=args.base_seed,
            max_steps=max_steps,
            resume=args.resume,
            fail_fast=args.fail_fast,
            fail_on_episode_error=args.fail_on_episode_error,
            capture_action_trace=not args.no_action_trace,
        ),
    )
    print(json.dumps(evaluator.run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

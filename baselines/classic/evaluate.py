"""Unified Random/ReferenceFollower/Seq2Seq/CMA evaluation command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from baselines.classic.common.config import load_classic_config, resolve_path
from baselines.classic.factory import build_classic_adapter, normalize_method
from satnav.core.env import Env
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.evaluation import EvaluationConfig, Evaluator


DEFAULT_CONFIGS = {
    "random": "configs/baselines/random_agent.yaml",
    "reference_follower": "configs/baselines/reference_follower.yaml",
    "seq2seq": "configs/baselines/seq2seq_eval.yaml",
    "cma": "configs/baselines/cma_eval.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("random", "reference", "reference_follower", "seq2seq", "cma"),
    )
    parser.add_argument("--config", help="experiment YAML; defaults by method")
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


def _resolve_dataset_paths(config: DictConfig, split: str) -> None:
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
    *, method: str, config_path: Path, split: str, config: DictConfig
) -> None:
    payload = {
        "method": method,
        "config_path": str(config_path),
        "split": split,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_from_args(args: argparse.Namespace) -> Mapping[str, object]:
    method = normalize_method(args.method)
    config_path = resolve_path(args.config or DEFAULT_CONFIGS[method])
    config = load_classic_config(config_path, overrides=tuple(args.overrides))
    split = _configured_split(config, args.split)
    if args.print_config:
        _print_resolved(
            method=method,
            config_path=config_path,
            split=split,
            config=config,
        )
        return {"status": "config_only"}

    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else OmegaConf.select(config, "ENVIRONMENT.MAX_EPISODE_STEPS", default=500)
    )
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    OmegaConf.update(config, "ENVIRONMENT.MAX_EPISODE_STEPS", max_steps, merge=False)
    _resolve_dataset_paths(config, split)
    dataset = SatNavDataset(config.DATASET)
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

    evaluator = Evaluator(
        environment=environment,
        policy=bundle.adapter,
        config=EvaluationConfig(
            output_dir=_resolved_output_dir(
                bundle.config, method, split, args.output_dir
            ),
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
        ),
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
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

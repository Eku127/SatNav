"""Thin SatNav data-module wrapper around the pinned StreamVLN trainer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from baselines.vlm.streamvln.bootstrap import bootstrap_streamvln
from baselines.vlm.streamvln.dataset import (
    SatNavActionDataset,
    make_satnav_collate_fn,
)


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASELINE_DIR / "configs" / "train.yaml"
DEFAULT_DEEPSPEED = BASELINE_DIR / "configs" / "zero2.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune pinned StreamVLN on SatNav trajectory_data. "
            "Normally launched through scripts/train.sh."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--streamvln-repo", type=Path)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--save-strategy", choices=("epoch", "steps"))
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--report-to", choices=("none", "swanlab"))
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--data-augmentation", dest="data_augmentation", action="store_true"
    )
    augmentation.add_argument(
        "--no-data-augmentation", dest="data_augmentation", action="store_false"
    )
    compilation = parser.add_mutually_exclusive_group()
    compilation.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    compilation.add_argument(
        "--no-torch-compile", dest="torch_compile", action="store_false"
    )
    parser.set_defaults(data_augmentation=None, torch_compile=None)
    parser.add_argument(
        "--print-upstream-args",
        action="store_true",
        help="print the resolved pinned-upstream arguments without importing StreamVLN",
    )
    return parser


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"training config must be a mapping: {path}")
    return value


def _section(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"training config section {name!r} must be a mapping")
    return dict(value)


def _bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def _add(arguments: list, name: str, value: Any) -> None:
    arguments.extend((name, str(value)))


def _validate_smoke_sample_cap(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> None:
    """Reject an explicit sample cap that cannot form one global microbatch."""

    configured_cap = os.environ.get("SATNAV_MAX_SAMPLES", "").strip()
    if not configured_cap:
        return
    try:
        sample_cap = int(configured_cap)
    except ValueError as error:
        raise ValueError("SATNAV_MAX_SAMPLES must be a positive integer") from error
    if sample_cap <= 0:
        raise ValueError("SATNAV_MAX_SAMPLES must be a positive integer")
    training = _section(config, "training")
    if not bool(training.get("dataloader_drop_last", True)):
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch_size = int(
        args.per_device_batch_size
        if args.per_device_batch_size is not None
        else training.get("per_device_train_batch_size", 3)
    )
    minimum = world_size * batch_size
    if sample_cap < minimum:
        raise ValueError(
            "SATNAV_MAX_SAMPLES cannot form one global microbatch with "
            f"dataloader_drop_last=true: {sample_cap} < {minimum} "
            f"(WORLD_SIZE={world_size}, per-device batch={batch_size})"
        )


class _ModelPathWithFamilyHint(str):
    """Preserve a real path while satisfying the pinned trainer's name probe."""

    def __new__(cls, value: str, family: str):
        instance = super().__new__(cls, value)
        instance.family = family
        return instance

    def lower(self) -> str:
        value = super().lower()
        return value if self.family in value else f"{value}#{self.family}"

    def __getnewargs__(self):
        return (str(self), self.family)


def _model_path_with_tokenizer_hint(value: Any) -> str:
    """Infer tokenizer family from metadata when a local dirname omits it.

    The pinned upstream chooses its tokenizer with substring checks against
    ``model_name_or_path``.  Fine-tuned SatNav checkpoint names need not retain
    ``qwen`` in their basename, so provide that hint without changing the path
    seen by filesystem and Hugging Face loaders.
    """

    path = Path(str(value)).expanduser()
    tokenizer_config = path / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return str(value)
    with tokenizer_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer_class = str(config.get("tokenizer_class", "")).lower()
    for family in ("qwen", "mistral", "llama"):
        if family in tokenizer_class:
            return _ModelPathWithFamilyHint(str(value), family)
    return str(value)


def build_upstream_arguments(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Sequence[str]:
    """Translate the maintained YAML into the pinned HF argument surface."""

    model = _section(config, "model")
    data = _section(config, "data")
    training = _section(config, "training")
    scheduler = _section(config, "scheduler")

    required_paths = {
        "model_path": args.model_path,
        "trajectory_root": args.trajectory_root,
        "deepspeed_config": args.deepspeed_config,
    }
    for name, value in required_paths.items():
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or output_dir.name
    values = []
    for name, value in (
        ("--deepspeed", Path(args.deepspeed_config).expanduser().resolve()),
        ("--model_name_or_path", Path(args.model_path).expanduser().resolve()),
        ("--version", model.get("version", "qwen_1_5")),
        ("--video_folder", Path(args.trajectory_root).expanduser().resolve()),
        ("--group_by_task", _bool(data.get("group_by_task", False))),
        ("--num_history", data.get("num_history", 8)),
        ("--num_future_steps", data.get("num_future_steps", 4)),
        ("--num_frames", data.get("num_frames", 32)),
        ("--data_augmentation", _bool(
            args.data_augmentation
            if args.data_augmentation is not None
            else data.get("data_augmentation", True)
        )),
        ("--mm_tunable_parts", model.get(
            "mm_tunable_parts",
            "mm_vision_tower,mm_mlp_adapter,mm_language_model",
        )),
        ("--vision_tower", args.vision_tower),
        ("--mm_projector_type", model.get("mm_projector_type", "mlp2x_gelu")),
        ("--mm_vision_select_layer", model.get("mm_vision_select_layer", -2)),
        ("--mm_use_im_start_end", _bool(model.get("mm_use_im_start_end", False))),
        ("--mm_use_im_patch_token", _bool(model.get("mm_use_im_patch_token", False))),
        ("--image_aspect_ratio", model.get("image_aspect_ratio", "anyres_max_9")),
        ("--image_grid_pinpoints", model.get("image_grid_pinpoints", "(1x1),...,(6x6)")),
        ("--bf16", _bool(training.get("bf16", True))),
        ("--run_name", run_name),
        ("--output_dir", output_dir),
        ("--num_train_epochs", args.num_train_epochs if args.num_train_epochs is not None else training.get("num_train_epochs", 1)),
        ("--per_device_train_batch_size", args.per_device_batch_size if args.per_device_batch_size is not None else training.get("per_device_train_batch_size", 3)),
        ("--per_device_eval_batch_size", training.get("per_device_eval_batch_size", 4)),
        ("--gradient_accumulation_steps", args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else training.get("gradient_accumulation_steps", 2)),
        ("--evaluation_strategy", "no"),
        ("--save_strategy", args.save_strategy or training.get("save_strategy", "epoch")),
        ("--save_total_limit", training.get("save_total_limit", 1)),
        ("--learning_rate", args.learning_rate if args.learning_rate is not None else training.get("learning_rate", 2e-5)),
        ("--mm_vision_tower_lr", training.get("mm_vision_tower_lr", 5e-6)),
        ("--weight_decay", training.get("weight_decay", 0.0)),
        ("--warmup_ratio", scheduler.get("warmup_ratio", 0.075)),
        ("--lr_scheduler_type", scheduler.get("type", "cosine_with_min_lr")),
        ("--lr_scheduler_kwargs", json.dumps(scheduler.get("kwargs", {"min_lr": 1.85e-5}), separators=(",", ":"))),
        ("--logging_steps", args.logging_steps if args.logging_steps is not None else training.get("logging_steps", 10)),
        ("--tf32", _bool(training.get("tf32", True))),
        ("--model_max_length", model.get("model_max_length", 32768)),
        ("--gradient_checkpointing", _bool(training.get("gradient_checkpointing", True))),
        ("--dataloader_num_workers", args.dataloader_num_workers if args.dataloader_num_workers is not None else training.get("dataloader_num_workers", 8)),
        ("--lazy_preprocess", _bool(data.get("lazy_preprocess", True))),
        ("--torch_compile", _bool(
            args.torch_compile
            if args.torch_compile is not None
            else training.get("torch_compile", True)
        )),
        ("--torch_compile_backend", training.get("torch_compile_backend", "inductor")),
        ("--dataloader_drop_last", _bool(training.get("dataloader_drop_last", True))),
        ("--report_to", args.report_to or training.get("report_to", "none")),
    ):
        _add(values, name, value)

    save_strategy = args.save_strategy or training.get("save_strategy", "epoch")
    if save_strategy == "steps":
        _add(
            values,
            "--save_steps",
            args.save_steps if args.save_steps is not None else training.get("save_steps", 1000),
        )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        _add(values, "--max_steps", args.max_steps)
    return values


def _install_satnav_data_module(upstream: Any, vision_tower_path: str) -> Any:
    def make_supervised_data_module(tokenizer: Any, vision_tower: Any, data_args: Any):
        del vision_tower
        dataset = SatNavActionDataset(tokenizer=tokenizer, data_args=data_args, task_id=0)
        upstream.rank0_print(f"len SatNav train_dataset: {len(dataset)}")
        return {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": make_satnav_collate_fn(tokenizer),
        }

    upstream.make_supervised_data_module = make_supervised_data_module

    # The pinned trainer loads model config before initialize_vision_modules().
    # Patch only that loader call so an explicit local tower replaces stale
    # checkpoint paths during offline training, then restore it immediately.
    original_get_model = upstream.get_model

    def get_model(model_args: Any, training_args: Any, data_args: Any, bnb_args: Any):
        original_from_pretrained = upstream.AutoConfig.from_pretrained

        def from_pretrained(*loader_args: Any, **loader_kwargs: Any):
            config = original_from_pretrained(*loader_args, **loader_kwargs)
            config.mm_vision_tower = vision_tower_path
            config.vision_tower = vision_tower_path
            return config

        upstream.AutoConfig.from_pretrained = from_pretrained
        try:
            return original_get_model(model_args, training_args, data_args, bnb_args)
        finally:
            upstream.AutoConfig.from_pretrained = original_from_pretrained

    upstream.get_model = get_model

    # Upstream selects Qwen/Mistral/Llama tokenizers from substrings in the
    # checkpoint dirname.  Preserve compatibility for renamed local exports by
    # attaching a string-only family hint after argument parsing.
    parser_class = upstream.transformers.HfArgumentParser
    original_parse = parser_class.parse_args_into_dataclasses

    def parse_args_into_dataclasses(parser: Any, *parse_args: Any, **parse_kwargs: Any):
        parsed = original_parse(parser, *parse_args, **parse_kwargs)
        if parsed:
            model_args = parsed[0]
            model_args.model_name_or_path = _model_path_with_tokenizer_hint(
                model_args.model_name_or_path
            )
        return parsed

    parser_class.parse_args_into_dataclasses = parse_args_into_dataclasses
    return original_parse


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    _validate_smoke_sample_cap(args, config)
    upstream_arguments = list(build_upstream_arguments(args, config))
    if args.print_upstream_args:
        print(json.dumps(upstream_arguments, indent=2))
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("STREAMVLN_OFFLINE", "true").lower() in (
        "1", "true", "yes", "on"
    ):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    bootstrap_streamvln(
        args.streamvln_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    from streamvln import streamvln_train as upstream

    original_parse = _install_satnav_data_module(upstream, args.vision_tower)
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0]] + upstream_arguments
        upstream.train()
    finally:
        sys.argv = previous_argv
        upstream.transformers.HfArgumentParser.parse_args_into_dataclasses = (
            original_parse
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

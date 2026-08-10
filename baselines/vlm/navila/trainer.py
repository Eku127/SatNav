"""Thin SatNav data-module wrapper around the pinned NaVILA trainer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from baselines.vlm.navila.artifacts import file_identity, navila_model_identity
from baselines.vlm.navila.bootstrap import bootstrap_navila, source_revision
from baselines.vlm.navila.dataset import (
    SatNavNaVILADataset,
    TRAINING_SENTINEL_TOKEN,
    load_annotation_records,
    validate_records,
)
from satnav.evaluation.manifest import ensure_manifest


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASELINE_DIR / "configs" / "train.yaml"
DEFAULT_DEEPSPEED = BASELINE_DIR / "configs" / "zero2.json"
PATH_VALUED_UPSTREAM_OPTIONS = frozenset(
    {
        "--deepspeed",
        "--model_name_or_path",
        "--data_path",
        "--image_folder",
        "--vision_tower",
        "--output_dir",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune pinned NaVILA on SatNav trajectory_data"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--navila-repo", type=Path)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--data-validation-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--train-components", choices=("all", "projector"))
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--print-upstream-args", action="store_true")
    return parser


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"training config must be a mapping: {path}")
    return value


def _value(args: argparse.Namespace, config: Mapping[str, Any], key: str, default=None):
    argument = key.replace("-", "_")
    value = getattr(args, argument, None)
    return config.get(key, default) if value is None else value


def build_upstream_arguments(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Sequence[str]:
    training = dict(config.get("training", {}))
    model = dict(config.get("model", {}))
    data = dict(config.get("data", {}))
    components = args.train_components or training.get("train-components", "all")
    tune_all = components == "all"
    trajectory_root = Path(args.trajectory_root).expanduser().resolve()
    annotation_path = trajectory_root / "annotations.json"
    arguments = [
        "--deepspeed",
        str(Path(args.deepspeed_config).expanduser().resolve()),
        "--model_name_or_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--version",
        str(model.get("version", "llama_3")),
        "--data_path",
        str(annotation_path),
        "--image_folder",
        str(trajectory_root),
        "--vision_tower",
        str(Path(args.model_path).expanduser().resolve() / "vision_tower"),
        "--mm_vision_select_feature",
        str(model.get("mm-vision-select-feature", "cls_patch")),
        "--mm_projector",
        str(model.get("mm-projector", "mlp_downsample")),
        "--num_video_frames",
        str(data.get("num-video-frames", 8)),
        "--tune_vision_tower",
        str(tune_all),
        "--tune_mm_projector",
        "True",
        "--tune_language_model",
        str(tune_all),
        "--mm_vision_select_layer",
        str(model.get("mm-vision-select-layer", -2)),
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "resize",
        "--data_mixture",
        "satnav",
        "--longvila_sampler",
        "False",
        "--bf16",
        "True",
        "--output_dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--num_train_epochs",
        str(training.get("num-train-epochs", 1)),
        "--per_device_train_batch_size",
        str(_value(args, training, "per-device-batch-size", 1)),
        "--gradient_accumulation_steps",
        str(_value(args, training, "gradient-accumulation-steps", 1)),
        "--do_eval",
        "False",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(_value(args, training, "save-steps", 1000)),
        "--save_total_limit",
        str(training.get("save-total-limit", 1)),
        "--learning_rate",
        str(_value(args, training, "learning-rate", 3e-5)),
        "--weight_decay",
        str(training.get("weight-decay", 0.0)),
        "--warmup_ratio",
        str(_value(args, training, "warmup-ratio", 0.03)),
        "--lr_scheduler_type",
        str(training.get("lr-scheduler-type", "cosine")),
        "--logging_steps",
        str(training.get("logging-steps", 1)),
        "--tf32",
        str(training.get("tf32", True)),
        "--model_max_length",
        str(model.get("model-max-length", 4096)),
        "--gradient_checkpointing",
        str(training.get("gradient-checkpointing", True)),
        "--dataloader_num_workers",
        str(_value(args, training, "dataloader-num-workers", 8)),
        "--lazy_preprocess",
        "True",
        "--report_to",
        str(args.report_to),
        "--seed",
        str(training.get("seed", 10)),
    ]
    if args.max_steps is not None:
        arguments.extend(("--max_steps", str(args.max_steps)))
    return arguments


def _semantic_upstream_arguments(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Sequence[str]:
    """Return behavior arguments without local filesystem bindings."""

    arguments = list(build_upstream_arguments(args, config))
    if len(arguments) % 2:
        raise ValueError("NaVILA upstream arguments must be option/value pairs")
    semantic = []
    for index in range(0, len(arguments), 2):
        option, value = arguments[index : index + 2]
        if not option.startswith("--"):
            raise ValueError(f"invalid NaVILA upstream option: {option!r}")
        if option not in PATH_VALUED_UPSTREAM_OPTIONS:
            semantic.extend((option, value))
    return semantic


def _validate_sample_cap(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    cap = os.environ.get("SATNAV_MAX_SAMPLES", "").strip()
    if not cap:
        return
    training = dict(config.get("training", {}))
    batch = int(_value(args, training, "per-device-batch-size", 1))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if int(cap) < batch * world:
        raise ValueError(
            "SATNAV_MAX_SAMPLES must fill at least one global microbatch: "
            f"{cap} < {batch * world}"
        )


def _training_payload(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    upstream_revision: str,
) -> Mapping[str, Any]:
    allow_external = os.environ.get(
        "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS", ""
    ).lower() in ("1", "true", "yes", "on")
    annotation_path, records = load_annotation_records(
        args.trajectory_root, allow_external_paths=allow_external
    )
    report_path = Path(args.data_validation_report).expanduser().resolve()
    with report_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if (
        validation.get("status") != "passed"
        or validation.get("strict_frames") is not True
        or int(validation.get("annotations", -1)) != len(records)
        or bool(validation.get("allow_external_paths", False)) != allow_external
        or validation.get("annotation_artifact", {}).get("sha256")
        != file_identity(annotation_path)["sha256"]
        or int(validation.get("frame_tree", {}).get("file_count", -1)) <= 0
        or not validation.get("frame_tree", {}).get("digest")
        or validation.get("frame_tree", {}).get("algorithm")
        != "sha256-relative-path-size-content-v1"
    ):
        raise ValueError(
            "--data-validation-report is not a strict, matching trajectory preflight"
        )
    training = dict(config.get("training", {}))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch = int(_value(args, training, "per-device-batch-size", 1))
    gradient_accumulation = int(
        _value(args, training, "gradient-accumulation-steps", 1)
    )
    return {
        "manifest_type": "navila_training",
        "schema_version": 1,
        "upstream_revision": upstream_revision,
        "source_model": navila_model_identity(args.model_path),
        "trajectory": {
            "artifact": file_identity(annotation_path),
            "annotation_count": len(records),
            "frame_tree": validation["frame_tree"],
            "validation": {
                "strict_frames": True,
                "full_content_hash": True,
                "allow_external_paths": allow_external,
            },
        },
        "configuration": _local_configuration(
            args,
            config,
            world_size=world_size,
            per_device_batch=per_device_batch,
            gradient_accumulation=gradient_accumulation,
        ),
    }


def _local_configuration(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    world_size: Optional[int] = None,
    per_device_batch: Optional[int] = None,
    gradient_accumulation: Optional[int] = None,
) -> Mapping[str, Any]:
    training = dict(config.get("training", {}))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if per_device_batch is None:
        per_device_batch = int(_value(args, training, "per-device-batch-size", 1))
    if gradient_accumulation is None:
        gradient_accumulation = int(
            _value(args, training, "gradient-accumulation-steps", 1)
        )
    return {
        "train": file_identity(args.config),
        "deepspeed": file_identity(args.deepspeed_config),
        "integration_source": {
            name: file_identity(BASELINE_DIR / name)
            for name in ("actions.py", "dataset.py", "trainer.py")
        },
        "semantic_upstream_arguments": list(
            _semantic_upstream_arguments(args, config)
        ),
        "sampling_environment": {
            name: os.environ.get(name, "")
            for name in (
                "SATNAV_MAX_EPISODES",
                "SATNAV_MAX_SAMPLES",
                "SATNAV_SAMPLE_RATIO",
                "SATNAV_SAMPLE_STRIDE",
                "SATNAV_HEAD_KEEP",
                "SATNAV_STOP_REPEAT",
                "SATNAV_ACTION_FORMAT",
                "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS",
            )
        },
        "distributed": {
            "launcher": "torch.distributed.run",
            "world_size": world_size,
            "per_device_batch_size": per_device_batch,
            "gradient_accumulation_steps": gradient_accumulation,
            "global_microbatch_size": world_size * per_device_batch,
            "global_update_batch_size": (
                world_size * per_device_batch * gradient_accumulation
            ),
        },
    }


def _prepare_data_validation(args: argparse.Namespace) -> Path:
    """Create one fresh full-content report per distributed trainer launch."""

    report_path = (
        Path(
            args.data_validation_report
            or Path(args.output_dir).expanduser().resolve() / "data_validation.json"
        )
        .expanduser()
        .resolve()
    )
    run_id = os.environ.get("SATNAV_VALIDATION_RUN_ID", "").strip()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not run_id:
        raise ValueError(
            "SATNAV_VALIDATION_RUN_ID is required for distributed training; "
            "use scripts/train.sh"
        )
    if not run_id:
        run_id = f"standalone-{os.getpid()}"
        os.environ["SATNAV_VALIDATION_RUN_ID"] = run_id

    if rank == 0:
        allow_external = os.environ.get(
            "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS", ""
        ).lower() in ("1", "true", "yes", "on")
        annotation_path, records = load_annotation_records(
            args.trajectory_root, allow_external_paths=allow_external
        )
        report = dict(
            validate_records(
                annotation_path,
                records,
                strict_frames=True,
                decode_samples=0,
                hash_frame_content=True,
            ),
            allow_external_paths=allow_external,
            completed_at_ns=time.time_ns(),
            validation_run_id=run_id,
        )
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.{run_id}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, report_path)
        print(encoded, end="", flush=True)
    else:
        deadline = time.monotonic() + 6 * 60 * 60
        while True:
            try:
                with report_path.open("r", encoding="utf-8") as handle:
                    report = json.load(handle)
                if report.get("validation_run_id") == run_id:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for rank-0 data validation: {report_path}"
                )
            time.sleep(1)
    args.data_validation_report = report_path
    return report_path


def _install_data_module(train_mod: Any) -> None:
    from llava.data.dataset import DataCollatorForSupervisedDataset

    def make_data_module(tokenizer, data_args, training_args):
        dataset = SatNavNaVILADataset(
            data_path=data_args.data_path,
            image_folder=data_args.image_folder,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
        )
        training_args.sample_lens = [len(dataset)]
        return {
            "train_dataset": dataset,
            "data_collator": DataCollatorForSupervisedDataset(
                tokenizer=tokenizer, data_args=data_args
            ),
        }

    train_mod.make_supervised_data_module = make_data_module


def _bind_existing_training_sentinel(tokenizer: Any) -> None:
    """Bind NaVILA's temporary label marker without growing the vocabulary."""

    vocabulary = tokenizer.get_vocab()
    if TRAINING_SENTINEL_TOKEN not in vocabulary:
        raise ValueError(
            "the pinned tokenizer is missing the reserved SatNav training sentinel"
        )
    token_id = int(vocabulary[TRAINING_SENTINEL_TOKEN])
    encoded = tokenizer(TRAINING_SENTINEL_TOKEN, add_special_tokens=False).input_ids
    if list(encoded) != [token_id]:
        raise ValueError("training sentinel must encode to exactly one token")
    if tokenizer.decode([token_id], skip_special_tokens=True):
        raise ValueError("training sentinel must already be a skipped special token")

    original_size = len(tokenizer)
    added = tokenizer.add_tokens([TRAINING_SENTINEL_TOKEN], special_tokens=True)
    if added != 0 or len(tokenizer) != original_size:
        raise RuntimeError("training sentinel unexpectedly changed tokenizer size")
    tokenizer.sentinel_token = TRAINING_SENTINEL_TOKEN
    tokenizer.sentinel_token_id = token_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    _validate_sample_cap(args, config)
    upstream_args = list(build_upstream_arguments(args, config))
    if args.print_upstream_args:
        print(json.dumps(upstream_args, indent=2))
        return 0
    repo = bootstrap_navila(
        args.navila_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    revision = source_revision(repo) or "unknown"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _prepare_data_validation(args)
    ensure_manifest(
        output / "training_manifest.json",
        _training_payload(args, config, revision),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from unittest import mock

    import llava.train.train as train_mod
    from llava.train.transformer_normalize_monkey_patch import patched_normalize
    import llava.utils.tokenizer as tokenizer_utils

    _install_data_module(train_mod)

    def batch_sampler_len(loader):
        return len(loader.batch_sampler)

    def batch_sampler_iter(loader):
        return iter(loader.batch_sampler)

    previous = sys.argv
    try:
        sys.argv = [previous[0]] + upstream_args
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "transformers.image_processing_utils.normalize",
                    new=patched_normalize,
                )
            )
            stack.enter_context(
                mock.patch(
                    "accelerate.data_loader.BatchSamplerShard.__len__",
                    new=batch_sampler_len,
                )
            )
            stack.enter_context(
                mock.patch(
                    "accelerate.data_loader.BatchSamplerShard.__iter__",
                    new=batch_sampler_iter,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    tokenizer_utils,
                    "SENTINEL_TOKEN",
                    TRAINING_SENTINEL_TOKEN,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    tokenizer_utils,
                    "_maybe_add_sentinel_token",
                    new=_bind_existing_training_sentinel,
                )
            )
            train_mod.train()
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

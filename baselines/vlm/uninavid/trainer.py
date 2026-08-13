"""Thin SatNav data-module wrapper around the pinned Uni-NaVid trainer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from unittest import mock

from baselines.vlm.uninavid.artifacts import (
    artifact_manifest,
    external_asset_identities,
    file_identity,
    model_identity,
)
from baselines.vlm.uninavid.bootstrap import (
    bootstrap_uninavid,
    install_image_dataset_decord_stub,
    source_revision,
)
from baselines.vlm.uninavid.checkpoint import (
    checkpoint_load_report,
    tokenizer_report,
)
from baselines.vlm.uninavid.dataset import (
    SatNavUniNaVidDataset,
    load_annotation_records,
    validate_records,
)
from satnav.training.manifest import (
    ManifestMismatchError,
    ensure_manifest,
    read_manifest,
)


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASELINE_DIR / "configs" / "train.yaml"
DEFAULT_DEEPSPEED = BASELINE_DIR / "configs" / "zero1.json"
CHECKPOINT_STATE_MANIFEST = "satnav_checkpoint_manifest.json"
PATH_VALUED_UPSTREAM_OPTIONS = frozenset(
    {
        "--deepspeed",
        "--model_name_or_path",
        "--data_path",
        "--image_folder",
        "--video_folder",
        "--vision_tower",
        "--image_processor",
        "--output_dir",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Uni-NaVid on SatNav")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--uninavid-repo", type=Path, required=True)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--eva-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--data-validation-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-strategy", choices=("no", "steps"))
    parser.add_argument("--train-components", choices=("all",))
    parser.add_argument("--no-flash-attention", action="store_true")
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


def _value(args: argparse.Namespace, values: Mapping[str, Any], key: str, default=None):
    argument = key.replace("-", "_")
    explicit = getattr(args, argument, None)
    return values.get(key, default) if explicit is None else explicit


def build_upstream_arguments(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Sequence[str]:
    training = dict(config.get("training", {}))
    model = dict(config.get("model", {}))
    components = args.train_components or training.get("train-components", "all")
    tune_projector_only = components == "projector"
    trajectory = Path(args.trajectory_root).expanduser().resolve()
    arguments = [
        "--deepspeed",
        str(Path(args.deepspeed_config).expanduser().resolve()),
        "--model_name_or_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--version",
        str(model.get("version", "imgsp_v1")),
        "--data_path",
        str(trajectory / "annotations.json"),
        "--image_folder",
        str(trajectory),
        "--video_folder",
        str(trajectory),
        "--vision_tower",
        str(Path(args.eva_path).expanduser().resolve()),
        "--image_processor",
        str(Path(args.processor_path).expanduser().resolve()),
        "--tune_vision_encoder",
        "False",
        "--tune_mm_mlp_adapter",
        str(tune_projector_only),
        "--mm_projector_type",
        str(model.get("mm-projector-type", "mlp2x_gelu")),
        "--mm_vision_select_layer",
        str(model.get("mm-vision-select-layer", -2)),
        "--mm_vision_select_feature",
        str(model.get("mm-vision-select-feature", "patch")),
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "pad",
        "--video_fps",
        str(model.get("video-fps", 1)),
        "--compress_type",
        str(model.get("compress-type", "grid:2")),
        "--bf16",
        "True",
        "--output_dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--num_train_epochs",
        str(training.get("num-train-epochs", 1)),
        "--per_device_train_batch_size",
        str(_value(args, training, "per-device-batch-size", 1)),
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(_value(args, training, "gradient-accumulation-steps", 1)),
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        str(_value(args, training, "save-strategy", "steps")),
        "--save_steps",
        str(_value(args, training, "save-steps", 1000)),
        "--save_total_limit",
        str(training.get("save-total-limit", 1)),
        "--learning_rate",
        str(_value(args, training, "learning-rate", 1e-5)),
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
        str(model.get("model-max-length", 1536)),
        "--gradient_checkpointing",
        str(training.get("gradient-checkpointing", True)),
        "--dataloader_num_workers",
        str(_value(args, training, "dataloader-num-workers", 2)),
        "--lazy_preprocess",
        "True",
        "--report_to",
        str(args.report_to),
        "--seed",
        str(training.get("seed", 10)),
    ]
    if args.max_steps is not None:
        arguments.extend(("--max_steps", str(args.max_steps)))
    if args.run_name is not None:
        arguments.extend(("--run_name", str(args.run_name)))
    return arguments


def _semantic_upstream_arguments(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Sequence[str]:
    """Return behavior arguments without local filesystem bindings."""

    arguments = list(build_upstream_arguments(args, config))
    if len(arguments) % 2:
        raise ValueError("Uni-NaVid upstream arguments must be option/value pairs")
    semantic = []
    for index in range(0, len(arguments), 2):
        option, value = arguments[index : index + 2]
        if not option.startswith("--"):
            raise ValueError(f"invalid Uni-NaVid upstream option: {option!r}")
        if option not in PATH_VALUED_UPSTREAM_OPTIONS:
            semantic.extend((option, value))
    return semantic


def _validate_report(
    args: argparse.Namespace, *, expected_run_id: str
) -> Mapping[str, Any]:
    report_path = Path(args.data_validation_report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    annotation_path, records = load_annotation_records(args.trajectory_root)
    frame_tree = report.get("frame_tree", {})
    if (
        report.get("status") != "passed"
        or report.get("validation_run_id") != expected_run_id
        or report.get("strict_frames") is not True
        or int(report.get("annotations", -1)) != len(records)
        or report.get("allow_external_paths") is not False
        or report.get("annotation_artifact", {}).get("sha256")
        != file_identity(annotation_path)["sha256"]
        or frame_tree.get("algorithm")
        != "sha256-relative-path-size-content-v1"
        or not frame_tree.get("digest")
        or int(frame_tree.get("file_count", -1)) <= 0
    ):
        raise ValueError("data validation report is not a strict matching preflight")
    return report


def _write_fresh_validation(args: argparse.Namespace, run_id: str) -> None:
    report_path = Path(args.data_validation_report).expanduser().resolve()
    annotation_path, records = load_annotation_records(
        args.trajectory_root, allow_external_paths=False
    )
    report = dict(
        validate_records(
            annotation_path,
            records,
            strict_frames=True,
            hash_frame_content=True,
            decode_samples=32,
        ),
        allow_external_paths=False,
        validation_run_id=run_id,
        completed_at_ns=time.time_ns(),
    )
    if int(report["annotations"]) != 105164:
        raise ValueError(
            f"canonical episode count mismatch: {report['annotations']} != 105164"
        )
    if int(report["window_samples"]) != 1399366:
        raise ValueError(
            "canonical Uni-NaVid window count mismatch: "
            f"{report['window_samples']} != 1399366"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _validate_training_shape(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> None:
    training = dict(config.get("training", {}))
    data = dict(config.get("data", {}))
    components = args.train_components or training.get("train-components", "all")
    if components != "all":
        raise ValueError("maintained Uni-NaVid training requires train-components=all")
    if int(data.get("window-size", 4)) != 4:
        raise ValueError("the pinned Uni-NaVid training contract requires window-size 4")
    if os.environ.get("SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise ValueError("canonical Uni-NaVid training forbids external trajectory paths")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch = int(_value(args, training, "per-device-batch-size", 1))
    accumulation = int(
        _value(args, training, "gradient-accumulation-steps", 1)
    )
    if world_size <= 0 or batch <= 0 or accumulation <= 0:
        raise ValueError("world size, batch size, and accumulation must be positive")
    if args.max_steps is not None and int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    cap = os.environ.get("SATNAV_MAX_SAMPLES", "").strip()
    if cap and int(cap) < world_size * batch:
        raise ValueError(
            "SATNAV_MAX_SAMPLES must fill at least one global microbatch: "
            f"{cap} < {world_size * batch}"
        )


def _validate_new_output(output: Path) -> None:
    allowed = {"data_validation.json", "data_validation.log", "train.log"}
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(
            "fresh output directory contains prior artifacts; use --resume or a "
            f"new directory: {unexpected}"
        )


def _training_payload(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    revision: str,
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "manifest_type": "uninavid_training",
        "schema_version": 1,
        "upstream_revision": revision,
        "source_model": model_identity(args.model_path),
        "external_assets": external_asset_identities(
            args.eva_path, args.processor_path
        ),
        "trajectory": {
            "artifact": validation["annotation_artifact"],
            "annotation_count": validation["annotations"],
            "window_samples": validation["window_samples"],
            "frame_tree": validation["frame_tree"],
            "strict_frames": True,
            "allow_external_paths": False,
        },
        "configuration": _local_configuration(args, config),
    }


def _local_configuration(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    training = dict(config.get("training", {}))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch = int(_value(args, training, "per-device-batch-size", 1))
    accumulation = int(_value(args, training, "gradient-accumulation-steps", 1))
    return {
        "train": file_identity(args.config),
        "deepspeed": file_identity(args.deepspeed_config),
        "integration_source": {
            name: file_identity(BASELINE_DIR / name)
            for name in (
                "actions.py",
                "artifacts.py",
                "bootstrap.py",
                "checkpoint.py",
                "dataset.py",
                "trainer.py",
            )
        },
        "semantic_upstream_arguments": list(
            _semantic_upstream_arguments(args, config)
        ),
        "sampling_environment": {
            name: os.environ.get(name, "")
            for name in (
                "SATNAV_MAX_EPISODES",
                "SATNAV_MAX_SAMPLES",
                "SATNAV_UNINAVID_AUGMENTATION",
                "SATNAV_UNINAVID_GRADIENT_AUDIT_STEPS",
                "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS",
            )
        },
        "distributed": {
            "launcher": "torch.distributed.run",
            "world_size": world_size,
            "per_device_batch_size": batch,
            "gradient_accumulation_steps": accumulation,
            "global_microbatch_size": world_size * batch,
            "global_update_batch_size": world_size * batch * accumulation,
        },
        "flash_attention": not args.no_flash_attention,
    }


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _prepare_manifest(
    args: argparse.Namespace, config: Mapping[str, Any], revision: str
) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    manifest_path = output / "training_manifest.json"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    run_id = os.environ.get("SATNAV_UNINAVID_RUN_ID", "").strip()
    if world_size > 1 and not run_id:
        raise ValueError(
            "SATNAV_UNINAVID_RUN_ID is required for distributed training; "
            "use scripts/train.sh"
        )
    if not run_id:
        run_id = f"standalone-{os.getpid()}"
    ready = output / f".manifest-ready-{run_id}"
    failed = output / f".manifest-error-{run_id}"
    if rank == 0:
        try:
            if args.resume:
                _validate_resume_checkpoint(output)
            else:
                _validate_new_output(output)
            _write_fresh_validation(args, run_id)
            validation = _validate_report(args, expected_run_id=run_id)
            ensure_manifest(
                manifest_path,
                _training_payload(args, config, revision, validation),
            )
            _atomic_text(ready, "ready\n")
        except BaseException as error:
            _atomic_text(failed, f"{type(error).__name__}: {error}\n")
            raise
    else:
        deadline = time.monotonic() + 6 * 60 * 60
        while not ready.is_file():
            if failed.is_file():
                message = failed.read_text(encoding="utf-8").strip()
                raise RuntimeError(f"rank-0 manifest preparation failed: {message}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {ready}")
            time.sleep(1)
        envelope = read_manifest(manifest_path)
        payload = envelope["payload"]
        if (
            payload.get("upstream_revision") != revision
            or payload.get("configuration") != _local_configuration(args, config)
        ):
            raise ManifestMismatchError(
                "rank-local Uni-NaVid training configuration differs from rank 0"
            )


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory name: {path.name}") from error


def _checkpoint_state_identity(path: Path) -> Mapping[str, Any]:
    files = [
        candidate
        for candidate in Path(path).rglob("*")
        if candidate.is_file() and candidate.name != CHECKPOINT_STATE_MANIFEST
    ]
    return artifact_manifest(path, files)


def _validate_resume_checkpoint(output: Path) -> None:
    checkpoints = sorted(output.glob("checkpoint-*"), key=_checkpoint_step)
    if not checkpoints:
        raise ValueError("--resume requested but no checkpoint-* exists")
    checkpoint = checkpoints[-1]
    marker = checkpoint / CHECKPOINT_STATE_MANIFEST
    envelope = read_manifest(marker)
    payload = envelope["payload"]
    actual = _checkpoint_state_identity(checkpoint)
    if (
        payload.get("manifest_type") != "uninavid_checkpoint_state"
        or int(payload.get("global_step", -1)) != _checkpoint_step(checkpoint)
        or payload.get("identity", {}).get("digest") != actual["digest"]
    ):
        raise ValueError(f"resume checkpoint identity mismatch: {checkpoint}")


def _install_data_module(train_module: Any) -> None:
    def make_data_module(tokenizer, data_args):
        dataset = SatNavUniNaVidDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        collator = train_module.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": collator,
        }

    train_module.make_supervised_data_module = make_data_module


def _component(name: str) -> str:
    if "vision_tower" in name:
        return "vision"
    if "mm_projector" in name:
        return "projector"
    return "language"


class _BackwardGradientCapture:
    """Capture small gradient summaries before DeepSpeed releases ``.grad``.

    ZeRO may clear or partition parameter gradients inside ``backward``.  A
    parameter hook observes the gradient first and retains only two detached
    device scalars (finite and max-absolute), never the gradient tensor.
    Summaries accumulate across microbatches that share an optimizer step.
    """

    def __init__(self, model: Any) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._global_step: Optional[int] = None
        self._microbatches = 0
        self._finite_values = []
        self._maxima = {
            name: [] for name in ("language", "projector", "vision")
        }
        self._vision_names: set[str] = set()
        self._reference_parameter = None
        self._handles = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._reference_parameter is None:
                self._reference_parameter = parameter
            self._handles.append(
                parameter.register_hook(self._make_hook(name, _component(name)))
            )
        if self._reference_parameter is None:
            raise RuntimeError("Uni-NaVid has no trainable parameters to audit")

    def _make_hook(self, name: str, component: str):
        def capture(gradient):
            with self._lock:
                if not self._active:
                    return gradient
            value = gradient.detach()
            finite = value.isfinite().all()
            maximum = value.abs().max().float()
            with self._lock:
                if self._active:
                    self._finite_values.append(finite)
                    self._maxima[component].append(maximum)
                    if component == "vision":
                        self._vision_names.add(name)
            return gradient

        return capture

    def begin_microbatch(self, global_step: int) -> None:
        with self._lock:
            if self._global_step != global_step:
                self._global_step = global_step
                self._microbatches = 0
                self._finite_values = []
                self._maxima = {
                    name: []
                    for name in ("language", "projector", "vision")
                }
                self._vision_names = set()
            self._microbatches += 1
            self._active = True

    def pause(self) -> None:
        with self._lock:
            self._active = False

    def summarize(self) -> Mapping[str, Any]:
        """Synchronize one four-scalar summary for the current optimizer step."""

        import torch

        with self._lock:
            self._active = False
            finite_values = tuple(self._finite_values)
            maxima = {
                name: tuple(values) for name, values in self._maxima.items()
            }
            vision_names = sorted(self._vision_names)
            microbatches = self._microbatches
            global_step = self._global_step
        reference = next(
            (
                value
                for values in maxima.values()
                for value in values
            ),
            None,
        )
        if reference is None:
            reference = self._reference_parameter.new_zeros((), dtype=torch.float32)

        def maximum(values):
            if not values:
                return reference.new_zeros((), dtype=torch.float32)
            return torch.stack(values).max().float()

        all_finite = (
            torch.stack(finite_values).all()
            if finite_values
            else reference.new_zeros((), dtype=torch.bool)
        )
        device_summary = torch.stack(
            (
                all_finite.float(),
                maximum(maxima["language"]),
                maximum(maxima["projector"]),
                maximum(maxima["vision"]),
            )
        )
        finite, language, projector, vision = device_summary.detach().cpu().tolist()
        return {
            "global_step": global_step,
            "microbatches": microbatches,
            "all_finite": bool(finite),
            "max_abs": {
                "language": float(language),
                "projector": float(projector),
                "vision": float(vision),
            },
            "vision_tensors": vision_names,
        }


def _install_training_audit(trainer_class: Any, output_dir: Path) -> None:
    import torch

    audit_steps = int(os.environ.get("SATNAV_UNINAVID_GRADIENT_AUDIT_STEPS", "2"))
    if audit_steps <= 0:
        raise ValueError("SATNAV_UNINAVID_GRADIENT_AUDIT_STEPS must be positive")
    original_init = trainer_class.__init__
    original_step = trainer_class.training_step
    original_save_checkpoint = trainer_class._save_checkpoint

    def audited_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        totals = {
            name: {"parameters": 0, "trainable": 0}
            for name in ("language", "projector", "vision")
        }
        for name, parameter in self.model.named_parameters():
            group = totals[_component(name)]
            group["parameters"] += int(parameter.numel())
            if parameter.requires_grad:
                group["trainable"] += int(parameter.numel())
        rank = int(os.environ.get("RANK", "0"))
        path = output_dir / f"trainability_rank_{rank:05d}.json"
        path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[uninavid-trainability] {json.dumps(totals, sort_keys=True)}", flush=True)
        if totals["vision"]["trainable"] != 0:
            raise RuntimeError("Uni-NaVid vision tower must be fully frozen")
        for required in ("language", "projector"):
            if totals[required]["trainable"] <= 0:
                raise RuntimeError(
                    f"Uni-NaVid {required} has no trainable parameters"
                )
        self._satnav_gradient_capture = _BackwardGradientCapture(self.model)

    def audited_step(self, model, inputs):
        valid = int((inputs["labels"] != -100).sum().item())
        if valid <= 0:
            raise RuntimeError("training batch has no supervised labels")
        global_step = int(self.state.global_step)
        capture = self._satnav_gradient_capture
        if global_step < audit_steps:
            capture.begin_microbatch(global_step)
        try:
            loss = original_step(self, model, inputs)
        finally:
            capture.pause()
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError(f"non-finite Uni-NaVid loss: {loss}")
        if global_step >= audit_steps:
            return loss
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not bool(accelerator.sync_gradients):
            return loss
        summary = capture.summarize()
        if not summary["all_finite"]:
            raise RuntimeError("non-finite Uni-NaVid gradient")
        if summary["vision_tensors"] or summary["max_abs"]["vision"] != 0:
            raise RuntimeError("frozen Uni-NaVid vision tower received a gradient")
        for required in ("language", "projector"):
            if summary["max_abs"][required] <= 0:
                raise RuntimeError(
                    f"Uni-NaVid {required} gradient is absent or identically zero"
                )
        gradient = {
            "language": {"max_abs": summary["max_abs"]["language"]},
            "projector": {"max_abs": summary["max_abs"]["projector"]},
            "vision": None,
        }
        event = {
            "rank": int(os.environ.get("RANK", "0")),
            "global_step_before_update": global_step,
            "microbatches_observed": summary["microbatches"],
            "loss": float(loss.detach().float().cpu()),
            "valid_labels": valid,
            "nonzero_gradient": gradient,
        }
        path = output_dir / f"step_audit_rank_{event['rank']:05d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(f"[uninavid-step-audit] {json.dumps(event, sort_keys=True)}", flush=True)
        return loss

    def audited_save_checkpoint(self, *values, **kwargs):
        result = original_save_checkpoint(self, *values, **kwargs)
        distributed = torch.distributed
        synchronized = distributed.is_available() and distributed.is_initialized()
        if synchronized:
            distributed.barrier()
        if bool(self.args.should_save):
            step = int(self.state.global_step)
            checkpoint = output_dir / f"checkpoint-{step}"
            identity = _checkpoint_state_identity(checkpoint)
            ensure_manifest(
                checkpoint / CHECKPOINT_STATE_MANIFEST,
                {
                    "manifest_type": "uninavid_checkpoint_state",
                    "schema_version": 1,
                    "global_step": step,
                    "identity": identity,
                },
            )
        if synchronized:
            distributed.barrier()
        return result

    trainer_class.__init__ = audited_init
    trainer_class.training_step = audited_step
    trainer_class._save_checkpoint = audited_save_checkpoint


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    _validate_training_shape(args, config)
    augmentation = bool(dict(config.get("data", {})).get("augmentation", True))
    os.environ.setdefault(
        "SATNAV_UNINAVID_AUGMENTATION", str(augmentation).lower()
    )
    upstream_arguments = list(build_upstream_arguments(args, config))
    if args.print_upstream_args:
        print(json.dumps(upstream_arguments, indent=2))
        return 0
    repo = bootstrap_uninavid(
        args.uninavid_repo,
        require_pinned_revision=not args.allow_upstream_mismatch,
    )
    revision = source_revision(repo) or "unknown"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.data_validation_report is None:
        args.data_validation_report = output / "data_validation.json"
    checkpoints = tuple(output.glob("checkpoint-*"))
    if checkpoints and not args.resume:
        raise ValueError("checkpoint exists; pass --resume or use a fresh output")
    if args.resume and not checkpoints:
        raise ValueError("--resume requested but no checkpoint-* exists")
    _prepare_manifest(args, config, revision)
    install_image_dataset_decord_stub()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import transformers
    import uninavid.model.uninavid_arch as architecture
    import uninavid.train.train as train_module
    from uninavid.model.language_model.llava_llama_vid import (
        LlavaLlamaAttForCausalLM,
    )
    from uninavid.train.llava_trainer import LLaVATrainer

    _install_data_module(train_module)
    _install_training_audit(LLaVATrainer, output)
    original_config_loader = transformers.AutoConfig.from_pretrained
    original_model_loader = LlavaLlamaAttForCausalLM.from_pretrained
    original_tokenizer_initializer = (
        LlavaLlamaAttForCausalLM.initialize_vision_tokenizer
    )
    original_vision_builder = architecture.build_vision_tower
    retained_vision_tower = []

    def checkpoint_vision_builder(value, **kwargs):
        if retained_vision_tower:
            return retained_vision_tower[0]
        kwargs["delay_load"] = False
        tower = original_vision_builder(value, **kwargs)
        retained_vision_tower.append(tower)
        return tower

    @classmethod
    def configured_model_loader(cls, *values, **kwargs):
        if not args.no_flash_attention:
            kwargs.setdefault("use_flash_attention_2", True)
        kwargs["output_loading_info"] = True
        model, loading_info = original_model_loader(*values, **kwargs)
        report = checkpoint_load_report(
            model,
            args.model_path,
            loading_info,
            flash_attention=not args.no_flash_attention,
        )
        rank = int(os.environ.get("RANK", "0"))
        report_path = output / f"initial_load_rank_{rank:05d}.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if report["status"] != "passed":
            raise RuntimeError(
                f"strict initial Uni-NaVid load failed: {report['errors']}"
            )
        return model

    def configured_auto_config(*values, **kwargs):
        loaded = original_config_loader(*values, **kwargs)
        loaded.mm_vision_tower = str(Path(args.eva_path).expanduser().resolve())
        loaded.image_processor = str(Path(args.processor_path).expanduser().resolve())
        return loaded

    def audited_tokenizer_initializer(self, model_args, tokenizer):
        before = len(tokenizer)
        result = original_tokenizer_initializer(self, model_args, tokenizer)
        report = dict(tokenizer_report(tokenizer, self), added_tokens=len(tokenizer) - before)
        rank = int(os.environ.get("RANK", "0"))
        report_path = output / f"tokenizer_contract_rank_{rank:05d}.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if report["added_tokens"] != 0 or report["errors"]:
            raise RuntimeError(f"Uni-NaVid tokenizer contract failed: {report}")
        return result

    previous = sys.argv
    try:
        sys.argv = [previous[0], *upstream_arguments]
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    LlavaLlamaAttForCausalLM,
                    "from_pretrained",
                    new=configured_model_loader,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    transformers.AutoConfig,
                    "from_pretrained",
                    new=configured_auto_config,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    architecture,
                    "build_vision_tower",
                    new=checkpoint_vision_builder,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    LlavaLlamaAttForCausalLM,
                    "initialize_vision_tokenizer",
                    new=audited_tokenizer_initializer,
                )
            )
            train_module.train()
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

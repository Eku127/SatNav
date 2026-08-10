"""Fail-closed OpenFly fine-tuning entrypoint for SatNav trajectory data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from baselines.vlm.openfly.artifacts import (
    artifact_manifest,
    file_identity,
    native_source_identity,
    openfly_model_identity,
    processor_identity,
)
from baselines.vlm.openfly.backends import (
    build_train_backend,
    patch_accelerate_optimizer_train_eval,
    resolve_backend,
)
from baselines.vlm.openfly.dataset import (
    FRAME_DIGEST_BYTES,
    FRAME_IDENTITY_INDEX_ALGORITHM,
    FRAME_IDENTITY_INDEX_ORDER,
    SatNavOpenFlyDataset,
    load_annotation_records,
    sample_summary,
    sampling_policy_from_environment,
    resolve_original_dim_loss_weights,
    validate_records,
)
from baselines.vlm.openfly.native_core import (
    resolve_native_checkpoint_path,
    resolve_native_processor_source,
)
from satnav.evaluation.manifest import (
    ensure_manifest,
    payload_digest,
    read_manifest,
)


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASELINE_DIR / "configs" / "train.yaml"
DEFAULT_DEEPSPEED = BASELINE_DIR / "configs" / "zero2.json"
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
_DEEPSPEED_OPTIMIZER_RE = re.compile(
    r"(?:bf16_|fp16_)?zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune bundled OpenFly on validated SatNav trajectory_data"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--backend", choices=("continue", "scratch"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-validation-report", type=Path)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest")
    parser.add_argument("--action-format", choices=("compact", "original"))
    parser.add_argument("--unnorm-key")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-bounded-data", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"training config must be a mapping: {path}")
    return value


def _pick(args: argparse.Namespace, config: Mapping[str, Any], key: str, default: Any):
    value = getattr(args, key, None)
    return config.get(key, default) if value is None else value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"refusing to replace contradictory metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _source_identity(
    backend: str, model_path: Path, processor_path: Optional[Path]
) -> tuple[Mapping[str, Any], str]:
    if backend == "continue":
        identity = openfly_model_identity(model_path)
        return identity, str(processor_path or model_path)
    native_checkpoint, _ = resolve_native_checkpoint_path(str(model_path))
    processor = resolve_native_processor_source(
        str(model_path), str(processor_path) if processor_path else None
    )
    return native_source_identity(Path(native_checkpoint), Path(processor)), processor


def _prepare_data_validation(
    args: argparse.Namespace,
) -> tuple[Path, Mapping[str, Any]]:
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
            "launch with baselines/vlm/openfly/scripts/train.sh"
        )
    if not run_id:
        run_id = f"standalone-{os.getpid()}"
        os.environ["SATNAV_VALIDATION_RUN_ID"] = run_id
    if rank == 0:
        annotation, episode_path, records = load_annotation_records(
            args.trajectory_root,
            allow_external_paths=False,
            require_episode_metadata=True,
        )
        report = dict(
            validate_records(
                annotation,
                episode_path,
                records,
                policy=sampling_policy_from_environment(),
                strict_frames=True,
                hash_frame_content=True,
                decode_samples=0,
                frame_identity_index=report_path.with_name(
                    f"{report_path.name}.frames.sha256"
                ),
            ),
            allow_external_paths=False,
            validation_run_id=run_id,
            completed_at_ns=time.time_ns(),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.{run_id}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, report_path)
    else:
        deadline = time.monotonic() + 6 * 60 * 60
        while True:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("validation_run_id") == run_id:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for rank-0 data validation: {report_path}"
                )
            time.sleep(1)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validation_run_id") != run_id or report.get("status") != "passed":
        raise ValueError("data validation report is stale or failed")
    return report_path, report


def _bounded_value(name: str) -> Optional[int]:
    configured = os.environ.get(name, "").strip()
    if not configured:
        return None
    value = int(configured)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _checkpoint_has_optimizer(path: Path) -> bool:
    if (path / "optimizer.pt").is_file():
        return True
    return any(item.is_file() for item in path.glob("global_step*/*optim_states.pt"))


def _checkpoint_content_manifest(checkpoint: Path) -> Mapping[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    sentinel = checkpoint / ".openfly_checkpoint_complete"
    files = tuple(
        path for path in checkpoint.rglob("*") if path.is_file() and path != sentinel
    )
    return artifact_manifest(checkpoint, files)


def _stable_validation(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"validation_run_id", "completed_at_ns"}
    }


def _identity_digest(value: Mapping[str, Any]) -> str:
    return str(value.get("digest", value.get("sha256", "missing")))


def _resolve_frame_identity_index(
    validation_path: Path, report: Mapping[str, Any]
) -> tuple[Path, Mapping[str, Any]]:
    metadata = report.get("frame_identity_index")
    if not isinstance(metadata, Mapping):
        raise ValueError("data validation report lacks a frame identity index")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("frame identity index lacks an artifact identity")
    artifact_id = str(artifact.get("id", ""))
    portable = PurePosixPath(artifact_id.replace("\\", "/"))
    if (
        not artifact_id
        or portable.is_absolute()
        or len(portable.parts) != 1
        or portable.name != artifact_id
    ):
        raise ValueError("frame identity index artifact id is unsafe")
    return validation_path.with_name(artifact_id), artifact


def _require_same_identity(
    label: str, expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    if dict(actual) != dict(expected):
        raise RuntimeError(
            f"{label} changed during training setup: "
            f"expected={_identity_digest(expected)}, "
            f"actual={_identity_digest(actual)}"
        )


def _validate_dataset_against_report(
    dataset: SatNavOpenFlyDataset,
    report: Mapping[str, Any],
    *,
    max_episodes: Optional[int],
    max_samples: Optional[int],
) -> Mapping[str, Any]:
    if (
        report.get("status") != "passed"
        or report.get("strict_frames") is not True
        or report.get("allow_external_paths") is not False
    ):
        raise ValueError("data validation report lacks strict fail-closed settings")
    frame_tree = report.get("frame_tree")
    if not isinstance(frame_tree, Mapping):
        raise ValueError("data validation report lacks a frame-tree identity")
    digest = str(frame_tree.get("digest", ""))
    try:
        digest_is_sha256 = len(digest) == 64 and int(digest, 16) >= 0
    except ValueError:
        digest_is_sha256 = False
    if (
        frame_tree.get("algorithm") != "sha256-relative-path-size-content-v1"
        or not digest_is_sha256
        or int(frame_tree.get("file_count", -1)) != int(report.get("mapped_frames", -2))
        or int(frame_tree.get("total_bytes", 0)) <= 0
    ):
        raise ValueError("data validation frame-tree identity is invalid")
    frame_index = report.get("frame_identity_index")
    if not isinstance(frame_index, Mapping):
        raise ValueError("data validation report lacks per-frame identities")
    frame_index_artifact = frame_index.get("artifact")
    if not isinstance(frame_index_artifact, Mapping):
        raise ValueError("data validation frame index artifact is invalid")
    mapped_frame_count = int(report.get("mapped_frames", -1))
    if (
        frame_index.get("algorithm") != FRAME_IDENTITY_INDEX_ALGORITHM
        or int(frame_index.get("digest_size_bytes", -1)) != FRAME_DIGEST_BYTES
        or frame_index.get("ordering") != FRAME_IDENTITY_INDEX_ORDER
        or int(frame_index.get("file_count", -1)) != mapped_frame_count
        or int(frame_index_artifact.get("size", -1))
        != mapped_frame_count * FRAME_DIGEST_BYTES
        or dataset.frame_identity_index_file_count != mapped_frame_count
        or dataset.frame_identity_index_artifact != frame_index_artifact
        or dataset.verify_frame_identity_index() != frame_index_artifact
    ):
        raise RuntimeError("per-frame identity index contradicts validation report")
    if dataset.episode_path is None:
        raise ValueError("OpenFly training dataset has no episode metadata")
    annotation = file_identity(dataset.annotation_path)
    episode = file_identity(dataset.episode_path)
    if annotation != report.get("annotation_artifact"):
        raise RuntimeError("annotation file changed after full-content validation")
    if episode != report.get("episode_metadata_artifact"):
        raise RuntimeError("episode metadata changed after full-content validation")

    record_count = len(dataset.records)
    mapped_frames = sum(len(record.actions) for record in dataset.records)
    primitive_actions = sum(len(record.actions) - 1 for record in dataset.records)
    type_counts = dict(
        sorted(Counter(record.trajectory_type for record in dataset.records).items())
    )
    summary = sample_summary(dataset.records, dataset.policy)
    validated_annotations = int(report.get("annotations", -1))
    validated_samples = int(report.get("samples", -1))
    full_record_crosscheck = max_episodes is None
    if full_record_crosscheck:
        expected_fields = {
            "annotations": record_count,
            "mapped_frames": mapped_frames,
            "primitive_actions_excluding_init": primitive_actions,
            "episode_trajectory_type_counts": type_counts,
            "samples": summary["samples"],
            "action_counts": summary["action_counts"],
            "trajectory_type_counts": summary["trajectory_type_counts"],
            "sampling_policy": summary["sampling_policy"],
        }
        mismatches = {
            key: {"expected": value, "report": report.get(key)}
            for key, value in expected_fields.items()
            if report.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "data validation report contradicts the loaded dataset: "
                + json.dumps(mismatches, sort_keys=True)
            )
    else:
        if record_count != min(int(max_episodes), validated_annotations):
            raise RuntimeError(
                "bounded dataset record count contradicts validation report"
            )
        bounded_counts = (
            (mapped_frames, int(report.get("mapped_frames", -1))),
            (
                primitive_actions,
                int(report.get("primitive_actions_excluding_init", -1)),
            ),
            (int(summary["samples"]), validated_samples),
        )
        if any(local < 0 or full < local for local, full in bounded_counts):
            raise RuntimeError("bounded dataset counts exceed full validation report")
        if dataset.policy.to_payload() != report.get("sampling_policy"):
            raise RuntimeError("bounded dataset sampling policy contradicts validation")

    expected_selected = int(summary["samples"])
    if max_samples is not None:
        expected_selected = min(expected_selected, int(max_samples))
    if (
        len(dataset) != expected_selected
        or sum(dataset.action_counts.values()) != len(dataset)
        or sum(dataset.trajectory_type_counts.values()) != len(dataset)
    ):
        raise RuntimeError("selected OpenFly samples contradict dataset counters")
    if max_samples is None and (
        {str(key): int(value) for key, value in dataset.action_counts.items()}
        != summary["action_counts"]
        or dict(sorted(dataset.trajectory_type_counts.items()))
        != summary["trajectory_type_counts"]
    ):
        raise RuntimeError("loaded dataset counters contradict full sample summary")
    return {
        "status": "passed",
        "annotation_artifact": annotation,
        "episode_metadata_artifact": episode,
        "full_record_crosscheck": full_record_crosscheck,
        "records_loaded": record_count,
        "selected_samples": len(dataset),
        "validated_annotations": validated_annotations,
        "validated_samples": validated_samples,
        "mapped_frames": mapped_frames,
        "primitive_actions_excluding_init": primitive_actions,
        "sampling_policy": dataset.policy.to_payload(),
        "frame_tree_digest": digest,
        "frame_identity_index": dict(frame_index),
    }


def _require_dataset_files_unchanged(
    dataset: SatNavOpenFlyDataset, crosscheck: Mapping[str, Any]
) -> None:
    annotation = crosscheck.get("annotation_artifact")
    episode = crosscheck.get("episode_metadata_artifact")
    if not isinstance(annotation, Mapping) or not isinstance(episode, Mapping):
        raise ValueError("dataset crosscheck lacks source-file identities")
    if dataset.episode_path is None:
        raise ValueError("OpenFly training dataset has no episode metadata")
    _require_same_identity(
        "annotation file", annotation, file_identity(dataset.annotation_path)
    )
    _require_same_identity(
        "episode metadata", episode, file_identity(dataset.episode_path)
    )
    if dataset.frame_identity_index_path is None:
        raise ValueError("OpenFly training dataset has no frame identity index")
    _require_same_identity(
        "frame identity index",
        dataset.frame_identity_index_artifact,
        file_identity(dataset.frame_identity_index_path),
    )
    dataset.verify_frame_identity_index()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def _validate_output_mode(output: Path, resume_requested: bool) -> None:
    output = Path(output).expanduser()
    if output.is_symlink():
        raise ValueError(f"training output directory must not be a symlink: {output}")
    if resume_requested:
        if not output.is_dir():
            raise ValueError("resume requires an existing output directory")
        return
    if output.exists():
        if not output.is_dir():
            raise ValueError("fresh training output exists and is not a directory")
        if any(output.iterdir()):
            raise ValueError(
                "fresh training requires a nonexistent or empty output directory"
            )


def _validate_output_paths(
    *,
    output: Path,
    validation_report: Path,
    model: Path,
    processor: Optional[Path],
    trajectory: Path,
    configuration: Sequence[Path] = (),
) -> None:
    protected = [model, trajectory, BASELINE_DIR, *configuration]
    if processor is not None:
        protected.append(processor)
    for path in protected:
        if _paths_overlap(output, path):
            raise ValueError(f"output directory overlaps protected input path: {path}")
        if _paths_overlap(validation_report, path):
            raise ValueError(
                f"data validation report overlaps protected input path: {path}"
            )
    try:
        Path(validation_report).resolve().relative_to(Path(output).resolve())
    except ValueError as error:
        raise ValueError(
            "data validation report must be stored inside --output-dir"
        ) from error


def resolve_resume_checkpoint(
    output_dir: Path, requested: Optional[str]
) -> Optional[Path]:
    if requested is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    root_manifest = read_manifest(output / "training_manifest.json")
    expected_digest = str(root_manifest["digest"])
    if requested == "latest":
        candidates = []
        for path in output.glob("checkpoint-*"):
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"no checkpoint-N directory under {output}")
        rejected = []
        for _, candidate in sorted(candidates, reverse=True):
            try:
                audit_resumable_checkpoint(
                    candidate, expected_manifest_digest=expected_digest
                )
            except (OSError, ValueError, RuntimeError) as error:
                rejected.append(f"{candidate.name}: {error}")
                continue
            if rejected:
                print(
                    "[OpenFly resume] skipped incomplete newer checkpoints: "
                    + " | ".join(rejected),
                    flush=True,
                )
            return candidate.resolve()
        raise ValueError(
            "no complete matching checkpoint-N is resumable: " + " | ".join(rejected)
        )
    else:
        checkpoint = Path(os.path.abspath(Path(requested).expanduser()))
    try:
        checkpoint.relative_to(output)
    except ValueError as error:
        raise ValueError("resume checkpoint must be inside --output-dir") from error
    if not _CHECKPOINT_RE.fullmatch(checkpoint.name) or not checkpoint.is_dir():
        raise ValueError(f"invalid resume checkpoint directory: {checkpoint}")
    audit_resumable_checkpoint(checkpoint, expected_manifest_digest=expected_digest)
    return checkpoint.resolve()


def audit_resumable_checkpoint(
    checkpoint: Path, *, expected_manifest_digest: Optional[str] = None
) -> Mapping[str, Any]:
    unresolved = Path(checkpoint).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"checkpoint-N directory must not be a symlink: {unresolved}")
    checkpoint = unresolved.resolve()
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        raise ValueError(f"invalid checkpoint-N directory: {checkpoint}")
    symlinks = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(f"resume checkpoint contains symlinks: {symlinks}")
    required = (
        checkpoint / "trainer_state.json",
        checkpoint / "scheduler.pt",
        checkpoint / "training_manifest.json",
        checkpoint / "backend_meta.json",
        checkpoint / "dataset_statistics.json",
        checkpoint / ".openfly_checkpoint_complete",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing or not _checkpoint_has_optimizer(checkpoint):
        raise ValueError(
            f"resume checkpoint lacks trainer/optimizer state: missing={missing}"
        )
    empty = [path.name for path in required[:-1] if path.stat().st_size <= 0]
    if empty:
        raise ValueError(f"resume checkpoint contains empty state files: {empty}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    directory_step = int(match.group(1))
    global_step = state.get("global_step")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step != directory_step
    ):
        raise ValueError("checkpoint dirname and trainer_state global_step disagree")
    checkpoint_manifest = read_manifest(checkpoint / "training_manifest.json")
    if (
        expected_manifest_digest is not None
        and checkpoint_manifest["digest"] != expected_manifest_digest
    ):
        raise ValueError("checkpoint training manifest differs from requested run")
    complete = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )
    if (
        not isinstance(complete, Mapping)
        or int(complete.get("global_step", -1)) != directory_step
        or complete.get("status") != "complete"
        or complete.get("training_manifest_digest") != checkpoint_manifest["digest"]
    ):
        raise ValueError("checkpoint completion sentinel is invalid")
    content = _checkpoint_content_manifest(checkpoint)
    expected_content = complete.get("content_manifest")
    if (
        not isinstance(expected_content, Mapping)
        or expected_content.get("digest") != content["digest"]
        or int(expected_content.get("file_count", -1)) != content["file_count"]
        or int(expected_content.get("total_bytes", -1)) != content["total_bytes"]
    ):
        raise ValueError("checkpoint content does not match completion sentinel")
    world_size = int(
        checkpoint_manifest["payload"]["configuration"]["distributed"]["world_size"]
    )
    if world_size <= 0:
        raise ValueError("checkpoint training manifest has invalid world_size")
    scheduler_state = _load_torch_mapping(checkpoint / "scheduler.pt", "scheduler")
    if not isinstance(scheduler_state.get("last_epoch"), int) or not isinstance(
        scheduler_state.get("_step_count"), int
    ):
        raise ValueError("scheduler state lacks integer last_epoch/_step_count")
    if int(scheduler_state["last_epoch"]) != directory_step:
        raise ValueError("scheduler last_epoch does not match checkpoint step")

    rng_states = sorted(checkpoint.glob("rng_state*.pth"))
    expected_rng = (
        {"rng_state.pth"}
        if world_size == 1
        else {f"rng_state_{rank}.pth" for rank in range(world_size)}
    )
    actual_rng = {path.name for path in rng_states}
    if actual_rng != expected_rng:
        raise ValueError(
            "checkpoint RNG rank set mismatch: "
            f"missing={sorted(expected_rng - actual_rng)}, "
            f"unexpected={sorted(actual_rng - expected_rng)}"
        )
    for path in rng_states:
        rng = _load_torch_mapping(path, f"RNG state {path.name}")
        if not {"python", "numpy", "cpu", "cuda"}.issubset(rng):
            raise ValueError(f"RNG state has incomplete structure: {path.name}")
        if not _valid_python_rng_state(rng["python"]) or not _valid_numpy_rng_state(
            rng["numpy"]
        ):
            raise ValueError(f"RNG Python/NumPy state is invalid: {path.name}")
        if not _nonempty_tensor(rng["cpu"]):
            raise ValueError(f"RNG CPU state is invalid: {path.name}")
        cuda_state = rng["cuda"]
        cuda_states = (
            cuda_state if isinstance(cuda_state, (list, tuple)) else [cuda_state]
        )
        if not cuda_states or not all(_nonempty_tensor(value) for value in cuda_states):
            raise ValueError(f"RNG CUDA state is invalid: {path.name}")

    optimizer_path = checkpoint / "optimizer.pt"
    global_step_entries = sorted(checkpoint.glob("global_step*"))
    optimizer_states: list[Path]
    model_states: list[Path]
    if optimizer_path.is_file():
        if world_size != 1 or global_step_entries:
            raise ValueError(
                "non-DeepSpeed optimizer state has contradictory rank shards"
            )
        optimizer = _load_torch_mapping(optimizer_path, "optimizer")
        if (
            not isinstance(optimizer.get("state"), Mapping)
            or not optimizer["state"]
            or not isinstance(optimizer.get("param_groups"), list)
            or not optimizer["param_groups"]
            or not all(
                isinstance(group, Mapping) for group in optimizer["param_groups"]
            )
            or not all(
                isinstance(value, Mapping) and value
                for value in optimizer["state"].values()
            )
            or not all(
                isinstance(group.get("params"), list) and group["params"]
                for group in optimizer["param_groups"]
            )
        ):
            raise ValueError("optimizer state is semantically empty or malformed")
        optimizer_states = []
        model_states = []
    else:
        expected_global = f"global_step{directory_step}"
        if [path.name for path in global_step_entries] != [expected_global]:
            raise ValueError(
                "DeepSpeed checkpoint global_step directory set mismatch: "
                f"{[path.name for path in global_step_entries]}"
            )
        global_dir = global_step_entries[0]
        if not global_dir.is_dir():
            raise ValueError("DeepSpeed global_step entry is not a directory")
        optimizer_states = sorted(global_dir.glob("*optim_states.pt"))
        optimizer_ranks = []
        for path in optimizer_states:
            rank_match = _DEEPSPEED_OPTIMIZER_RE.fullmatch(path.name)
            if rank_match is None:
                raise ValueError(f"unexpected DeepSpeed optimizer shard: {path.name}")
            optimizer_ranks.append(int(rank_match.group(1)))
        if optimizer_ranks != list(range(world_size)):
            raise ValueError(
                "DeepSpeed optimizer rank set mismatch: "
                f"expected={list(range(world_size))}, actual={optimizer_ranks}"
            )
        for path in optimizer_states:
            state = _load_torch_mapping(path, f"DeepSpeed optimizer {path.name}")
            nested = state.get("optimizer_state_dict")
            if not isinstance(nested, Mapping):
                raise ValueError(
                    f"DeepSpeed optimizer structure is invalid: {path.name}"
                )
            partition_count = nested.get("partition_count")
            if isinstance(partition_count, list):
                valid_partition = partition_count and all(
                    int(value) == world_size for value in partition_count
                )
            else:
                valid_partition = int(partition_count or -1) == world_size
            if int(nested.get("zero_stage", -1)) != 2 or not valid_partition:
                raise ValueError(
                    f"DeepSpeed optimizer partition metadata is invalid: {path.name}"
                )
            base_optimizer = nested.get("base_optimizer_state")
            partitions = nested.get("single_partition_of_fp32_groups")
            if (
                not isinstance(base_optimizer, Mapping)
                or not isinstance(base_optimizer.get("state"), Mapping)
                or not base_optimizer["state"]
                or not isinstance(base_optimizer.get("param_groups"), list)
                or not base_optimizer["param_groups"]
                or not all(
                    isinstance(value, Mapping) and value
                    for value in base_optimizer["state"].values()
                )
                or not all(
                    isinstance(group, Mapping)
                    and isinstance(group.get("params"), list)
                    and group["params"]
                    for group in base_optimizer["param_groups"]
                )
                or not isinstance(partitions, list)
                or not partitions
                or not all(_nonempty_tensor(value) for value in partitions)
            ):
                raise ValueError(
                    f"DeepSpeed optimizer payload is empty or invalid: {path.name}"
                )
        model_states = sorted(global_dir.glob("*model_states.pt"))
        if [path.name for path in model_states] != ["mp_rank_00_model_states.pt"]:
            raise ValueError(
                "DeepSpeed model-state shard set mismatch: "
                f"{[path.name for path in model_states]}"
            )
        expected_state_files = {path.name for path in optimizer_states + model_states}
        actual_state_entries = {path.name for path in global_dir.iterdir()}
        if actual_state_entries != expected_state_files:
            raise ValueError(
                "DeepSpeed checkpoint state-file set mismatch: "
                f"missing={sorted(expected_state_files - actual_state_entries)}, "
                f"unexpected={sorted(actual_state_entries - expected_state_files)}"
            )
        model_state = _load_torch_mapping(model_states[0], "DeepSpeed model state")
        if (
            int(model_state.get("global_steps", -1)) != directory_step
            or int(model_state.get("dp_world_size", -1)) != world_size
            or int(model_state.get("mp_world_size", -1)) != 1
            or not isinstance(model_state.get("module"), Mapping)
            or not model_state["module"]
            or not isinstance(model_state.get("param_shapes"), list)
            or not model_state["param_shapes"]
        ):
            raise ValueError(
                "DeepSpeed model state has invalid structure or step metadata"
            )
    identity = openfly_model_identity(checkpoint)
    return {
        "status": "passed",
        "global_step": directory_step,
        "world_size": world_size,
        "training_manifest_digest": checkpoint_manifest["digest"],
        "optimizer_files": sorted(
            item.relative_to(checkpoint).as_posix()
            for item in (
                optimizer_states if optimizer_states else [checkpoint / "optimizer.pt"]
            )
        ),
        "rng_files": sorted(item.name for item in rng_states),
        "checkpoint_digest": identity["digest"],
        "content_manifest_digest": content["digest"],
    }


def _load_torch_mapping(path: Path, label: str) -> Mapping[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} file is missing, empty, or a symlink: {path.name}")
    import torch

    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:
        raise ValueError(f"cannot load {label}: {path.name}: {error}") from error
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping: {path.name}")
    return value


def _nonempty_tensor(value: Any) -> bool:
    return bool(
        hasattr(value, "numel") and hasattr(value, "dtype") and int(value.numel()) > 0
    )


def _valid_python_rng_state(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    version, internal, gaussian = value
    return bool(
        isinstance(version, int)
        and not isinstance(version, bool)
        and isinstance(internal, tuple)
        and internal
        and all(
            isinstance(item, int) and not isinstance(item, bool) for item in internal
        )
        and (
            gaussian is None
            or (
                isinstance(gaussian, (int, float))
                and not isinstance(gaussian, bool)
                and math.isfinite(float(gaussian))
            )
        )
    )


def _valid_numpy_rng_state(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) != 5:
        return False
    algorithm, keys, position, has_gaussian, cached_gaussian = value
    return bool(
        isinstance(algorithm, str)
        and algorithm
        and hasattr(keys, "size")
        and int(keys.size) > 0
        and isinstance(position, int)
        and not isinstance(position, bool)
        and position >= 0
        and isinstance(has_gaussian, int)
        and not isinstance(has_gaussian, bool)
        and has_gaussian in (0, 1)
        and isinstance(cached_gaussian, (int, float))
        and not isinstance(cached_gaussian, bool)
        and math.isfinite(float(cached_gaussian))
    )


def _training_manifest_payload(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    processor_source_identity: Mapping[str, Any],
    comparison_model_identity: Mapping[str, Any],
    validation_path: Path,
    validation: Mapping[str, Any],
    data_crosscheck: Mapping[str, Any],
    dataset: SatNavOpenFlyDataset,
    effective: Mapping[str, Any],
) -> Mapping[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device = int(effective["per_device_batch_size"])
    accumulation = int(effective["gradient_accumulation_steps"])
    return {
        "manifest_type": "openfly_training",
        "schema_version": 1,
        "backend": args.backend,
        "source_model": source_identity,
        "comparison_model": comparison_model_identity,
        "processor_source": processor_source_identity,
        "trajectory": {
            "annotation": file_identity(dataset.annotation_path),
            "episode_metadata": file_identity(dataset.episode_path),
            "validation_report_id": validation_path.name,
            "validation_payload_digest": payload_digest(_stable_validation(validation)),
            "frame_tree": validation["frame_tree"],
            "annotation_count": int(validation["annotations"]),
            "sample_count": len(dataset),
            "sampling_policy": dataset.policy.to_payload(),
            "trajectory_type_counts": dict(dataset.trajectory_type_counts),
            "action_counts": {
                str(key): int(value) for key, value in dataset.action_counts.items()
            },
            "allow_external_paths": False,
            "dataset_crosscheck": data_crosscheck,
        },
        "configuration": {
            "train": file_identity(Path(args.config).resolve()),
            "deepspeed": file_identity(Path(args.deepspeed_config).resolve()),
            "integration_source": _integration_source_identity(),
            "effective": dict(effective),
            "distributed": {
                "world_size": world_size,
                "per_device_batch_size": per_device,
                "gradient_accumulation_steps": accumulation,
                "global_microbatch_size": world_size * per_device,
                "global_update_batch_size": world_size * per_device * accumulation,
            },
        },
    }


def _integration_source_identity() -> Mapping[str, Any]:
    files = list(BASELINE_DIR.glob("*.py"))
    for directory in (
        "backends",
        "native_core",
        "openfly_core",
        "configs",
        "environment",
        "scripts",
    ):
        files.extend(
            path
            for path in (BASELINE_DIR / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml", ".yml", ".json", ".sh"}
        )
    files.extend(
        BASELINE_DIR / name
        for name in ("requirements.txt", "UPSTREAM.md", "NOTICE", "LICENSE.upstream")
    )
    return artifact_manifest(BASELINE_DIR, files)


def _parameter_probe(model: Any) -> tuple[str, Any]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "projector" in name
    ]
    if not candidates:
        candidates = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    if not candidates:
        raise ValueError("OpenFly model has no trainable parameters")
    name, parameter = min(candidates, key=lambda item: item[1].numel())
    return name, parameter.detach().float().cpu().clone()


def _parameter_delta(model: Any, name: str, before: Any) -> Mapping[str, Any]:
    import torch

    parameters = dict(model.named_parameters())
    if name not in parameters:
        raise ValueError(f"parameter probe disappeared: {name}")
    after = parameters[name].detach().float().cpu()
    if tuple(after.shape) != tuple(before.shape):
        raise ValueError(f"parameter probe changed shape: {name}")
    delta = (after - before).abs()
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise RuntimeError("parameter probe contains NaN or infinity")
    if not torch.isfinite(delta).all():
        raise RuntimeError("parameter delta contains NaN or infinity")
    result = {
        "tensor": name,
        "elements": int(delta.numel()),
        "changed_elements": int(torch.count_nonzero(delta).item()),
        "max_abs_delta": float(delta.max().item()),
        "l2_delta": float(torch.linalg.vector_norm(delta).item()),
    }
    if result["changed_elements"] <= 0 or result["max_abs_delta"] <= 0:
        raise RuntimeError("training completed without a nonzero parameter update")
    return result


def _effective_config(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    training = dict(config.get("training", {}))
    model = dict(config.get("model", {}))
    data = dict(config.get("data", {}))
    configured_weights = data.get("original_dim_loss_weights")
    weights = resolve_original_dim_loss_weights(configured_weights)
    return {
        "action_format": _pick(args, data, "action_format", "compact"),
        "unnorm_key": _pick(args, data, "unnorm_key", "satnav_original"),
        "original_dim_loss_weights": list(weights),
        "grid_size": int(model.get("grid_size", 16)),
        "torch_dtype": _pick(args, model, "torch_dtype", "bfloat16"),
        "use_flash_attention_2": not args.no_flash_attention
        and bool(model.get("use_flash_attention_2", True)),
        "gradient_checkpointing": not args.no_gradient_checkpointing
        and bool(training.get("gradient_checkpointing", True)),
        "num_train_epochs": float(_pick(args, training, "num_train_epochs", 1.0)),
        "max_steps": int(_pick(args, training, "max_steps", -1)),
        "learning_rate": float(_pick(args, training, "learning_rate", 2e-5)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "warmup_ratio": float(_pick(args, training, "warmup_ratio", 0.03)),
        "per_device_batch_size": int(_pick(args, training, "per_device_batch_size", 1)),
        "gradient_accumulation_steps": int(
            _pick(args, training, "gradient_accumulation_steps", 8)
        ),
        "dataloader_num_workers": int(
            _pick(args, training, "dataloader_num_workers", 4)
        ),
        "save_steps": int(_pick(args, training, "save_steps", 1000)),
        "save_total_limit": int(_pick(args, training, "save_total_limit", 2)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "seed": int(training.get("seed", 42)),
    }


def _validate_effective(effective: Mapping[str, Any]) -> None:
    finite_positive = ("num_train_epochs", "learning_rate")
    for key in finite_positive:
        value = float(effective[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    positive_integers = (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "save_steps",
        "save_total_limit",
        "logging_steps",
    )
    for key in positive_integers:
        if int(effective[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    weight_decay = float(effective["weight_decay"])
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    warmup_ratio = float(effective["warmup_ratio"])
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be finite and in [0, 1]")
    if int(effective["dataloader_num_workers"]) < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if int(effective["max_steps"]) == 0 or int(effective["max_steps"]) < -1:
        raise ValueError("max_steps must be -1 or positive")
    if int(effective["grid_size"]) != 16:
        raise ValueError("bundled OpenFly requires grid_size=16")


def _write_checkpoint_metadata(
    root: Path,
    *,
    processor: Any,
    backend_meta: Mapping[str, Any],
    dataset_statistics: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
) -> None:
    root = Path(root)
    match = _CHECKPOINT_RE.fullmatch(root.name)
    sentinel = root / ".openfly_checkpoint_complete"
    if match and sentinel.is_file():
        existing_manifest = read_manifest(root / "training_manifest.json")
        expected_manifest = ensure_manifest(
            root / "training_manifest.json", training_manifest
        )
        complete = json.loads(sentinel.read_text(encoding="utf-8"))
        if (
            existing_manifest["digest"] != expected_manifest["digest"]
            or int(complete.get("global_step", -1)) != int(match.group(1))
            or complete.get("status") != "complete"
            or complete.get("training_manifest_digest") != expected_manifest["digest"]
        ):
            raise ValueError(f"existing OpenFly checkpoint sentinel is invalid: {root}")
        _atomic_json(root / "backend_meta.json", backend_meta)
        _atomic_json(root / "dataset_statistics.json", dataset_statistics)
        content = _checkpoint_content_manifest(root)
        expected_content = complete.get("content_manifest", {})
        if (
            expected_content.get("digest") != content["digest"]
            or int(expected_content.get("file_count", -1)) != content["file_count"]
            or int(expected_content.get("total_bytes", -1)) != content["total_bytes"]
        ):
            raise ValueError(f"existing checkpoint content is not complete: {root}")
        return
    if (root / "backend_meta.json").is_file():
        _atomic_json(root / "backend_meta.json", backend_meta)
    if (root / "dataset_statistics.json").is_file():
        _atomic_json(root / "dataset_statistics.json", dataset_statistics)
    if (root / "training_manifest.json").is_file():
        ensure_manifest(root / "training_manifest.json", training_manifest)
    processor.save_pretrained(root)
    _atomic_json(root / "backend_meta.json", backend_meta)
    _atomic_json(root / "dataset_statistics.json", dataset_statistics)
    manifest = ensure_manifest(root / "training_manifest.json", training_manifest)
    if match:
        content = _checkpoint_content_manifest(root)
        _atomic_json(
            root / ".openfly_checkpoint_complete",
            {
                "global_step": int(match.group(1)),
                "status": "complete",
                "training_manifest_digest": manifest["digest"],
                "content_manifest": {
                    "algorithm": content["algorithm"],
                    "digest": content["digest"],
                    "file_count": content["file_count"],
                    "total_bytes": content["total_bytes"],
                },
            },
        )


def _trainer_class():
    import torch
    from transformers import Trainer

    class OpenFlyTrainer(Trainer):
        checkpoint_metadata: Mapping[str, Any]
        probe_name: Optional[str] = None
        probe_before: Any = None

        def training_step(self, model, inputs, *args, **kwargs):
            if self.probe_before is None:
                unwrapped = self.accelerator.unwrap_model(model)
                self.probe_name, self.probe_before = _parameter_probe(unwrapped)
            return super().training_step(model, inputs, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            loss_weights = inputs.pop("loss_weights", None)
            if loss_weights is None:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                visual_tokens = int(logits.shape[1] - labels.shape[1])
                if visual_tokens < 0:
                    raise ValueError("OpenFly logits are shorter than text labels")
                ignored = labels.new_full((labels.shape[0], visual_tokens), -100)
                zero_weights = loss_weights.new_zeros(
                    (loss_weights.shape[0], visual_tokens)
                )
                expanded_labels = torch.cat(
                    [labels[:, :1], ignored, labels[:, 1:]], dim=1
                )
                expanded_weights = torch.cat(
                    [loss_weights[:, :1], zero_weights, loss_weights[:, 1:]], dim=1
                )
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = expanded_labels[:, 1:].contiguous()
                shift_weights = expanded_weights[:, 1:].contiguous()
                token_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)
                denominator = shift_weights.sum()
                if float(denominator.detach().cpu()) <= 0:
                    raise ValueError("OpenFly original batch has zero loss weight")
                loss = (token_loss * shift_weights).sum() / denominator
            return (loss, outputs) if return_outputs else loss

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if self.args.should_save:
                checkpoint = (
                    Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                )
                _write_checkpoint_metadata(checkpoint, **self.checkpoint_metadata)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    return OpenFlyTrainer


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.backend = resolve_backend(args.backend)
    args.model_path = Path(args.model_path).expanduser().resolve()
    args.trajectory_root = Path(args.trajectory_root).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.config = Path(args.config).expanduser().resolve()
    args.deepspeed_config = Path(args.deepspeed_config).expanduser().resolve()
    prevalidated_resume: Optional[Path] = None
    if not args.print_config:
        resume_requested = args.resume_from_checkpoint is not None
        rank = int(os.environ.get("RANK", "0"))
        if resume_requested or rank == 0:
            _validate_output_mode(args.output_dir, resume_requested=resume_requested)
        prevalidated_resume = resolve_resume_checkpoint(
            args.output_dir, args.resume_from_checkpoint
        )
    config = _load_yaml(args.config)
    effective = _effective_config(args, config)
    _validate_effective(effective)
    if args.print_config:
        print(json.dumps(effective, indent=2, sort_keys=True))
        return 0
    caps = {
        name: _bounded_value(name)
        for name in ("SATNAV_MAX_EPISODES", "SATNAV_MAX_SAMPLES")
    }
    if (
        any(value is not None for value in caps.values())
        and not args.allow_bounded_data
    ):
        raise ValueError(
            "bounded data requires --allow-bounded-data so a cap cannot look like full training"
        )
    if args.processor_path is not None:
        args.processor_path = Path(args.processor_path).expanduser().resolve()
    if args.backend == "scratch":
        exact_checkpoint, _ = resolve_native_checkpoint_path(str(args.model_path))
        args.model_path = Path(exact_checkpoint).resolve()
    validation_candidate = (
        Path(args.data_validation_report or args.output_dir / "data_validation.json")
        .expanduser()
        .resolve()
    )
    source_identity, processor_source = _source_identity(
        args.backend, args.model_path, args.processor_path
    )
    resolved_processor = Path(processor_source).expanduser().resolve()
    processor_source_identity = processor_identity(resolved_processor)
    _validate_output_paths(
        output=args.output_dir,
        validation_report=validation_candidate,
        model=args.model_path,
        processor=resolved_processor,
        trajectory=args.trajectory_root,
        configuration=(args.config, args.deepspeed_config),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path, validation = _prepare_data_validation(args)
    frame_identity_index_path, frame_identity_index_artifact = (
        _resolve_frame_identity_index(validation_path, validation)
    )
    dataset = SatNavOpenFlyDataset(
        args.trajectory_root,
        action_format=str(effective["action_format"]),
        policy=sampling_policy_from_environment(),
        max_episodes=caps["SATNAV_MAX_EPISODES"],
        max_samples=caps["SATNAV_MAX_SAMPLES"],
        frame_identity_index=frame_identity_index_path,
        frame_identity_index_artifact=frame_identity_index_artifact,
    )
    data_crosscheck = _validate_dataset_against_report(
        dataset,
        validation,
        max_episodes=caps["SATNAV_MAX_EPISODES"],
        max_samples=caps["SATNAV_MAX_SAMPLES"],
    )
    comparison_cache_root = args.output_dir.with_name(
        f"{args.output_dir.name}.source-cache"
    )
    for protected in (
        args.model_path,
        resolved_processor,
        args.trajectory_root,
        args.output_dir,
        args.config,
        args.deepspeed_config,
        BASELINE_DIR,
    ):
        if _paths_overlap(comparison_cache_root, protected):
            raise ValueError(
                f"scratch comparison cache overlaps protected path: {protected}"
            )
    model_args = SimpleNamespace(
        backend=args.backend,
        model_name_or_path=str(args.model_path),
        processor_name_or_path=processor_source,
        cache_dir=os.environ.get("HF_HOME") or None,
        comparison_cache_root=str(comparison_cache_root),
        grid_size=int(effective["grid_size"]),
        torch_dtype=str(effective["torch_dtype"]),
        use_flash_attention_2=bool(effective["use_flash_attention_2"]),
    )
    data_args = SimpleNamespace(
        action_format=str(effective["action_format"]),
        unnorm_key=str(effective["unnorm_key"]),
        original_dim_loss_weights=tuple(effective["original_dim_loss_weights"]),
    )
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name or args.output_dir.name,
        deepspeed=str(args.deepspeed_config),
        num_train_epochs=float(effective["num_train_epochs"]),
        max_steps=int(effective["max_steps"]),
        per_device_train_batch_size=int(effective["per_device_batch_size"]),
        gradient_accumulation_steps=int(effective["gradient_accumulation_steps"]),
        learning_rate=float(effective["learning_rate"]),
        weight_decay=float(effective["weight_decay"]),
        warmup_ratio=float(effective["warmup_ratio"]),
        dataloader_num_workers=int(effective["dataloader_num_workers"]),
        save_strategy="steps",
        save_steps=int(effective["save_steps"]),
        save_total_limit=int(effective["save_total_limit"]),
        logging_steps=int(effective["logging_steps"]),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=bool(effective["gradient_checkpointing"]),
        bf16=str(effective["torch_dtype"]) == "bfloat16",
        fp16=str(effective["torch_dtype"]) == "float16",
        save_safetensors=True,
        seed=int(effective["seed"]),
        ddp_find_unused_parameters=False,
    )
    source_before_load, processor_before_load = _source_identity(
        args.backend, args.model_path, args.processor_path
    )
    if Path(processor_before_load).expanduser().resolve() != resolved_processor:
        raise RuntimeError("OpenFly processor source path changed before model loading")
    _require_same_identity("OpenFly source", source_identity, source_before_load)
    _require_same_identity(
        "OpenFly processor",
        processor_source_identity,
        processor_identity(resolved_processor),
    )
    patch_accelerate_optimizer_train_eval()
    backend = build_train_backend(model_args, data_args, training_args)
    source_after_load, processor_after_load = _source_identity(
        args.backend, args.model_path, args.processor_path
    )
    if Path(processor_after_load).expanduser().resolve() != resolved_processor:
        raise RuntimeError("OpenFly processor source path changed during model loading")
    _require_same_identity("OpenFly source", source_identity, source_after_load)
    _require_same_identity(
        "OpenFly processor",
        processor_source_identity,
        processor_identity(resolved_processor),
    )
    source_metadata_key = (
        "source_model_identity" if args.backend == "continue" else "source_identity"
    )
    loaded_source = backend.backend_meta.get(source_metadata_key)
    loaded_processor = backend.backend_meta.get("processor_identity")
    loaded_comparison = backend.backend_meta.get("comparison_model_identity")
    if not isinstance(loaded_source, Mapping):
        raise RuntimeError("OpenFly backend omitted its loaded source identity")
    if not isinstance(loaded_processor, Mapping):
        raise RuntimeError("OpenFly backend omitted its loaded processor identity")
    if not isinstance(loaded_comparison, Mapping):
        raise RuntimeError("OpenFly backend omitted its comparison-model identity")
    _require_same_identity("loaded OpenFly source", source_identity, loaded_source)
    _require_same_identity(
        "loaded OpenFly processor", processor_source_identity, loaded_processor
    )
    comparison_model_path = Path(backend.comparison_model_path).expanduser().resolve()
    comparison_model_identity = openfly_model_identity(comparison_model_path)
    _require_same_identity(
        "OpenFly comparison model", loaded_comparison, comparison_model_identity
    )
    _require_dataset_files_unchanged(dataset, data_crosscheck)
    manifest_payload = _training_manifest_payload(
        args,
        config,
        source_identity=source_identity,
        processor_source_identity=processor_source_identity,
        comparison_model_identity=comparison_model_identity,
        validation_path=validation_path,
        validation=validation,
        data_crosscheck=data_crosscheck,
        dataset=dataset,
        effective=effective,
    )
    ensure_manifest(args.output_dir / "training_manifest.json", manifest_payload)
    resume = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume != prevalidated_resume:
        raise RuntimeError("resumable checkpoint selection changed during setup")
    statistics = {
        "format": "satnav_openfly_training_statistics",
        "action_format": data_args.action_format,
        "num_samples": len(dataset),
        "num_episodes": len(dataset.records),
        "action_counts": {
            str(key): int(value) for key, value in dataset.action_counts.items()
        },
        "trajectory_type_counts": dict(dataset.trajectory_type_counts),
        "sampling_policy": dataset.policy.to_payload(),
        "action_metadata": backend.metadata,
    }
    checkpoint_metadata = {
        "processor": backend.processor,
        "backend_meta": backend.backend_meta,
        "dataset_statistics": statistics,
        "training_manifest": manifest_payload,
    }
    trainer_cls = _trainer_class()
    trainer = trainer_cls(
        model=backend.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=backend.data_collator,
        processing_class=backend.processor,
    )
    trainer.checkpoint_metadata = checkpoint_metadata
    start_step = 0
    if resume is not None:
        resume_state = json.loads(
            (resume / "trainer_state.json").read_text(encoding="utf-8")
        )
        start_step = int(resume_state["global_step"])
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.state.global_step <= start_step:
        raise RuntimeError("OpenFly trainer completed without a new optimizer step")
    if trainer.probe_name is None or trainer.probe_before is None:
        raise RuntimeError("OpenFly trainer executed no training batch")
    optimizer_probe = _parameter_delta(
        backend.model, trainer.probe_name, trainer.probe_before
    )
    if training_args.should_save:
        _require_dataset_files_unchanged(dataset, data_crosscheck)
        trainer.save_model(str(args.output_dir))
        _write_checkpoint_metadata(args.output_dir, **checkpoint_metadata)
        _atomic_replace_json(
            args.output_dir / "optimizer_step_probe.json",
            {
                **optimizer_probe,
                "start_global_step": start_step,
                "global_step": int(trainer.state.global_step),
                "status": "passed",
            },
        )
        from baselines.vlm.openfly.checkpoint import parameter_delta

        full_delta = parameter_delta(
            comparison_model_path,
            args.output_dir,
            tensor_name=trainer.probe_name,
        )
        _atomic_replace_json(
            args.output_dir / "parameter_delta.json",
            {
                **full_delta,
                "start_global_step": start_step,
                "global_step": int(trainer.state.global_step),
            },
        )
        checkpoints = []
        for path in args.output_dir.glob("checkpoint-*"):
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if match and int(match.group(1)) > start_step:
                checkpoints.append(path)
        if not checkpoints:
            raise RuntimeError("training produced no new checkpoint after resume start")
        manifest_digest = read_manifest(args.output_dir / "training_manifest.json")[
            "digest"
        ]
        for path in checkpoints:
            audit_resumable_checkpoint(
                path, expected_manifest_digest=str(manifest_digest)
            )
        openfly_model_identity(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

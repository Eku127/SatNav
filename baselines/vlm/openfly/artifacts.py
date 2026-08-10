"""Content-addressed OpenFly model and training-input identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from baselines.vlm.openfly.actions import SUPPORTED_ACTION_FORMATS


class OpenFlyArtifactError(ValueError):
    """Raised when a checkpoint is incomplete or internally contradictory."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(entries), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_manifest(root: Path, files: Iterable[Path]) -> Mapping[str, Any]:
    root = Path(root).expanduser().resolve()
    paths = sorted({Path(path).expanduser().resolve() for path in files})
    entries = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"identity artifact not found: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise OpenFlyArtifactError(
                f"identity artifact is outside {root}: {path}"
            ) from error
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    if not entries:
        raise OpenFlyArtifactError(f"no identity artifacts found below {root}")
    return {
        "algorithm": "sha256-file-manifest-v1",
        "digest": _manifest_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "files": entries,
    }


def file_identity(path: Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "id": resolved.name,
        "sha256": file_sha256(resolved),
        "size": resolved.stat().st_size,
    }


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenFlyArtifactError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise OpenFlyArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _safe_shard(root: Path, value: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or len(portable.parts) != 1:
        raise OpenFlyArtifactError(f"unsafe checkpoint shard path: {value!r}")
    if not portable.name.endswith(".safetensors"):
        raise OpenFlyArtifactError(f"checkpoint shard is not safetensors: {value!r}")
    path = root / portable.name
    if path.is_symlink():
        raise OpenFlyArtifactError(f"checkpoint shard must not be a symlink: {value!r}")
    return path


def _weight_layout(root: Path) -> tuple[Sequence[Path], int, int]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise OpenFlyArtifactError("safetensors index has no weight_map")
        shards = sorted(
            {_safe_shard(root, str(value)) for value in weight_map.values()}
        )
        missing = [path.name for path in shards if not path.is_file()]
        if missing:
            raise OpenFlyArtifactError(f"checkpoint is missing shards: {missing}")
        resolved_shards = {shard.resolve() for shard in shards}
        unexpected = sorted(
            path.name
            for path in root.glob("*.safetensors")
            if path.resolve() not in resolved_shards
        )
        if unexpected:
            raise OpenFlyArtifactError(
                f"checkpoint has unreferenced safetensors files: {unexpected}"
            )
        metadata = index.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise OpenFlyArtifactError("safetensors index metadata must be an object")
        declared = int(metadata.get("total_size", 0))
        if declared < 0:
            raise OpenFlyArtifactError("safetensors total_size must be non-negative")
        return (index_path, *shards), len(weight_map), declared
    single = root / "model.safetensors"
    if not single.is_file():
        raise OpenFlyArtifactError(
            f"OpenFly checkpoint needs model.safetensors or an index: {root}"
        )
    if single.is_symlink():
        raise OpenFlyArtifactError("model.safetensors must not be a symlink")
    unexpected = sorted(
        path.name for path in root.glob("*.safetensors") if path != single
    )
    if unexpected:
        raise OpenFlyArtifactError(
            f"checkpoint has ambiguous safetensors files: {unexpected}"
        )
    return (single,), -1, int(single.stat().st_size)


_PROCESSOR_REQUIRED_NAMES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)
_PROCESSOR_OPTIONAL_NAMES = (
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "image_processor_config.json",
    "merges.txt",
    "processor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
)
_PROCESSOR_ARTIFACT_NAMES = (
    *_PROCESSOR_REQUIRED_NAMES,
    *_PROCESSOR_OPTIONAL_NAMES,
)
_INFERENCE_OPTIONAL_NAMES = (
    *_PROCESSOR_OPTIONAL_NAMES,
    "backend_meta.json",
    "dataset_statistics.json",
    "generation_config.json",
    "training_manifest.json",
)


def _inference_artifacts(
    root: Path, weight_files: Sequence[Path], required: Sequence[Path]
) -> tuple[Path, ...]:
    """Select only immutable files needed to construct or identify inference.

    Optimizer shards, trainer state, validation evidence, logs, and nested
    ``checkpoint-N`` directories deliberately belong to resumability/evidence
    manifests, not to the inference model identity.
    """

    selected = set(weight_files) | set(required)
    for name in _INFERENCE_OPTIONAL_NAMES:
        candidate = root / name
        if candidate.is_symlink() or candidate.exists():
            if candidate.is_symlink() or not candidate.is_file():
                raise OpenFlyArtifactError(
                    f"inference artifact must be a regular file: {candidate}"
                )
            selected.add(candidate)
    symlinks = sorted(path.name for path in selected if path.is_symlink())
    if symlinks:
        raise OpenFlyArtifactError(
            f"OpenFly inference artifacts contain symlinks: {symlinks}"
        )
    return tuple(sorted(selected))


def _canonical_backend(value: Any) -> str:
    rendered = str(value or "").strip().lower()
    aliases = {
        "hf": "continue",
        "continue": "continue",
        "native": "scratch",
        "scratch": "scratch",
    }
    if rendered not in aliases:
        raise OpenFlyArtifactError(f"unknown OpenFly backend metadata: {rendered!r}")
    return aliases[rendered]


def _declared_facts(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(config.get("model_type")) != "openvla":
        raise OpenFlyArtifactError("OpenFly config.model_type must be 'openvla'")
    action_format = str(config.get("action_format", "")).strip().lower()
    if action_format not in SUPPORTED_ACTION_FORMATS:
        raise OpenFlyArtifactError(
            "OpenFly config.action_format must be compact or original"
        )
    facts: dict[str, Any] = {
        "action_format": action_format,
        "model_type": "openvla",
        "architectures": list(config.get("architectures", ())),
        "grid_size": int(config.get("grid_size", 16)),
        "torch_dtype": str(config.get("torch_dtype", "unknown")),
        "use_cache": bool(config.get("use_cache", False)),
        "backend": "unknown",
    }
    statistics_path = root / "dataset_statistics.json"
    if statistics_path.is_file():
        statistics = _json_object(statistics_path)
        declared = str(statistics.get("action_format", "")).strip().lower()
        if declared != action_format:
            raise OpenFlyArtifactError(
                "dataset_statistics.action_format contradicts config.action_format"
            )
        facts["training_samples"] = int(statistics.get("num_samples", -1))
    backend_path = root / "backend_meta.json"
    if backend_path.is_file():
        backend = _json_object(backend_path)
        facts["backend"] = _canonical_backend(backend.get("backend"))
        declared = backend.get("action_format")
        if declared is not None and str(declared).strip().lower() != action_format:
            raise OpenFlyArtifactError(
                "backend_meta.action_format contradicts config.action_format"
            )
    return facts


def openfly_model_identity(model_path: Path) -> Mapping[str, Any]:
    """Hash root-level inference artifacts after validating model metadata."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = (
        root / "config.json",
        root / "preprocessor_config.json",
        root / "tokenizer_config.json",
        root / "tokenizer.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise OpenFlyArtifactError(f"OpenFly checkpoint is missing: {missing}")
    weight_files, tensor_count, declared_tensor_bytes = _weight_layout(root)
    config = _json_object(root / "config.json")
    facts = _declared_facts(root, config)
    inference_files = _inference_artifacts(root, weight_files, required)
    snapshot = artifact_manifest(root, inference_files)
    return {
        **snapshot,
        "scope": "openfly-inference-root-artifacts",
        "tensor_count": tensor_count,
        "declared_tensor_bytes": declared_tensor_bytes,
        "facts": facts,
    }


def processor_identity(processor_path: Path) -> Mapping[str, Any]:
    """Hash every file that defines OpenFly preprocessing and tokenization."""

    root = Path(processor_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = tuple(root / name for name in _PROCESSOR_REQUIRED_NAMES)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise OpenFlyArtifactError(f"OpenFly processor is missing: {missing}")
    config = _json_object(root / "config.json")
    if str(config.get("model_type")) != "openvla":
        raise OpenFlyArtifactError(
            "OpenFly processor config.model_type must be 'openvla'"
        )
    symlinks = [path.name for path in required if path.is_symlink()]
    if symlinks:
        raise OpenFlyArtifactError(f"OpenFly processor contains symlinks: {symlinks}")
    files = []
    for name in _PROCESSOR_ARTIFACT_NAMES:
        candidate = root / name
        if candidate.is_symlink():
            raise OpenFlyArtifactError(
                f"OpenFly processor artifact must not be a symlink: {candidate}"
            )
        if candidate.exists() and not candidate.is_file():
            raise OpenFlyArtifactError(
                f"OpenFly processor artifact must be a regular file: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    manifest = artifact_manifest(root, files)
    return {
        **manifest,
        "scope": "openfly-processor-and-tokenizer",
        "model_type": "openvla",
    }


def native_source_identity(
    native_checkpoint: Path, processor_path: Path
) -> Mapping[str, Any]:
    checkpoint = file_identity(Path(native_checkpoint))
    processor = processor_identity(Path(processor_path))
    summary = [
        {"component": "native_checkpoint", "digest": checkpoint["sha256"]},
        {"component": "processor", "digest": processor["digest"]},
    ]
    return {
        "algorithm": "sha256-openfly-native-source-v1",
        "digest": _manifest_digest(summary),
        "native_checkpoint": checkpoint,
        "processor": processor,
        "scope": "native-prismatic-plus-openfly-processor",
    }

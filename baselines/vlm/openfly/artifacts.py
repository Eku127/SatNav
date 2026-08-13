"""Content-addressed OpenFly model and training-input identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from baselines.vlm.openfly.model_facts import (
    OpenFlyArtifactError,
    _declared_facts,
    _json_object,
    _weight_layout,
)


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

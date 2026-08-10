"""Exact content identities for Uni-NaVid models and external assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_EVA_SHA256 = (
    "99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f"
)
EXPECTED_PROCESSOR_DIGEST = (
    "09b4f25b721d72544d8a0f5a67d0b7dd27608c05fb9261086c24b4e1cbe94803"
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
            raise ValueError(f"identity artifact is outside {root}: {path}") from error
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    if not entries:
        raise ValueError(f"no identity artifacts found below {root}")
    return {
        "algorithm": "sha256-file-manifest-v1",
        "digest": _manifest_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "files": entries,
    }


def directory_identity(root: Path) -> Mapping[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return artifact_manifest(root, (path for path in root.rglob("*") if path.is_file()))


def model_identity(model_path: Path) -> Mapping[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    required = (
        model_path / "config.json",
        model_path / "tokenizer.model",
        model_path / "tokenizer_config.json",
        model_path / "special_tokens_map.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"incomplete Uni-NaVid checkpoint: {path}")
    weights = tuple(model_path.glob("pytorch_model*.bin")) + tuple(
        model_path.glob("model*.safetensors")
    )
    if not weights:
        raise FileNotFoundError(f"no full checkpoint weights in {model_path}")
    optional_names = (
        "added_tokens.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
    )
    files = [*required, *weights]
    files.extend(
        model_path / name
        for name in optional_names
        if (model_path / name).is_file()
    )
    return dict(
        artifact_manifest(model_path, files),
        scope="complete-uninavid-inference-checkpoint",
    )


def external_asset_identities(
    eva_path: Path, processor_path: Path
) -> Mapping[str, Any]:
    eva = file_identity(eva_path)
    processor = directory_identity(processor_path)
    if eva["sha256"] != EXPECTED_EVA_SHA256:
        raise ValueError(
            "EVA checkpoint digest mismatch: "
            f"expected {EXPECTED_EVA_SHA256}, found {eva['sha256']}"
        )
    if processor["digest"] != EXPECTED_PROCESSOR_DIGEST:
        raise ValueError(
            "Uni-NaVid processor digest mismatch: "
            f"expected {EXPECTED_PROCESSOR_DIGEST}, found {processor['digest']}"
        )
    return {"eva": eva, "processor": processor}


def file_identity(path: Path) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"id": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}

"""Root-relative exact artifact identities for NaVILA train/eval manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
    """Hash every named file and record paths relative to the artifact root."""

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


def _all_regular_files(root: Path) -> Sequence[Path]:
    return tuple(path for path in Path(root).rglob("*") if path.is_file())


def navila_model_identity(model_path: Path) -> Mapping[str, Any]:
    """Cover the full composite checkpoint, tokenizer, projector, and tower."""

    model_path = Path(model_path).expanduser().resolve()
    root_config = model_path / "config.json"
    if not root_config.is_file():
        raise FileNotFoundError(f"NaVILA root config not found: {root_config}")
    components = {}
    for name in ("llm", "mm_projector", "vision_tower"):
        component_root = model_path / name
        if not component_root.is_dir():
            raise FileNotFoundError(f"NaVILA component not found: {component_root}")
        components[name] = dict(
            artifact_manifest(component_root, _all_regular_files(component_root)),
            scope=f"complete-{name}-snapshot",
        )
    root_files = tuple(path for path in model_path.iterdir() if path.is_file())
    root = dict(
        artifact_manifest(model_path, root_files),
        scope="complete-root-file-snapshot",
    )
    summary = [
        {"component": "root", "digest": root["digest"]},
        *[
            {"component": name, "digest": components[name]["digest"]}
            for name in sorted(components)
        ],
    ]
    return {
        "algorithm": "sha256-composite-model-v1",
        "digest": _manifest_digest(summary),
        "root": root,
        "components": components,
        "scope": "complete-navila-composite-checkpoint",
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

"""Portable content identities for evaluation data artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from satnav.dataset.scene_resolver import SceneResolver
from satnav.evaluation.manifest import payload_digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_file(root: Path, episode: Any, logical_id: str) -> Path:
    configured = (
        episode.get("scene_path")
        if isinstance(episode, Mapping)
        else getattr(episode, "scene_path", None)
    )
    candidates = []
    if configured:
        path = Path(str(configured)).expanduser()
        candidates.extend((path, Path(str(path) + ".tif")))
    candidates.extend((root / logical_id, root / f"{logical_id}.tif"))
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"scene artifact cannot be a symlink: {logical_id}")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"scene artifact is outside the configured scene root: {logical_id}"
            ) from error
        return resolved
    raise FileNotFoundError(f"scene artifact does not exist: {logical_id}")


def referenced_scene_identity(
    scenes_dir: Path, episodes: Iterable[Any]
) -> Mapping[str, Any]:
    """Hash every distinct scene file referenced by a selected episode set."""

    root = Path(scenes_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("configured scene root is not a directory")
    resolved_by_id: Dict[str, Path] = {}
    for episode in episodes:
        scene_id = (
            episode.get("scene_id", "")
            if isinstance(episode, Mapping)
            else getattr(episode, "scene_id", "")
        )
        logical_id = SceneResolver.logical_scene_id(
            scene_id
        )
        if not logical_id:
            raise ValueError("selected episode has an empty logical scene id")
        resolved = _scene_file(root, episode, logical_id)
        previous = resolved_by_id.setdefault(logical_id, resolved)
        if previous != resolved:
            raise ValueError(
                f"logical scene id resolves to multiple artifacts: {logical_id}"
            )

    entries = []
    for logical_id, path in sorted(resolved_by_id.items()):
        entries.append(
            {
                "scene_id": logical_id,
                "artifact_id": path.name,
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
        )
    return {
        "algorithm": "sha256-logical-scene-size-content-v1",
        "digest": payload_digest({"scenes": entries}),
        "scene_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "scenes": entries,
    }


__all__ = ["referenced_scene_identity"]

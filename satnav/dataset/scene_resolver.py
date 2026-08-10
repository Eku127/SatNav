"""Resolve stable scene identifiers to machine-local scene paths."""

import os
from typing import Any, Mapping, Optional, Tuple, Union


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


class SceneResolver:
    """Keep logical scene identity separate from local filesystem layout.

    ``scene_id`` values exposed by the dataset are normalized logical names.
    Filesystem paths are resolved only at the simulator boundary and must not be
    written to benchmark results.
    """

    def __init__(self, scenes_dir: Optional[Union[str, os.PathLike]] = None):
        self.scenes_dir = str(scenes_dir) if scenes_dir is not None else None

    @staticmethod
    def logical_scene_id(scene_reference: Union[str, os.PathLike]) -> str:
        """Return a stable logical identifier for a path or dataset reference."""
        reference = str(scene_reference or "").strip().replace("\\", "/")
        if reference.startswith(DEFAULT_SCENE_PATH_PREFIX):
            reference = reference[len(DEFAULT_SCENE_PATH_PREFIX) :]
        reference = reference.rstrip("/")
        if not reference:
            return ""
        name = reference.rsplit("/", 1)[-1]
        lowered = name.lower()
        if lowered.endswith(".tiff"):
            return name[:-5]
        if lowered.endswith(".tif"):
            return name[:-4]
        return name

    def resolve(self, scene_reference: Union[str, os.PathLike]) -> str:
        """Resolve a logical ID or legacy path to a local scene path.

        Absolute paths and explicit relative paths remain usable for backwards
        compatibility.  Bare logical IDs are rooted under ``scenes_dir`` when
        configured.  SatSim itself remains responsible for adding ``.tif``.
        """
        raw_reference = str(scene_reference or "").strip()
        normalized = raw_reference.replace("\\", "/")
        had_dataset_prefix = normalized.startswith(DEFAULT_SCENE_PATH_PREFIX)
        if had_dataset_prefix:
            normalized = normalized[len(DEFAULT_SCENE_PATH_PREFIX) :]

        if os.path.isabs(raw_reference):
            return raw_reference

        has_explicit_directory = "/" in normalized
        if has_explicit_directory and not had_dataset_prefix:
            return normalized

        if self.scenes_dir:
            return os.path.join(self.scenes_dir, normalized)
        return normalized


def logical_scene_id(value: Any) -> str:
    """Return one non-empty logical scene name for cross-artifact joins."""

    scene = SceneResolver.logical_scene_id(str(value or ""))
    if not scene:
        raise ValueError("episode metadata has an empty logical scene id")
    return scene


def normalized_instruction_text(value: Any) -> str:
    """Normalize instruction text exactly as trajectory-training joins do."""

    if isinstance(value, Mapping):
        value = value.get("instruction_text", value.get("text", ""))
    instruction = " ".join(str(value or "").split())
    if not instruction:
        raise ValueError("episode metadata has an empty instruction")
    return instruction


def trajectory_training_key(episode: Mapping[str, Any]) -> Tuple[str, str, str]:
    """Build the exact scene/trajectory/instruction key used by VLM training."""

    trajectory_id = str(episode.get("trajectory_id", ""))
    if not trajectory_id:
        raise ValueError("episode metadata has an empty trajectory id")
    return (
        logical_scene_id(episode.get("scene_id")),
        trajectory_id,
        normalized_instruction_text(episode.get("instruction")).lower(),
    )


__all__ = [
    "DEFAULT_SCENE_PATH_PREFIX",
    "SceneResolver",
    "logical_scene_id",
    "normalized_instruction_text",
    "trajectory_training_key",
]

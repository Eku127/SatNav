"""Shared schema and artifact helpers for trajectory generation."""

import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from omegaconf import OmegaConf
from PIL import Image

from satnav.evaluation.artifacts import referenced_scene_identity
from satnav.task.actions import INITIAL_ACTION_INDEX, encode_action


PUBLIC_ANNOTATION_FIELDS = (
    "id",
    "trajectory_id",
    "steps",
    "video",
    "instructions",
    "actions",
)
PUBLIC_ANNOTATION_FIELD_SET = frozenset(PUBLIC_ANNOTATION_FIELDS)
SOURCE_DIGEST_ALGORITHM = "sha256-canonical-source-episode-v1"
GENERATION_DIGEST_ALGORITHM = "sha256-trajectory-generation-contract-v1"


def _reject_duplicate_object_keys(items: Sequence[Any]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("JSON object contains a duplicate key")
        value[key] = item
    return value


def strict_json_load(handle: Any) -> Any:
    """Load JSON while rejecting duplicate object keys at every depth."""

    return json.load(handle, object_pairs_hook=_reject_duplicate_object_keys)


def strict_json_loads(payload: str) -> Any:
    """Parse JSON while rejecting duplicate object keys at every depth."""

    return json.loads(payload, object_pairs_hook=_reject_duplicate_object_keys)


def _plain_value(value: Any) -> Any:
    """Convert episode/config values to deterministic JSON-compatible values."""

    if OmegaConf.is_config(value):
        return _plain_value(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _plain_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return _plain_value(vars(value))
    if hasattr(value, "item") and callable(value.item):
        return _plain_value(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_source_episode(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Mirror the lossless dataset loader's defaults for raw or loaded rows."""

    value = dict(_plain_value(source))
    for runtime_field in ("scene_path", "episode_key", "split"):
        value.pop(runtime_field, None)
    known = {
        "episode_id",
        "trajectory_id",
        "trajectory_type",
        "trajectory_subtype",
        "scene_id",
        "start_position",
        "start_rotation",
        "goals",
        "instruction",
        "waypoints",
        "reference_path",
        "aux_info",
    }
    canonical = {key: item for key, item in value.items() if key not in known}
    instruction = value.get("instruction", {})
    if isinstance(instruction, Mapping):
        instruction = dict(instruction)
        instruction["instruction_text"] = str(instruction.get("instruction_text", ""))
        if instruction.get("instruction_type") is None:
            instruction.pop("instruction_type", None)
        if instruction.get("difficulty_level") is None:
            instruction.pop("difficulty_level", None)
    else:
        instruction = {"instruction_text": str(instruction or "")}
    goals = []
    for goal in value.get("goals") or []:
        if isinstance(goal, Mapping):
            normalized_goal = dict(goal)
            normalized_goal["position"] = normalized_goal.get("position", [])
        else:
            normalized_goal = {"position": goal}
        goals.append(normalized_goal)
    canonical.update(
        {
            "episode_id": str(value.get("episode_id", "")),
            "trajectory_id": str(value.get("trajectory_id", "")),
            "trajectory_type": value.get("trajectory_type"),
            "scene_id": normalize_scene_id(value.get("scene_id", "")),
            "start_position": value.get("start_position", [0.0, 0.0, 0.0]),
            "start_rotation": value.get("start_rotation", 0.0),
            "goals": goals,
            "instruction": instruction,
            "waypoints": value.get("waypoints") or [],
            "reference_path": value.get("reference_path") or [],
            "aux_info": value.get("aux_info") or {},
            "trajectory_subtype": value.get("trajectory_subtype"),
        }
    )
    return canonical


def episode_source_digest(episode_idx: int, episode: Any) -> str:
    """Bind a cached trajectory to one complete, path-free source episode."""

    if isinstance(episode, Mapping):
        source = dict(episode)
    elif hasattr(episode, "to_dict"):
        source = dict(episode.to_dict())
    elif hasattr(episode, "__dict__"):
        source = dict(vars(episode))
    else:
        raise TypeError("source episode must be a mapping or serializable object")
    return _canonical_digest(
        {
            "algorithm": SOURCE_DIGEST_ALGORITHM,
            "episode_index": int(episode_idx),
            "episode": _canonical_source_episode(source),
        }
    )


def _select_config(config: Any, dotted_key: str, default: Any = None) -> Any:
    current = config
    for name in dotted_key.split("."):
        if isinstance(current, Mapping):
            if name not in current:
                return default
            current = current[name]
        else:
            if not hasattr(current, name):
                return default
            current = getattr(current, name)
    return _plain_value(current)


def generation_contract_digest(
    config: Any,
    *,
    scene_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the behavior-affecting contract shared by both generators."""

    contract = {
        "algorithm": GENERATION_DIGEST_ALGORITHM,
        "generator": "reference-path-follower-v1",
        "environment": _select_config(config, "ENVIRONMENT", {}),
        "simulator": _select_config(config, "SIMULATOR", {}),
        "task": _select_config(config, "TASK", {}),
        "scene_artifact": (None if scene_identity is None else dict(scene_identity)),
        "action_encoding": {
            "initial": INITIAL_ACTION_INDEX,
            "stop": encode_action("STOP"),
            "move_forward": encode_action("MOVE_FORWARD"),
            "turn_left": encode_action("TURN_LEFT"),
            "turn_right": encode_action("TURN_RIGHT"),
        },
    }
    return _canonical_digest(contract)


def episode_generation_digest(
    config: Any,
    episode: Any,
    *,
    scene_cache: Optional[MutableMapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Bind generation behavior to the referenced GeoTIFF's exact bytes."""

    scenes_dir = _select_config(config, "DATASET.SCENES_DIR")
    if scenes_dir is None:
        raise ValueError("DATASET.SCENES_DIR is required for trajectory generation")
    scene_id = (
        episode.get("scene_id", "")
        if isinstance(episode, Mapping)
        else getattr(episode, "scene_id", "")
    )
    logical_id = normalize_scene_id(scene_id)
    scene_identity = None if scene_cache is None else scene_cache.get(logical_id)
    if scene_identity is None:
        scene_identity = referenced_scene_identity(Path(str(scenes_dir)), [episode])
        if scene_cache is not None:
            scene_cache[logical_id] = scene_identity
    return generation_contract_digest(config, scene_identity=scene_identity)


def normalize_scene_id(scene_id: str) -> str:
    """Return a stable logical scene name without a machine-local path."""
    normalized = str(scene_id).replace("\\", "/").rstrip("/")
    scene_name = normalized.rsplit("/", 1)[-1]
    if scene_name.lower().endswith(".tif"):
        scene_name = scene_name[:-4]
    if not scene_name or scene_name in {".", ".."}:
        raise ValueError(f"Invalid scene_id: {scene_id!r}")
    return scene_name


def format_episode_dirname(scene_id: str, dataset_name: str, episode_idx: int) -> str:
    """Format episode directory name.

    Args:
        scene_id: Scene ID (e.g., "MN2").
        dataset_name: Dataset name (e.g., "satnav").
        episode_idx: Episode index (integer).

    Returns:
        Directory name in format: "{scene_id}_{dataset_name}_{episode_idx:06d}"

    Example:
        >>> format_episode_dirname("MN2", "satnav", 5)
        'MN2_satnav_000005'
    """
    scene_name = normalize_scene_id(scene_id)
    return f"{scene_name}_{dataset_name}_{episode_idx:06d}"


def build_annotation(
    *,
    episode_idx: int,
    trajectory_id: Any,
    episode_dirname: str,
    instruction_text: Any,
    actions: Sequence[int],
) -> Dict[str, Any]:
    """Build the stable public trajectory annotation schema."""
    instructions = (
        [instruction_text] if isinstance(instruction_text, str) else instruction_text
    )
    action_list = [encode_action(action) for action in actions]
    return {
        "id": episode_idx,
        "trajectory_id": trajectory_id,
        "steps": len(action_list) - 1,
        "video": os.path.join("images", episode_dirname),
        "instructions": instructions,
        "actions": action_list,
    }


def is_strict_public_annotation(
    annotation: Mapping[str, Any], *, exact_fields: bool = False
) -> bool:
    """Check the exact JSON scalar/container contract shared by all consumers."""

    if not isinstance(annotation, Mapping):
        return False
    if exact_fields and set(annotation) != PUBLIC_ANNOTATION_FIELD_SET:
        return False
    if not PUBLIC_ANNOTATION_FIELD_SET.issubset(annotation):
        return False
    episode_id = annotation["id"]
    steps = annotation["steps"]
    if (
        isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or episode_id < 0
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 0
    ):
        return False
    trajectory_id = annotation["trajectory_id"]
    if (
        isinstance(trajectory_id, bool)
        or not isinstance(trajectory_id, (int, str))
        or not str(trajectory_id)
        or any(
            character in str(trajectory_id)
            for character in ("/", "\\", "\x00", "\r", "\n", "\t")
        )
    ):
        return False
    if not isinstance(annotation["video"], str) or not annotation["video"]:
        return False
    instructions = annotation["instructions"]
    if (
        not isinstance(instructions, list)
        or not instructions
        or any(
            not isinstance(instruction, str) or not instruction.split()
            for instruction in instructions
        )
    ):
        return False
    actions = annotation["actions"]
    if (
        not isinstance(actions, list)
        or len(actions) != steps + 1
        or any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or action not in (-1, 0, 1, 2, 3)
            for action in actions
        )
        or not actions
        or actions[0] != INITIAL_ACTION_INDEX
        or actions[-1] != encode_action("STOP")
        or any(action == INITIAL_ACTION_INDEX for action in actions[1:])
    ):
        return False
    return True


def public_annotation(annotation: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop resume-only metadata while retaining the public field order."""
    return {field: annotation[field] for field in PUBLIC_ANNOTATION_FIELDS}


def validate_jpeg_rgb(payload: bytes) -> None:
    """Fully decode one JPEG through the same RGB path used by consumers."""

    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "JPEG" or image.width < 1 or image.height < 1:
            raise ValueError("trajectory frame is not a non-empty JPEG")
        rgb = image.convert("RGB")
        rgb.load()


def build_summary_entry(
    annotation: Mapping[str, Any],
    *,
    scene_id: Optional[str] = None,
    episode_id: Any = None,
) -> Dict[str, Any]:
    """Build a summary record without exposing an absolute scene path."""
    entry = public_annotation(annotation)
    if scene_id is not None:
        entry["scene_id"] = normalize_scene_id(scene_id)
    if episode_id is not None:
        entry["episode_id"] = episode_id
    if "_source_digest" in annotation:
        entry["source_digest"] = annotation["_source_digest"]
    if "_generation_digest" in annotation:
        entry["generation_digest"] = annotation["_generation_digest"]
    return entry


def _resolve_relative_artifact(root: Path, relative_path: str) -> Optional[Path]:
    """Resolve an output artifact path only when it stays below ``root``."""
    if not isinstance(relative_path, str) or not relative_path:
        return None

    portable_path = Path(relative_path.replace("\\", "/"))
    if portable_path.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        return None

    try:
        root = root.resolve(strict=True)
        candidate = (root / portable_path).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _resolve_existing_artifact(root: Path, candidate: Path) -> Optional[Path]:
    """Resolve an existing nested artifact only when its target stays in root."""

    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def annotation_artifacts_complete(
    output_path: str,
    annotation: Mapping[str, Any],
    *,
    expected_source_digest: Optional[str] = None,
    expected_generation_digest: Optional[str] = None,
    expected_video: Optional[str] = None,
    expected_scene_id: Optional[str] = None,
    expected_episode_id: Any = None,
) -> bool:
    """Validate an annotation and its contiguous on-disk RGB frame sequence."""
    if not is_strict_public_annotation(annotation):
        return False
    steps = annotation["steps"]
    video_rel = annotation["video"]
    video_parts = Path(video_rel.replace("\\", "/")).parts
    if len(video_parts) != 2 or video_parts[0] != "images":
        return False
    scene_name, separator, _ = video_parts[1].rpartition("_satnav_")
    if not separator:
        return False
    try:
        canonical_video = os.path.join(
            "images",
            format_episode_dirname(scene_name, "satnav", annotation["id"]),
        )
    except (TypeError, ValueError):
        return False
    if video_rel.replace("\\", "/") != canonical_video.replace("\\", "/"):
        return False
    if expected_video is not None and video_rel.replace(
        "\\", "/"
    ) != expected_video.replace("\\", "/"):
        return False
    source_digest = annotation.get("_source_digest", annotation.get("source_digest"))
    generation_digest = annotation.get(
        "_generation_digest", annotation.get("generation_digest")
    )
    if expected_source_digest is not None and source_digest != expected_source_digest:
        return False
    if (
        expected_generation_digest is not None
        and generation_digest != expected_generation_digest
    ):
        return False
    recorded_scene_id = annotation.get("_scene_id", annotation.get("scene_id"))
    if expected_scene_id is not None:
        try:
            if not isinstance(recorded_scene_id, str) or normalize_scene_id(
                recorded_scene_id
            ) != normalize_scene_id(expected_scene_id):
                return False
        except (TypeError, ValueError):
            return False
    recorded_episode_id = annotation.get("_episode_id", annotation.get("episode_id"))
    if expected_episode_id is not None and (
        isinstance(recorded_episode_id, bool)
        or not isinstance(recorded_episode_id, (int, str))
        or type(recorded_episode_id) is not type(expected_episode_id)
        or recorded_episode_id != expected_episode_id
    ):
        return False
    output_root = Path(output_path)
    video_path = _resolve_relative_artifact(output_root, video_rel)
    if video_path is None:
        return False
    rgb_dir = _resolve_existing_artifact(output_root, video_path / "rgb")
    if rgb_dir is None or not rgb_dir.is_dir():
        return False

    jpgs = []
    for path in rgb_dir.iterdir():
        if path.suffix.lower() != ".jpg":
            continue
        resolved = _resolve_existing_artifact(output_root, path)
        if resolved is None or not resolved.is_file():
            return False
        try:
            payload = resolved.read_bytes()
            validate_jpeg_rgb(payload)
        except (OSError, SyntaxError, ValueError):
            return False
        jpgs.append(path.name)
    jpgs.sort()
    if len(jpgs) != steps + 1:
        return False
    return all(
        filename == f"{idx:03d}.jpg" for idx, filename in enumerate(jpgs, start=1)
    )

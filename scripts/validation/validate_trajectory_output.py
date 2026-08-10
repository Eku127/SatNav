#!/usr/bin/env python3
"""Validate a complete SatNav trajectory-generation output tree.

The checker is intentionally standalone and dependency-light.  It verifies
episode coverage, annotation/action/frame alignment, path safety, summary
identity, and (when requested) decodes every JPEG while computing an ordered
content digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from applications.trajectory_generation.utils import (
    episode_generation_digest,
    episode_source_digest,
    format_episode_dirname,
    is_strict_public_annotation,
    normalize_scene_id,
    public_annotation,
    validate_jpeg_rgb,
)
from satnav.dataset.scene_resolver import trajectory_training_key


REQUIRED_FIELDS = {
    "id",
    "trajectory_id",
    "steps",
    "video",
    "instructions",
    "actions",
}
RESUME_CACHE_FIELDS = REQUIRED_FIELDS | {
    "_scene_id",
    "_episode_id",
    "_source_digest",
    "_generation_digest",
}
SUMMARY_FIELDS = REQUIRED_FIELDS | {
    "scene_id",
    "episode_id",
    "source_digest",
    "generation_digest",
}
MAX_RESUME_CACHE_BYTES = 1024 * 1024
DECODE_BATCH_SIZE = 4096


def _reject_duplicate_object_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object contains a duplicate key")
        value[key] = item
    return value


def _strict_json_load(handle: Any) -> Any:
    return json.load(handle, object_pairs_hook=_reject_duplicate_object_keys)


def _strict_json_loads(payload: str) -> Any:
    return json.loads(payload, object_pairs_hook=_reject_duplicate_object_keys)


def _load_annotations(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = _strict_json_load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON annotation list")
    return value


def _load_summary(path: Optional[Path]) -> Dict[Any, Any]:
    if path is None:
        return {}
    rows: Dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _strict_json_loads(line)
            index = value.get("id") if isinstance(value, Mapping) else None
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                index = f"invalid-summary-row-{line_number}"
            if index in rows:
                raise ValueError(
                    f"duplicate summary id {index} at {path}:{line_number}"
                )
            rows[index] = value
    return rows


def _load_source_episodes(
    path: Optional[Path],
) -> Optional[List[Mapping[str, Any]]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = _strict_json_load(handle)
    episodes = value.get("episodes") if isinstance(value, dict) else value
    if not isinstance(episodes, list):
        raise ValueError(f"{path} does not contain an episode list")
    if not all(isinstance(episode, Mapping) for episode in episodes):
        raise ValueError(f"{path} contains a non-object episode")
    return episodes


def _file_identity(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if path is None:
        return None
    resolved = path.resolve()
    payload = resolved.read_bytes()
    return {
        "id": resolved.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _source_instruction(episode: Mapping[str, Any]) -> List[str]:
    instruction = episode.get("instruction", {})
    if isinstance(instruction, Mapping):
        instruction = instruction.get("instruction_text", "")
    return [str(instruction or "")]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_safe_identifier(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return False
    rendered = str(value)
    return bool(rendered) and not any(
        character in rendered for character in ("/", "\\", "\x00", "\r", "\n", "\t")
    )


def _safe_video_dir(root: Path, value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(value).is_absolute():
        return None
    try:
        root = root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _safe_existing_path(root: Path, candidate: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _decode_and_hash(item: Tuple[str, Path]) -> Tuple[str, str, Optional[str]]:
    relative, path = item
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            payload = handle.read()
        digest.update(payload)
        validate_jpeg_rgb(payload)
        return relative, digest.hexdigest(), None
    except Exception as error:  # the report needs every corrupt file, not a crash
        return relative, digest.hexdigest(), type(error).__name__


def _batches(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Bound executor submissions so production validation stays memory-safe."""

    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _inventory_images_tree(
    output_root: Path,
    expected_artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, int]:
    """Inventory the complete images tree without following any symlinks.

    Episode directories not referenced by the public annotations are pruned
    from the walk immediately.  Their contents are deliberately not read, but
    their presence still fails validation.
    """

    counts = {
        "root_errors": 0,
        "walk_errors": 0,
        "present_episode_directories": 0,
        "missing_episode_directories": 0,
        "missing_rgb_directories": 0,
        "unreferenced_episode_directories": 0,
        "unexpected_entries": 0,
        "symlinks": 0,
        "jpeg_files": 0,
        "resume_sidecar_pairs": 0,
        "resume_sidecar_errors": 0,
    }
    expected_frames = {
        name: int(artifact["frames"]) for name, artifact in expected_artifacts.items()
    }
    images_root = output_root / "images"
    if not images_root.exists():
        if expected_frames:
            counts["root_errors"] = 1
        return counts
    if images_root.is_symlink() or not images_root.is_dir():
        counts["root_errors"] = 1
        return counts

    seen_episode_directories = set()
    seen_rgb_directories = set()
    seen_sidecars: Dict[str, set] = {name: set() for name in expected_artifacts}

    def record_walk_error(_error: OSError) -> None:
        counts["walk_errors"] += 1

    for directory, directory_names, filenames in os.walk(
        images_root,
        topdown=True,
        followlinks=False,
        onerror=record_walk_error,
    ):
        current = Path(directory)
        relative_parts = current.relative_to(images_root).parts
        traversable_directories = []
        for name in sorted(directory_names):
            candidate = current / name
            if candidate.is_symlink():
                counts["symlinks"] += 1
                counts["unexpected_entries"] += 1
                continue
            if len(relative_parts) == 0:
                if name not in expected_frames:
                    counts["unreferenced_episode_directories"] += 1
                    continue
                seen_episode_directories.add(name)
                traversable_directories.append(name)
            elif len(relative_parts) == 1:
                if relative_parts[0] in expected_frames and name == "rgb":
                    seen_rgb_directories.add(relative_parts[0])
                    traversable_directories.append(name)
                else:
                    counts["unexpected_entries"] += 1
            else:
                counts["unexpected_entries"] += 1
        directory_names[:] = traversable_directories

        for name in filenames:
            candidate = current / name
            if candidate.is_symlink():
                counts["symlinks"] += 1
                counts["unexpected_entries"] += 1
                continue
            if not candidate.is_file():
                counts["unexpected_entries"] += 1
                continue
            if len(relative_parts) == 1 and name in {".annotation.json", ".done"}:
                episode_name = relative_parts[0]
                seen_sidecars[episode_name].add(name)
                if name == ".done":
                    try:
                        if candidate.stat().st_size != 0:
                            counts["resume_sidecar_errors"] += 1
                    except OSError:
                        counts["resume_sidecar_errors"] += 1
                    continue

                expected = expected_artifacts[episode_name]
                try:
                    if candidate.stat().st_size > MAX_RESUME_CACHE_BYTES:
                        raise ValueError("resume cache is too large")
                    cache = _strict_json_loads(candidate.read_text(encoding="utf-8"))
                    if not isinstance(cache, Mapping):
                        raise ValueError("resume cache is not an object")
                    if set(cache) != RESUME_CACHE_FIELDS:
                        raise ValueError("resume cache fields differ")
                    if not is_strict_public_annotation(
                        public_annotation(cache), exact_fields=True
                    ):
                        raise ValueError("resume cache public schema is invalid")
                    annotation = expected["annotation"]
                    if any(
                        cache.get(field) != annotation[field]
                        for field in REQUIRED_FIELDS
                    ):
                        raise ValueError("resume cache public fields differ")
                    scene_id = cache["_scene_id"]
                    if (
                        not isinstance(scene_id, str)
                        or normalize_scene_id(scene_id) != scene_id
                    ):
                        raise ValueError("resume cache scene id is not logical")
                    episode_id = cache["_episode_id"]
                    if (
                        isinstance(episode_id, bool)
                        or not isinstance(episode_id, (int, str))
                        or not str(episode_id)
                        or any(
                            separator in str(episode_id) for separator in ("/", "\\")
                        )
                    ):
                        raise ValueError("resume cache episode id is unsafe")
                    for field in ("_source_digest", "_generation_digest"):
                        if not _is_sha256(cache[field]):
                            raise ValueError("resume cache digest is invalid")
                    summary = expected.get("summary")
                    if summary is not None and (
                        cache["_scene_id"] != summary.get("scene_id")
                        or str(cache["_episode_id"]) != str(summary.get("episode_id"))
                        or cache["_source_digest"] != summary.get("source_digest")
                        or cache["_generation_digest"]
                        != summary.get("generation_digest")
                    ):
                        raise ValueError("resume cache identity differs from summary")
                except (KeyError, OSError, TypeError, UnicodeError, ValueError):
                    counts["resume_sidecar_errors"] += 1
                continue
            if len(relative_parts) == 2 and relative_parts[1] == "rgb":
                episode_name = relative_parts[0]
                frame_stem, extension = os.path.splitext(name)
                try:
                    frame_index = int(frame_stem)
                except ValueError:
                    frame_index = -1
                if (
                    extension.lower() == ".jpg"
                    and frame_index >= 1
                    and name == f"{frame_index:03d}.jpg"
                    and frame_index <= expected_frames.get(episode_name, 0)
                ):
                    counts["jpeg_files"] += 1
                    continue
            counts["unexpected_entries"] += 1

    counts["present_episode_directories"] = len(seen_episode_directories)
    counts["missing_episode_directories"] = len(
        set(expected_frames).difference(seen_episode_directories)
    )
    counts["missing_rgb_directories"] = len(
        set(expected_frames).difference(seen_rgb_directories)
    )
    for sidecars in seen_sidecars.values():
        if not sidecars:
            continue
        if sidecars == {".annotation.json", ".done"}:
            counts["resume_sidecar_pairs"] += 1
        else:
            counts["resume_sidecar_errors"] += 1
    return counts


def _inventory_output_root(
    output_root: Path,
    allowed_files: Iterable[Optional[Path]],
) -> Mapping[str, int]:
    """Reject undeclared top-level output entries without exposing names."""

    counts = {"root_errors": 0, "unexpected_entries": 0, "symlinks": 0}
    if output_root.is_symlink() or not output_root.is_dir():
        counts["root_errors"] = 1
        return counts
    allowed_names = set()
    for path in allowed_files:
        if path is None:
            continue
        candidate = Path(path).absolute()
        if candidate.parent == output_root:
            allowed_names.add(candidate.name)
    for candidate in output_root.iterdir():
        if candidate.is_symlink():
            counts["symlinks"] += 1
            counts["unexpected_entries"] += 1
        elif candidate.name == "images" and candidate.is_dir():
            continue
        elif candidate.name in allowed_names and candidate.is_file():
            continue
        else:
            counts["unexpected_entries"] += 1
    return counts


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _has_symlink_component(path: Path) -> bool:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validated_output_root(path: Path) -> Path:
    lexical = _lexical_absolute(path)
    if _has_symlink_component(lexical):
        raise ValueError("trajectory output root must not contain symlinks")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("trajectory output root is not a directory")
    return resolved


def _validated_report_path(args: argparse.Namespace) -> Path:
    """Resolve a report destination that cannot mutate any validated input."""

    raw_report = Path(args.report).expanduser()
    if not raw_report.name or raw_report.name in {".", ".."}:
        raise ValueError("validation report destination is invalid")
    absolute_report = (
        raw_report if raw_report.is_absolute() else Path.cwd() / raw_report
    )
    report = absolute_report.parent.resolve(strict=False) / absolute_report.name
    if report.is_symlink() or (report.exists() and not report.is_file()):
        raise ValueError("validation report destination is unsafe")

    protected_directories = [_validated_output_root(Path(args.output_root))]
    scenes_dir = getattr(args, "scenes_dir", None)
    if scenes_dir is not None:
        protected_directories.append(Path(scenes_dir).resolve(strict=True))
    if any(_is_within(report, directory) for directory in protected_directories):
        raise ValueError("validation report must be outside protected inputs")

    protected_files = (
        getattr(args, "annotations", None),
        getattr(args, "summary", None),
        getattr(args, "source_episodes", None),
        getattr(args, "generation_config", None),
    )
    for protected_file in protected_files:
        if protected_file is not None and report == Path(protected_file).resolve(
            strict=False
        ):
            raise ValueError("validation report conflicts with an input file")
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate(args: argparse.Namespace) -> Mapping[str, Any]:
    output_root = _validated_output_root(Path(args.output_root))
    if _has_symlink_component(Path(args.annotations)):
        raise ValueError("trajectory annotations path must not contain symlinks")
    annotations_path = _lexical_absolute(Path(args.annotations))
    if annotations_path != output_root / "annotations.json":
        raise ValueError("annotations must be output_root/annotations.json")
    annotations = _load_annotations(annotations_path)
    summary_path = args.summary
    if summary_path is None:
        discovered = output_root / "summary.json"
        if discovered.is_file():
            summary_path = discovered
    if summary_path is not None:
        if _has_symlink_component(Path(summary_path)):
            raise ValueError("trajectory summary path must not contain symlinks")
        summary_path = _lexical_absolute(Path(summary_path))
        if summary_path != output_root / "summary.json":
            raise ValueError("summary must be output_root/summary.json")
    summaries = _load_summary(summary_path)
    expected = args.expected_count
    source_episodes = _load_source_episodes(args.source_episodes)
    source_count = None if source_episodes is None else len(source_episodes)
    if expected is None:
        expected = source_count

    errors: List[str] = []
    if expected is not None and int(expected) <= 0:
        errors.append("expected episode count must be positive")
    if source_count is not None and source_count <= 0:
        errors.append("source episode list must be non-empty")
    generation_config_path = getattr(args, "generation_config", None)
    scenes_dir = getattr(args, "scenes_dir", None)
    generation_config = None
    scene_cache: Dict[str, Mapping[str, Any]] = {}
    if (generation_config_path is None) != (scenes_dir is None):
        errors.append("generation validation requires both config and scene directory")
    elif generation_config_path is not None:
        if source_episodes is None:
            errors.append("generation validation requires source episodes")
        else:
            from omegaconf import OmegaConf

            generation_config = OmegaConf.load(generation_config_path)
            OmegaConf.set_struct(generation_config, False)
            generation_config.DATASET.SCENES_DIR = str(scenes_dir.resolve())
            OmegaConf.set_struct(generation_config, True)
    if source_episodes is not None and not summaries:
        errors.append("source episode validation requires a non-empty summary")
    if (
        source_count is not None
        and args.expected_count is not None
        and int(args.expected_count) != source_count
    ):
        errors.append("expected count does not equal source episode count")
    ids = set()
    videos = set()
    jpeg_items: List[Tuple[str, Path]] = []
    jpeg_file_count = 0
    total_steps = 0
    stop_terminated = 0
    source_rows_matched = 0
    expected_artifacts: Dict[str, Mapping[str, Any]] = {}
    source_training_keys = set()
    if source_episodes is not None:
        for source_index, source_episode in enumerate(source_episodes):
            try:
                training_key = trajectory_training_key(source_episode)
            except (TypeError, ValueError):
                errors.append(
                    f"source episode {source_index} has incomplete training identity"
                )
                continue
            trajectory_type = str(source_episode.get("trajectory_type", "")).strip()
            if not trajectory_type:
                errors.append(f"source episode {source_index} lacks trajectory_type")
            if training_key in source_training_keys:
                errors.append("source episode training identity is not unique")
            source_training_keys.add(training_key)

    for row_number, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            errors.append(f"annotation[{row_number}] is not an object")
            continue
        missing = REQUIRED_FIELDS.difference(annotation)
        if missing:
            errors.append(f"annotation[{row_number}] missing fields {sorted(missing)}")
            continue
        if set(annotation) != REQUIRED_FIELDS:
            errors.append(f"annotation[{row_number}] contains non-public fields")
        if not is_strict_public_annotation(annotation, exact_fields=True):
            errors.append(f"annotation[{row_number}] violates strict public schema")
        if (
            isinstance(annotation["id"], bool)
            or not isinstance(annotation["id"], int)
            or isinstance(annotation["steps"], bool)
            or not isinstance(annotation["steps"], int)
        ):
            errors.append(f"annotation[{row_number}] has invalid id/steps")
            continue
        episode_id = annotation["id"]
        steps = annotation["steps"]
        if episode_id < 0:
            errors.append(f"annotation {episode_id} has a negative id")
            continue
        if episode_id in ids:
            errors.append(f"duplicate annotation id {episode_id}")
        ids.add(episode_id)
        if steps < 0:
            errors.append(f"annotation {episode_id} has negative steps")
            continue
        total_steps += steps

        source_episode = None
        if not _is_safe_identifier(annotation["trajectory_id"]):
            errors.append(f"annotation {episode_id} has an unsafe trajectory_id")
        if source_episodes is not None and 0 <= episode_id < len(source_episodes):
            source_episode = source_episodes[episode_id]
            source_mismatch = False
            if str(annotation["trajectory_id"]) != str(
                source_episode.get("trajectory_id", "")
            ):
                errors.append(
                    f"annotation {episode_id} trajectory_id differs from source"
                )
                source_mismatch = True
            if annotation["instructions"] != _source_instruction(source_episode):
                errors.append(
                    f"annotation {episode_id} instructions differ from source"
                )
                source_mismatch = True
            if not source_mismatch:
                source_rows_matched += 1
        elif source_episodes is not None:
            errors.append(f"annotation {episode_id} has no source episode")

        actions = annotation["actions"]
        if not isinstance(annotation["instructions"], list) or not all(
            isinstance(instruction, str) for instruction in annotation["instructions"]
        ):
            errors.append(f"annotation {episode_id} has invalid instructions")
        if not isinstance(actions, list) or len(actions) != steps + 1:
            errors.append(
                f"annotation {episode_id} actions length does not equal steps+1"
            )
        elif not actions or actions[0] != -1:
            errors.append(f"annotation {episode_id} lacks initial -1 sentinel")
        else:
            if any(
                isinstance(action, bool)
                or not isinstance(action, int)
                or action not in (-1, 0, 1, 2, 3)
                for action in actions
            ):
                errors.append(f"annotation {episode_id} has an invalid action")
            if any(action == -1 for action in actions[1:]):
                errors.append(f"annotation {episode_id} repeats initial sentinel")
            if actions[-1] == 0:
                stop_terminated += 1
            else:
                errors.append(f"annotation {episode_id} does not end in STOP")

        video = annotation["video"]
        if not isinstance(video, str):
            errors.append(f"annotation {episode_id} has an invalid video path")
            continue
        video_parts = Path(video.replace("\\", "/")).parts
        if len(video_parts) != 2 or video_parts[0] != "images":
            errors.append(
                f"annotation {episode_id} video path does not name one images directory"
            )
        else:
            video_name = video_parts[1]
            try:
                if source_episode is not None:
                    expected_video_name = format_episode_dirname(
                        str(source_episode.get("scene_id", "")),
                        "satnav",
                        episode_id,
                    )
                else:
                    scene_name, separator, _ = video_name.rpartition("_satnav_")
                    if not separator:
                        raise ValueError("missing canonical separator")
                    expected_video_name = format_episode_dirname(
                        scene_name, "satnav", episode_id
                    )
            except (TypeError, ValueError):
                expected_video_name = None
            if video_name != expected_video_name:
                errors.append(
                    f"annotation {episode_id} video path has a non-canonical identity"
                )
            expected_artifacts[video_parts[1]] = {
                "frames": steps + 1,
                "annotation": annotation,
                "summary": summaries.get(episode_id),
            }
        if video in videos:
            errors.append("duplicate video path")
        videos.add(video)
        video_dir = _safe_video_dir(output_root, video)
        if video_dir is None:
            errors.append(f"annotation {episode_id} has unsafe video path")
            continue
        rgb_dir = _safe_existing_path(output_root, video_dir / "rgb")
        if rgb_dir is None or not rgb_dir.is_dir():
            errors.append(
                f"annotation {episode_id} has an unsafe or missing RGB directory"
            )
            continue
        images = sorted(rgb_dir.glob("*.jpg"))
        jpeg_file_count += len(images)
        if len(images) != steps + 1:
            errors.append(
                f"annotation {episode_id} has {len(images)} JPEGs, expected {steps + 1}"
            )
        for frame_index, image in enumerate(images, start=1):
            if image.name != f"{frame_index:03d}.jpg":
                errors.append(
                    f"annotation {episode_id} has non-contiguous frame {image.name}"
                )
                break
            safe_image = _safe_existing_path(output_root, image)
            if safe_image is None or not safe_image.is_file():
                errors.append(
                    f"annotation {episode_id} has unsafe JPEG frame {image.name}"
                )
                continue
            relative = (
                Path(str(video).replace("\\", "/")) / "rgb" / image.name
            ).as_posix()
            jpeg_items.append((relative, safe_image))

        if summaries:
            summary = summaries.get(episode_id)
            if summary is None:
                errors.append(f"annotation {episode_id} is missing from summary")
            else:
                for field in sorted(REQUIRED_FIELDS):
                    if summary.get(field) != annotation[field]:
                        errors.append(
                            f"summary {episode_id} field {field} differs from annotation"
                        )
                scene_id = summary.get("scene_id")
                if isinstance(scene_id, str) and (
                    os.path.isabs(scene_id) or PureWindowsPath(scene_id).is_absolute()
                ):
                    errors.append(f"summary {episode_id} leaks an absolute scene_id")
                if source_episode is not None:
                    try:
                        expected_scene = normalize_scene_id(
                            str(source_episode.get("scene_id", ""))
                        )
                    except ValueError:
                        errors.append(
                            f"source episode {episode_id} has an invalid scene_id"
                        )
                        expected_scene = None
                    if scene_id != expected_scene:
                        errors.append(
                            f"summary {episode_id} scene_id differs from source"
                        )
                    if str(summary.get("episode_id", "")) != str(
                        source_episode.get("episode_id", "")
                    ):
                        errors.append(
                            f"summary {episode_id} episode_id differs from source"
                        )
                    recorded_source_digest = summary.get("source_digest")
                    if recorded_source_digest is None:
                        errors.append(f"summary {episode_id} lacks a source digest")
                    elif recorded_source_digest != episode_source_digest(
                        episode_id, source_episode
                    ):
                        errors.append(
                            f"summary {episode_id} source digest differs from source"
                        )
                    if generation_config is not None:
                        recorded_generation_digest = summary.get("generation_digest")
                        if recorded_generation_digest is None:
                            errors.append(
                                f"summary {episode_id} lacks a generation digest"
                            )
                        elif recorded_generation_digest != (
                            episode_generation_digest(
                                generation_config,
                                source_episode,
                                scene_cache=scene_cache,
                            )
                        ):
                            errors.append(
                                f"summary {episode_id} generation digest differs"
                            )

    missing_ids: List[int] = []
    extra_ids: List[int] = []
    if expected is not None:
        target = set(range(int(expected)))
        missing_ids = sorted(target.difference(ids))
        extra_ids = sorted(ids.difference(target))
        if missing_ids:
            errors.append(f"missing {len(missing_ids)} expected annotation ids")
        if extra_ids:
            errors.append(f"found {len(extra_ids)} out-of-range annotation ids")
    if summaries and set(summaries) != ids:
        errors.append("summary id set does not equal annotation id set")

    for summary_id, summary in summaries.items():
        if not isinstance(summary, Mapping) or set(summary) != SUMMARY_FIELDS:
            errors.append(f"summary {summary_id} fields differ from public schema")
            continue
        if not is_strict_public_annotation(
            public_annotation(summary), exact_fields=True
        ):
            errors.append(f"summary {summary_id} violates strict public schema")
        scene_id = summary["scene_id"]
        try:
            logical_scene_id = normalize_scene_id(scene_id)
        except (TypeError, ValueError):
            logical_scene_id = None
        if logical_scene_id != scene_id:
            errors.append(f"summary {summary_id} has an unsafe scene_id")
        if not _is_safe_identifier(summary["episode_id"]):
            errors.append(f"summary {summary_id} has an unsafe episode_id")
        if not _is_sha256(summary["source_digest"]):
            errors.append(f"summary {summary_id} has an invalid source digest")
        if not _is_sha256(summary["generation_digest"]):
            errors.append(f"summary {summary_id} has an invalid generation digest")

    image_tree = _inventory_images_tree(output_root, expected_artifacts)
    if image_tree["root_errors"] or image_tree["walk_errors"]:
        errors.append("images tree is missing, unsafe, or unreadable")
    if image_tree["missing_episode_directories"]:
        errors.append(
            "images tree is missing "
            f"{image_tree['missing_episode_directories']} referenced episode directories"
        )
    if image_tree["missing_rgb_directories"]:
        errors.append(
            "images tree is missing "
            f"{image_tree['missing_rgb_directories']} referenced RGB directories"
        )
    if image_tree["unreferenced_episode_directories"]:
        errors.append(
            "images tree contains "
            f"{image_tree['unreferenced_episode_directories']} unreferenced episode directories"
        )
    if image_tree["unexpected_entries"]:
        errors.append(
            "images tree contains "
            f"{image_tree['unexpected_entries']} unexpected or unsafe entries"
        )
    if image_tree["resume_sidecar_errors"]:
        errors.append(
            "images tree contains "
            f"{image_tree['resume_sidecar_errors']} invalid resume sidecar artifacts"
        )
    if image_tree["jpeg_files"] != jpeg_file_count:
        errors.append("complete images-tree JPEG count differs from annotations")

    output_tree = _inventory_output_root(
        output_root,
        (
            annotations_path,
            summary_path,
        ),
    )
    if output_tree["root_errors"]:
        errors.append("output root is unsafe or unreadable")
    if output_tree["unexpected_entries"]:
        errors.append(
            "output root contains "
            f"{output_tree['unexpected_entries']} undeclared or unsafe entries"
        )

    image_tree_digest = None
    decode_failures: List[str] = []
    if args.decode_images:
        tree_digest = hashlib.sha256()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for batch in _batches(jpeg_items, DECODE_BATCH_SIZE):
                for relative, image_digest, error in executor.map(
                    _decode_and_hash, batch, chunksize=32
                ):
                    encoded = relative.encode("utf-8")
                    tree_digest.update(len(encoded).to_bytes(8, "big"))
                    tree_digest.update(encoded)
                    tree_digest.update(bytes.fromhex(image_digest))
                    if error is not None:
                        decode_failures.append(f"{relative}: {error}")
        image_tree_digest = tree_digest.hexdigest()
        if decode_failures:
            errors.append(f"{len(decode_failures)} JPEG files failed decoding")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "inputs": {
            "annotations": _file_identity(annotations_path),
            "output_root": {"id": output_root.name or "output"},
            "summary": _file_identity(summary_path),
            "source_episodes": _file_identity(args.source_episodes),
            "generation_config": _file_identity(generation_config_path),
            "scenes": (
                None
                if scenes_dir is None
                else {"id": scenes_dir.resolve().name or "scenes"}
            ),
        },
        "validation_scope": {
            "annotation_schema": True,
            "complete_images_tree_inventory": True,
            "jpeg_content": bool(args.decode_images),
            "source_episode_identity": source_episodes is not None,
            "generation_contract_and_scene_content": generation_config is not None,
        },
        "counts": {
            "expected_episodes": expected,
            "source_episodes": source_count,
            "source_rows_matched": source_rows_matched,
            "annotations": len(annotations),
            "unique_ids": len(ids),
            "unique_videos": len(videos),
            "summary_rows": len(summaries),
            "steps": total_steps,
            "stop_terminated": stop_terminated,
            "jpeg_files": jpeg_file_count,
            "jpeg_decode_failures": len(decode_failures),
            "image_tree_episode_directories": image_tree["present_episode_directories"],
            "image_tree_jpeg_files": image_tree["jpeg_files"],
            "unreferenced_image_directories": image_tree[
                "unreferenced_episode_directories"
            ],
            "unexpected_image_entries": image_tree["unexpected_entries"],
            "image_tree_symlinks": image_tree["symlinks"],
            "resume_sidecar_pairs": image_tree["resume_sidecar_pairs"],
            "resume_sidecar_errors": image_tree["resume_sidecar_errors"],
            "unexpected_output_root_entries": output_tree["unexpected_entries"],
            "output_root_symlinks": output_tree["symlinks"],
        },
        "digests": {
            "annotations_sha256": hashlib.sha256(
                annotations_path.read_bytes()
            ).hexdigest(),
            "image_tree_sha256": image_tree_digest,
        },
        "missing_ids": missing_ids[:100],
        "extra_ids": extra_ids[:100],
        "decode_failures": decode_failures[:100],
        "errors": errors[:200],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--source-episodes", type=Path)
    parser.add_argument("--generation-config", type=Path)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _validate_release_cli_args(args: argparse.Namespace) -> None:
    """Require every production-validation scope at the public CLI boundary."""

    required = (
        args.source_episodes,
        args.generation_config,
        args.scenes_dir,
    )
    if any(value is None for value in required) or not args.decode_images:
        raise ValueError(
            "release validation requires source, generation config, scenes, and JPEG decode"
        )
    if args.expected_count is not None and args.expected_count <= 0:
        raise ValueError("release validation expected count must be positive")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    try:
        _validate_release_cli_args(args)
    except (TypeError, ValueError):
        print("error=invalid release validation arguments")
        return 2
    try:
        report_path = _validated_report_path(args)
    except (OSError, RuntimeError, ValueError):
        print("error=unsafe validation report destination")
        return 2
    try:
        report = validate(args)
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        print("error=trajectory validation could not be completed")
        return 2
    _atomic_json(report_path, report)
    print(json.dumps(report["counts"], sort_keys=True))
    print(f"status={report['status']} report={report_path.name}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

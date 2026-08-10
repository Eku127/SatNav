"""Validated SatNav trajectory adapter for bundled OpenFly training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional, Sequence

from baselines.vlm.openfly.actions import (
    ACTION_TO_NAME,
    COMPACT,
    ORIGINAL,
    action_to_text,
    get_normalized_original_action_vector,
    get_original_action_vector,
    resolve_action_format,
)
from baselines.vlm.openfly.artifacts import file_identity
from baselines.vlm.openfly.prompting import (
    ASSISTANT_BOUNDARY,
    DEFAULT_ACTION_HISTORY_LIMIT,
    build_openfly_prompt,
    build_openfly_prompt_with_answer,
)
from satnav.dataset.scene_resolver import (
    logical_scene_id,
    normalized_instruction_text,
    trajectory_training_key,
)


IGNORE_INDEX = -100
DEFAULT_ORIGINAL_DIM_LOSS_WEIGHTS = (0.4, 1.2, 1.2, 1.2)
DEFAULT_HEAD_KEEP = 7
DEFAULT_SAMPLE_STRIDE = 3
DEFAULT_STOP_REPEAT = 2
DEFAULT_STOP_HISTORY_AUG = 1
DEFAULT_STOP_WINDOW = 0
DEFAULT_TAIL_KEEP = 5
FRAME_DIGEST_BYTES = 32
FRAME_IDENTITY_INDEX_ALGORITHM = "sha256-frame-digest-index-v1"
FRAME_IDENTITY_INDEX_ORDER = "annotation-record-order-then-frame-index-ascending"


def resolve_original_dim_loss_weights(
    value: Optional[Sequence[float]] = None,
) -> tuple[float, float, float, float]:
    if value is None:
        configured = os.environ.get("OPENFLY_ORIGINAL_DIM_LOSS_WEIGHTS", "").strip()
        value = (
            tuple(float(item.strip()) for item in configured.split(",") if item.strip())
            if configured
            else DEFAULT_ORIGINAL_DIM_LOSS_WEIGHTS
        )
    weights = tuple(float(item) for item in value)
    if len(weights) != 4 or any(
        not math.isfinite(item) or item <= 0 for item in weights
    ):
        raise ValueError(
            "original action loss weights require four finite positive values"
        )
    return weights


@dataclass(frozen=True)
class AnnotationRecord:
    source_root: Path
    path_anchor: Path
    video_dir: Path
    logical_scene: str
    instruction: str
    actions: tuple[int, ...]
    trajectory_id: str
    trajectory_type: str
    payload: Mapping[str, Any]
    allow_external_paths: bool = False


@dataclass(frozen=True, slots=True)
class SampleIndex:
    record_index: int
    step_index: int
    history_first: int
    history_second: int


@dataclass(frozen=True)
class SamplingPolicy:
    head_keep: int = DEFAULT_HEAD_KEEP
    sample_stride: int = DEFAULT_SAMPLE_STRIDE
    stop_repeat: int = DEFAULT_STOP_REPEAT
    stop_history_aug: int = DEFAULT_STOP_HISTORY_AUG
    stop_window: int = DEFAULT_STOP_WINDOW
    tail_keep: int = DEFAULT_TAIL_KEEP
    action_history_limit: int = DEFAULT_ACTION_HISTORY_LIMIT

    def __post_init__(self) -> None:
        if self.head_keep < 0 or self.tail_keep < 0 or self.stop_window < 0:
            raise ValueError("head/tail/stop-window values must be non-negative")
        if (
            self.sample_stride <= 0
            or self.stop_repeat <= 0
            or self.stop_history_aug <= 0
        ):
            raise ValueError(
                "stride/repeat/history-augmentation values must be positive"
            )
        if self.action_history_limit < 0:
            raise ValueError("action_history_limit must be non-negative")

    def to_payload(self) -> Mapping[str, int]:
        return {
            "head_keep": self.head_keep,
            "sample_stride": self.sample_stride,
            "stop_repeat": self.stop_repeat,
            "stop_history_aug": self.stop_history_aug,
            "stop_window": self.stop_window,
            "tail_keep": self.tail_keep,
            "action_history_limit": self.action_history_limit,
        }


def sampling_policy_from_environment() -> SamplingPolicy:
    def value(name: str, default: int) -> int:
        configured = os.environ.get(name, "").strip()
        return int(configured) if configured else int(default)

    return SamplingPolicy(
        head_keep=value("SATNAV_HEAD_KEEP", DEFAULT_HEAD_KEEP),
        sample_stride=value("SATNAV_SAMPLE_STRIDE", DEFAULT_SAMPLE_STRIDE),
        stop_repeat=value("SATNAV_STOP_REPEAT", DEFAULT_STOP_REPEAT),
        stop_history_aug=value("SATNAV_STOP_HISTORY_AUG", DEFAULT_STOP_HISTORY_AUG),
        stop_window=value("SATNAV_STOP_WINDOW", DEFAULT_STOP_WINDOW),
        tail_keep=value("SATNAV_TAIL_KEEP", DEFAULT_TAIL_KEEP),
        action_history_limit=value(
            "OPENFLY_ACTION_HISTORY_LIMIT", DEFAULT_ACTION_HISTORY_LIMIT
        ),
    )


def _positive_optional(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _annotation_path(configured: Path) -> tuple[Path, Path]:
    path = Path(configured).expanduser().resolve()
    if path.is_file():
        return path, path.parent
    if path.name == "images" and (path.parent / "annotations.json").is_file():
        return path.parent / "annotations.json", path.parent
    return path / "annotations.json", path


def resolve_video_directory(
    trajectory_root: Path,
    video: str,
    *,
    allow_external_paths: bool = False,
) -> tuple[Path, Path]:
    root = Path(trajectory_root).expanduser().resolve()
    portable = str(video).replace("\\", "/")
    relative = Path(portable)
    if relative.is_absolute() or PureWindowsPath(portable).is_absolute():
        if not allow_external_paths:
            raise ValueError("absolute annotation video paths are disabled")
        resolved = relative.expanduser().resolve()
        return resolved, resolved.parent
    if not portable or any(part in ("", "..") for part in relative.parts):
        raise ValueError(f"unsafe relative video path: {video!r}")
    resolved = (root / relative).resolve()
    if not allow_external_paths:
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"annotation video path escapes trajectory root: {video!r}"
            ) from error
    return resolved, root


def _logical_scene(value: Any) -> str:
    return logical_scene_id(value)


def scene_from_video(video: str) -> str:
    name = Path(str(video).replace("\\", "/")).name
    if "_satnav_" not in name:
        raise ValueError(f"trajectory video name does not encode a scene: {video!r}")
    return _logical_scene(name.rsplit("_satnav_", 1)[0])


def _instruction_text(value: Any) -> str:
    return normalized_instruction_text(value)


def _join_instruction(value: Any) -> str:
    return _instruction_text(value).lower()


def _episode_list(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("episodes")
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ValueError("episode metadata must contain a list of objects")
    return payload


def resolve_train_episode_path(annotation_path: Path) -> Path:
    configured = os.environ.get("SATNAV_OPENFLY_TRAIN_EPISODES", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    trajectory_root = Path(annotation_path).resolve().parent
    return trajectory_root.parent / "episodes" / "train" / "all_episodes.json"


def load_trajectory_type_index(path: Path) -> Mapping[tuple[str, str, str], str]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        episodes = _episode_list(json.load(handle))
    index: dict[tuple[str, str, str], str] = {}
    for episode in episodes:
        key = trajectory_training_key(episode)
        trajectory_type = str(episode.get("trajectory_type", "")).strip()
        if not key[1] or not trajectory_type:
            raise ValueError(f"incomplete episode metadata key in {path}")
        if key in index:
            raise ValueError(
                "episode metadata composite key is not unique: "
                f"scene={key[0]!r}, trajectory_id={key[1]!r}"
            )
        index[key] = trajectory_type
    return index


def _validate_actions(item: Mapping[str, Any]) -> tuple[int, ...]:
    raw = item.get("actions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("annotation actions must be a non-empty list")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
        raise ValueError("annotation actions must be JSON integers, not coerced values")
    actions = tuple(raw)
    if actions[0] != -1:
        raise ValueError("canonical actions must begin with INIT (-1)")
    if any(action not in ACTION_TO_NAME for action in actions[1:]):
        raise ValueError("trajectory contains an unsupported primitive action")
    if actions[-1] != 0:
        raise ValueError("canonical actions must terminate with STOP (0)")
    steps = item.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("annotation steps must be a JSON integer")
    if steps != len(actions) - 1:
        raise ValueError("annotation steps must equal len(actions)-1")
    return actions


def load_annotation_records(
    configured: Path,
    *,
    max_episodes: Optional[int] = None,
    allow_external_paths: bool = False,
    require_episode_metadata: bool = True,
) -> tuple[Path, Optional[Path], list[AnnotationRecord]]:
    max_episodes = _positive_optional(max_episodes, "max_episodes")
    annotation_path, source_root = _annotation_path(configured)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path} must contain a JSON list")
    episode_path = resolve_train_episode_path(annotation_path)
    if require_episode_metadata and not episode_path.is_file():
        raise FileNotFoundError(
            "canonical OpenFly training requires train episode metadata: "
            f"{episode_path}"
        )
    type_index = (
        load_trajectory_type_index(episode_path) if episode_path.is_file() else {}
    )
    records: list[AnnotationRecord] = []
    matched_keys: set[tuple[str, str, str]] = set()
    for item in payload:
        if max_episodes is not None and len(records) >= max_episodes:
            break
        if not isinstance(item, Mapping):
            raise ValueError("trajectory annotation must be an object")
        instructions = item.get("instructions")
        if isinstance(instructions, str):
            instructions = [instructions]
        if not isinstance(instructions, list) or not instructions:
            raise ValueError("annotation instructions must be a non-empty list")
        instruction = _instruction_text(instructions[0])
        video = item.get("video")
        if not isinstance(video, str) or not video:
            raise ValueError("annotation video must be a non-empty string")
        video_dir, anchor = resolve_video_directory(
            source_root, video, allow_external_paths=allow_external_paths
        )
        scene = scene_from_video(video)
        trajectory_id = str(item.get("trajectory_id", ""))
        key = (scene, trajectory_id, _join_instruction(instruction))
        if key in matched_keys:
            raise ValueError(
                "trajectory annotations repeat an exact episode composite key: "
                f"scene={scene!r}, trajectory_id={trajectory_id!r}"
            )
        trajectory_type = type_index.get(key)
        if require_episode_metadata and trajectory_type is None:
            raise ValueError(
                "annotation has no exact train-episode composite match: "
                f"id={item.get('id')!r}, scene={scene!r}, trajectory_id={trajectory_id!r}"
            )
        matched_keys.add(key)
        records.append(
            AnnotationRecord(
                source_root=source_root,
                path_anchor=anchor,
                video_dir=video_dir,
                logical_scene=scene,
                instruction=instruction,
                actions=_validate_actions(item),
                trajectory_id=trajectory_id,
                trajectory_type=trajectory_type or "unknown",
                payload=dict(item),
                allow_external_paths=allow_external_paths,
            )
        )
    if require_episode_metadata and max_episodes is None:
        unmatched = set(type_index) - matched_keys
        if unmatched:
            example = sorted(unmatched)[0]
            raise ValueError(
                "train episode metadata has no exact trajectory annotation match: "
                f"scene={example[0]!r}, trajectory_id={example[1]!r}"
            )
    return annotation_path, episode_path if episode_path.is_file() else None, records


def frame_path(record: AnnotationRecord, frame_index: int) -> Path:
    path = (record.video_dir / "rgb" / f"{int(frame_index):03d}.jpg").resolve()
    if not record.allow_external_paths:
        try:
            path.relative_to(record.path_anchor)
        except ValueError as error:
            raise ValueError("RGB frame symlink escapes trajectory root") from error
    return path


def _history_indices(current: int) -> tuple[int, int]:
    if current <= 2:
        return 1, 1
    return current - 1, current - 2


def _stop_history_variants(current: int, count: int) -> list[tuple[int, int]]:
    variants: list[tuple[int, int]] = []
    seen = set()
    for shift in range(max(int(count), 1)):
        pair = (
            _history_indices(current)
            if shift == 0
            else (
                max(1, current - shift - 1),
                max(1, current - shift - 2),
            )
        )
        if pair not in seen:
            variants.append(pair)
            seen.add(pair)
        if len(variants) >= count:
            break
    return variants


def select_step_indices(actions: Sequence[int], policy: SamplingPolicy) -> list[int]:
    last = len(actions) - 1
    if last <= 0:
        return []
    kept: set[int] = set()
    for index in range(1, min(policy.head_keep + 1, last + 1)):
        if int(actions[index]) in ACTION_TO_NAME:
            kept.add(index)
    kept.add(last)
    tail_start = max(policy.head_keep + 1, last - policy.tail_keep)
    for index in range(tail_start, last):
        if int(actions[index]) in ACTION_TO_NAME:
            kept.add(index)
    consecutive_forward = 0
    middle_end = min(last - 1, tail_start - 1)
    for index in range(policy.head_keep + 1, middle_end + 1):
        action = int(actions[index])
        if action == 1:
            if consecutive_forward % policy.sample_stride == 0:
                kept.add(index)
            consecutive_forward += 1
        elif action in ACTION_TO_NAME:
            kept.add(index)
            consecutive_forward = 0
        else:
            consecutive_forward = 0
    return sorted(kept)


def sample_action(
    record: AnnotationRecord, step_index: int, policy: SamplingPolicy
) -> int:
    last = len(record.actions) - 1
    in_stop_window = (
        step_index == last
        if policy.stop_window <= 0
        else step_index >= max(1, last - policy.stop_window + 1)
    )
    return 0 if in_stop_window else int(record.actions[step_index])


def iter_sample_indices(
    records: Sequence[AnnotationRecord], policy: SamplingPolicy
) -> Iterable[SampleIndex]:
    for record_index, record in enumerate(records):
        for step_index in select_step_indices(record.actions, policy):
            action = sample_action(record, step_index, policy)
            if action == 0:
                variants = _stop_history_variants(step_index, policy.stop_history_aug)
                repeat = policy.stop_repeat
            else:
                variants = [_history_indices(step_index)]
                repeat = 1
            for repeat_index in range(repeat):
                first, second = variants[repeat_index % len(variants)]
                yield SampleIndex(record_index, step_index, first, second)


def build_sample_indices(
    records: Sequence[AnnotationRecord],
    policy: SamplingPolicy,
    *,
    max_samples: Optional[int] = None,
) -> list[SampleIndex]:
    max_samples = _positive_optional(max_samples, "max_samples")
    samples = []
    for sample in iter_sample_indices(records, policy):
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def sample_summary(
    records: Sequence[AnnotationRecord], policy: SamplingPolicy
) -> Mapping[str, Any]:
    actions: Counter[int] = Counter()
    types: Counter[str] = Counter()
    total = 0
    for sample in iter_sample_indices(records, policy):
        record = records[sample.record_index]
        actions[sample_action(record, sample.step_index, policy)] += 1
        types[record.trajectory_type] += 1
        total += 1
    return {
        "samples": total,
        "action_counts": {str(key): actions[key] for key in sorted(actions)},
        "trajectory_type_counts": dict(sorted(types.items())),
        "sampling_policy": policy.to_payload(),
    }


class SatNavOpenFlyDataset:
    def __init__(
        self,
        trajectory_root: Path,
        *,
        action_format: str = COMPACT,
        policy: Optional[SamplingPolicy] = None,
        max_episodes: Optional[int] = None,
        max_samples: Optional[int] = None,
        frame_identity_index: Optional[Path] = None,
        frame_identity_index_artifact: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.policy = policy or sampling_policy_from_environment()
        annotation, episode_path, records = load_annotation_records(
            trajectory_root,
            max_episodes=max_episodes,
            allow_external_paths=False,
            require_episode_metadata=True,
        )
        self.annotation_path = annotation
        self.episode_path = episode_path
        self.records = records
        self._frame_offsets = []
        frame_offset = 0
        for record in records:
            self._frame_offsets.append(frame_offset)
            frame_offset += len(record.actions)
        self._frame_count = frame_offset
        self.frame_identity_index_path = (
            Path(os.path.abspath(Path(frame_identity_index).expanduser()))
            if frame_identity_index is not None
            else None
        )
        self.frame_identity_index_artifact = (
            dict(frame_identity_index_artifact)
            if frame_identity_index_artifact is not None
            else None
        )
        self._frame_index_snapshot: Optional[bytes] = None
        self._verified_frame_index_artifact: Optional[Mapping[str, Any]] = None
        if (self.frame_identity_index_path is None) != (
            self.frame_identity_index_artifact is None
        ):
            raise ValueError(
                "frame identity index path and artifact must be provided together"
            )
        if self.frame_identity_index_path is not None:
            index_size = int(self.frame_identity_index_artifact.get("size", -1))
            if (
                index_size % FRAME_DIGEST_BYTES != 0
                or index_size < self._frame_count * FRAME_DIGEST_BYTES
            ):
                raise ValueError("frame identity index size contradicts dataset frames")
            self.frame_identity_index_file_count = index_size // FRAME_DIGEST_BYTES
            self._ensure_frame_identity_index()
        else:
            self.frame_identity_index_file_count = 0
        self.samples = build_sample_indices(
            records, self.policy, max_samples=max_samples
        )
        self.action_format = resolve_action_format(action_format)
        self.action_counts: Counter[int] = Counter()
        self.trajectory_type_counts: Counter[str] = Counter()
        for sample in self.samples:
            record = records[sample.record_index]
            self.action_counts[
                sample_action(record, sample.step_index, self.policy)
            ] += 1
            self.trajectory_type_counts[record.trajectory_type] += 1
        print(
            "SatNavOpenFlyDataset: "
            f"episodes={len(records)}, samples={len(self.samples)}, "
            f"action_format={self.action_format}, "
            f"action_counts={dict(self.action_counts)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _ensure_frame_identity_index(self) -> bytes:
        if self.frame_identity_index_path is None:
            raise RuntimeError("OpenFly dataset has no verified frame identity index")
        if self._frame_index_snapshot is not None:
            return self._frame_index_snapshot
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.frame_identity_index_path, flags)
        except OSError as error:
            raise RuntimeError("cannot open immutable frame identity index") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError("frame identity index is not a regular file")
            expected_path = self.frame_identity_index_path.resolve(strict=True)
            descriptor_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            if descriptor_path != expected_path:
                raise RuntimeError("frame identity index changed while opening")
            digest = hashlib.sha256()
            chunks = []
            while True:
                chunk = os.read(descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise RuntimeError("frame identity index changed while hashing")
            actual = {
                "id": self.frame_identity_index_path.name,
                "sha256": digest.hexdigest(),
                "size": int(after.st_size),
            }
            if actual != self.frame_identity_index_artifact:
                raise RuntimeError(
                    "frame identity index content does not match validation"
                )
            snapshot = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(snapshot) != int(actual["size"]):
            raise RuntimeError("frame identity index snapshot is truncated")
        self._frame_index_snapshot = snapshot
        self._verified_frame_index_artifact = actual
        return snapshot

    def verify_frame_identity_index(self) -> Mapping[str, Any]:
        self._ensure_frame_identity_index()
        assert self._verified_frame_index_artifact is not None
        return dict(self._verified_frame_index_artifact)

    def _expected_frame_digest(self, record_index: int, frame_index: int) -> bytes:
        if not 0 <= record_index < len(self.records):
            raise IndexError("frame record index is outside the validated dataset")
        record = self.records[record_index]
        if not 1 <= int(frame_index) <= len(record.actions):
            raise IndexError("frame index is outside the validated trajectory")
        snapshot = self._ensure_frame_identity_index()
        ordinal = self._frame_offsets[record_index] + int(frame_index) - 1
        start = ordinal * FRAME_DIGEST_BYTES
        expected = snapshot[start : start + FRAME_DIGEST_BYTES]
        if len(expected) != FRAME_DIGEST_BYTES:
            raise RuntimeError("frame identity index is truncated")
        return expected

    def _verified_frame_bytes(self, record_index: int, frame_index: int) -> bytes:
        record = self.records[record_index]
        path = record.video_dir / "rgb" / f"{int(frame_index):03d}.jpg"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise RuntimeError(f"cannot open validated frame: {path}") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError("validated frame is no longer a regular file")
            descriptor_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            if not record.allow_external_paths:
                try:
                    descriptor_path.relative_to(record.path_anchor)
                except ValueError as error:
                    raise RuntimeError(
                        "validated frame escaped trajectory root"
                    ) from error
            chunks = []
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise RuntimeError("validated frame changed while reading")
        finally:
            os.close(descriptor)
        if self.frame_identity_index_path is not None:
            expected = self._expected_frame_digest(record_index, frame_index)
            if digest.digest() != expected:
                raise RuntimeError(
                    "consumed frame content does not match preflight identity"
                )
        return b"".join(chunks)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        sample = self.samples[index]
        record = self.records[sample.record_index]
        try:
            frame_indices = (
                sample.step_index,
                sample.history_first,
                sample.history_second,
            )
            from PIL import Image

            images = []
            frame_cache = {}
            for frame_index in frame_indices:
                if frame_index not in frame_cache:
                    frame_cache[frame_index] = self._verified_frame_bytes(
                        sample.record_index, frame_index
                    )
                with Image.open(io.BytesIO(frame_cache[frame_index])) as image:
                    images.append(image.convert("RGB").copy())
        except Exception as error:
            raise RuntimeError(
                "failed to load exact OpenFly sample "
                f"index={index}, annotation_id={record.payload.get('id')}, "
                f"step={sample.step_index}"
            ) from error
        action = sample_action(record, sample.step_index, self.policy)
        history = [
            int(value)
            for value in record.actions[: sample.step_index]
            if int(value) in ACTION_TO_NAME
        ]
        answer = action_to_text(action)
        return {
            "annotation_id": record.payload.get("id"),
            "trajectory_id": record.trajectory_id,
            "trajectory_type": record.trajectory_type,
            "step_index": sample.step_index,
            "instruction": record.instruction,
            "images": images,
            "action": action,
            "answer": answer,
            "raw_action_vector": get_original_action_vector(action),
            "normalized_action_vector": get_normalized_original_action_vector(action),
            "action_history": history,
            "prompt_text": build_openfly_prompt(
                record.instruction, history, self.policy.action_history_limit
            ),
            "full_text": build_openfly_prompt_with_answer(
                record.instruction, answer, history, self.policy.action_history_limit
            ),
        }


def _assistant_token_positions(tokenizer: Any, full_text: str, max_length: int):
    boundary = full_text.rfind(ASSISTANT_BOUNDARY)
    if boundary < 0:
        raise ValueError("last OpenFly assistant boundary is missing")
    boundary_end = boundary + len(ASSISTANT_BOUNDARY)
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned invalid offset mapping")
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(start) and int(start) >= boundary_end
    ]
    if not positions:
        raise ValueError("OpenFly assistant answer was truncated or has no tokens")
    return input_ids, positions


@dataclass
class OpenFlyDataCollator:
    processor: Any
    model_max_length: int
    pad_token_id: int
    action_format: str = COMPACT
    original_dim_loss_weights: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        self.action_format = resolve_action_format(self.action_format)
        self.original_supervised_dims = 4
        self.original_dim_loss_weights = resolve_original_dim_loss_weights(
            self.original_dim_loss_weights
        )
        if self.action_format == ORIGINAL:
            from baselines.vlm.openfly.action_tokenizer import ActionTokenizer

            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        else:
            self.action_tokenizer = None

    def __call__(self, instances: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor
        input_rows = []
        label_rows = []
        attention_rows = []
        weight_rows = []
        pixel_rows = []
        for instance in instances:
            if self.action_format == COMPACT:
                full_text = str(instance["full_text"])
            else:
                normalized_action = instance["normalized_action_vector"]
                action_token_ids = self.action_tokenizer.encode_action_token_ids(
                    normalized_action
                ).tolist()
                answer = self.action_tokenizer(normalized_action)
                full_text = build_openfly_prompt_with_answer(
                    instance["instruction"],
                    answer,
                    instance.get("action_history", ()),
                    self._history_limit(),
                )
            ids, assistant_positions = _assistant_token_positions(
                tokenizer, full_text, self.model_max_length
            )
            labels = [IGNORE_INDEX] * len(ids)
            weights = [0.0] * len(ids)
            if self.action_format == COMPACT:
                selected = assistant_positions
                for position in selected:
                    labels[position] = ids[position]
            else:
                selected = self._exact_action_positions(
                    ids, assistant_positions, action_token_ids
                )[: self.original_supervised_dims]
                if len(selected) != self.original_supervised_dims:
                    raise ValueError(
                        "original action answer has incomplete supervision"
                    )
                for weight_index, position in enumerate(selected):
                    labels[position] = ids[position]
                    weights[position] = float(
                        self.original_dim_loss_weights[weight_index]
                    )
            if not any(value != IGNORE_INDEX for value in labels):
                raise ValueError("OpenFly batch item has no assistant supervision")
            input_rows.append(torch.tensor(ids, dtype=torch.long))
            label_rows.append(torch.tensor(labels, dtype=torch.long))
            attention_rows.append(torch.ones(len(ids), dtype=torch.bool))
            if self.action_format == ORIGINAL:
                weight_rows.append(torch.tensor(weights, dtype=torch.float32))
            pixels = image_processor(images=instance["images"], return_tensors="pt")[
                "pixel_values"
            ]
            pixel_rows.append(pixels)
        input_ids = pad_sequence(
            input_rows, batch_first=True, padding_value=self.pad_token_id
        )
        labels = pad_sequence(label_rows, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequence(
            attention_rows, batch_first=True, padding_value=False
        )
        batch = {
            "input_ids": input_ids[:, : self.model_max_length],
            "labels": labels[:, : self.model_max_length],
            "attention_mask": attention_mask[:, : self.model_max_length],
            "pixel_values": torch.stack(pixel_rows, dim=0),
        }
        if self.action_format == ORIGINAL:
            batch["loss_weights"] = pad_sequence(
                weight_rows, batch_first=True, padding_value=0.0
            )[:, : self.model_max_length]
        return batch

    @staticmethod
    def _history_limit() -> int:
        configured = os.environ.get("OPENFLY_ACTION_HISTORY_LIMIT", "").strip()
        return int(configured) if configured else int(DEFAULT_ACTION_HISTORY_LIMIT)

    @staticmethod
    def _exact_action_positions(
        input_ids: Sequence[int],
        assistant_positions: Sequence[int],
        action_token_ids: Sequence[int],
    ) -> list[int]:
        if not action_token_ids:
            raise ValueError("original OpenFly action encoded to zero tokens")
        allowed = set(int(position) for position in assistant_positions)
        width = len(action_token_ids)
        for start in assistant_positions:
            stop = int(start) + width
            positions = list(range(int(start), stop))
            if not all(position in allowed for position in positions):
                continue
            if list(input_ids[int(start) : stop]) == list(action_token_ids):
                return positions
        raise ValueError(
            "original OpenFly action tokens are absent or truncated in assistant answer"
        )


def validate_records(
    annotation_path: Path,
    episode_path: Optional[Path],
    records: Sequence[AnnotationRecord],
    *,
    policy: SamplingPolicy,
    strict_frames: bool,
    hash_frame_content: bool,
    decode_samples: int,
    frame_identity_index: Optional[Path] = None,
) -> Mapping[str, Any]:
    if hash_frame_content and not strict_frames:
        raise ValueError("--hash-frame-content requires --strict-frames")
    if frame_identity_index is not None and not (strict_frames and hash_frame_content):
        raise ValueError(
            "frame identity index requires strict content-hashed frame validation"
        )
    index_path: Optional[Path] = None
    index_temporary: Optional[Path] = None
    index_handle = None
    if frame_identity_index is not None:
        index_path = Path(os.path.abspath(Path(frame_identity_index).expanduser()))
        if index_path.is_symlink() or (
            index_path.exists() and not index_path.is_file()
        ):
            raise ValueError("frame identity index target must be a regular file")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_temporary = index_path.with_name(
            f".{index_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        index_handle = os.fdopen(os.open(index_temporary, flags, 0o600), "wb")
    total_frames = 0
    total_actions = 0
    total_bytes = 0
    decoded = 0
    tree = hashlib.sha256()
    decode_stride = max(len(records) // max(int(decode_samples), 1), 1)
    for record_index, record in enumerate(records):
        expected_count = len(record.actions)
        total_frames += expected_count
        total_actions += expected_count - 1
        rgb = record.video_dir / "rgb"
        if not rgb.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {rgb}")
        first, last = frame_path(record, 1), frame_path(record, expected_count)
        if not first.is_file() or not last.is_file():
            raise FileNotFoundError(
                f"trajectory {record.payload.get('id')} misses first/last frame"
            )
        if strict_frames:
            entries = sorted(
                (
                    entry
                    for entry in os.scandir(rgb)
                    if entry.is_file(follow_symlinks=False)
                    and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
                ),
                key=lambda entry: entry.name,
            )
            expected_names = [
                f"{index:03d}.jpg" for index in range(1, expected_count + 1)
            ]
            if [entry.name for entry in entries] != expected_names:
                raise ValueError(
                    f"trajectory {record.payload.get('id')} frame/action mapping mismatch"
                )
            video = str(record.payload["video"]).replace("\\", "/").strip("/")
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(entry.path, flags)
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError("RGB frame is not a regular file")
                    if (
                        entry_stat.st_dev,
                        entry_stat.st_ino,
                        entry_stat.st_size,
                        entry_stat.st_mtime_ns,
                        entry_stat.st_ctime_ns,
                    ) != (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ):
                        raise RuntimeError("RGB frame changed before hashing")
                    total_bytes += int(before.st_size)
                    tree.update(
                        f"{video}/rgb/{entry.name}\0{before.st_size}\0".encode()
                    )
                    if hash_frame_content:
                        frame_digest = hashlib.sha256()
                        while True:
                            chunk = os.read(descriptor, 1024 * 1024)
                            if not chunk:
                                break
                            tree.update(chunk)
                            frame_digest.update(chunk)
                        tree.update(b"\0")
                        if index_handle is not None:
                            index_handle.write(frame_digest.digest())
                    else:
                        tree.update(f"{before.st_mtime_ns}\n".encode())
                    after = os.fstat(descriptor)
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise RuntimeError("RGB frame changed while hashing")
                finally:
                    os.close(descriptor)
        if (
            decode_samples > 0
            and record_index % decode_stride == 0
            and decoded < decode_samples
        ):
            from PIL import Image

            with Image.open(last) as image:
                image.convert("RGB").load()
            decoded += 1
    if index_handle is not None:
        index_handle.flush()
        os.fsync(index_handle.fileno())
        index_handle.close()
        assert index_path is not None and index_temporary is not None
        if index_temporary.stat().st_size != total_frames * FRAME_DIGEST_BYTES:
            index_temporary.unlink(missing_ok=True)
            raise RuntimeError("frame identity index length is inconsistent")
        os.replace(index_temporary, index_path)
    result: dict[str, Any] = {
        "status": "passed",
        "annotations": len(records),
        "primitive_actions_excluding_init": total_actions,
        "mapped_frames": total_frames,
        "decoded_samples": decoded,
        "strict_frames": bool(strict_frames),
        "annotation_artifact": file_identity(annotation_path),
        "episode_metadata_artifact": file_identity(episode_path)
        if episode_path
        else None,
        "episode_trajectory_type_counts": dict(
            sorted(Counter(record.trajectory_type for record in records).items())
        ),
        **sample_summary(records, policy),
    }
    if strict_frames:
        result["frame_tree"] = {
            "algorithm": (
                "sha256-relative-path-size-content-v1"
                if hash_frame_content
                else "sha256-relative-path-size-mtime-v1"
            ),
            "digest": tree.hexdigest(),
            "file_count": total_frames,
            "total_bytes": total_bytes,
        }
    if index_path is not None:
        result["frame_identity_index"] = {
            "algorithm": FRAME_IDENTITY_INDEX_ALGORITHM,
            "digest_size_bytes": FRAME_DIGEST_BYTES,
            "ordering": FRAME_IDENTITY_INDEX_ORDER,
            "file_count": total_frames,
            "artifact": file_identity(index_path),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical SatNav trajectory_data for OpenFly"
    )
    parser.add_argument("trajectory_root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--strict-frames", action="store_true")
    parser.add_argument("--hash-frame-content", action="store_true")
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    policy = sampling_policy_from_environment()
    annotation, episodes, records = load_annotation_records(
        args.trajectory_root,
        allow_external_paths=False,
        require_episode_metadata=True,
    )
    frame_identity_index = None
    if args.report is not None and args.strict_frames and args.hash_frame_content:
        frame_identity_index = args.report.with_name(
            f"{args.report.name}.frames.sha256"
        )
    report = validate_records(
        annotation,
        episodes,
        records,
        policy=policy,
        strict_frames=args.strict_frames,
        hash_frame_content=args.hash_frame_content,
        decode_samples=args.decode_samples,
        frame_identity_index=frame_identity_index,
    )
    if (
        args.expected_episodes is not None
        and report["annotations"] != args.expected_episodes
    ):
        raise ValueError(
            f"expected {args.expected_episodes} annotations, found {report['annotations']}"
        )
    if args.expected_samples is not None and report["samples"] != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} samples, found {report['samples']}"
        )
    report = dict(
        report,
        allow_external_paths=False,
        validation_run_id=os.environ.get("SATNAV_VALIDATION_RUN_ID", "standalone"),
        completed_at_ns=time.time_ns(),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

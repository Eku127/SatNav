"""Safe SatNav trajectory-data adapter for pinned NaVILA training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from baselines.vlm.navila.actions import (
    ACTION_NAMES,
    action_text,
    build_prompt,
    normalize_action_format,
)
from baselines.vlm.navila.artifacts import file_identity


# NaVILA's AUTO conversation preprocessor needs a marker that is removed before
# producing model inputs.  The pinned Llama-3 tokenizer already treats this
# reserved token as one indivisible, skipped special token.  Reusing it avoids
# persisting NaVILA's lazily-added <vila/sentinel> without changing inputs or
# supervision labels.
TRAINING_SENTINEL_TOKEN = "<|reserved_special_token_250|>"


@dataclass(frozen=True)
class AnnotationRecord:
    """Validated runtime view of one trajectory annotation."""

    source_root: Path
    path_anchor: Path
    video_dir: Path
    instructions: Tuple[str, ...]
    actions: Tuple[int, ...]
    payload: Mapping[str, Any]
    allow_external_paths: bool = False


@dataclass(frozen=True)
class SampleIndex:
    record_index: int
    instruction_index: int
    step_index: int


def _positive_optional(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _env_positive(name: str) -> Optional[int]:
    value = os.environ.get(name, "").strip()
    return None if not value else _positive_optional(int(value), name)


def resolve_video_directory(
    trajectory_root: Path,
    video: str,
    *,
    allow_external_paths: bool = False,
) -> Tuple[Path, Path]:
    """Resolve canonical/legacy layouts and reject escape paths by default."""

    root = Path(trajectory_root).expanduser().resolve()
    portable = str(video).replace("\\", "/")
    relative = Path(portable)
    if relative.is_absolute() or PureWindowsPath(portable).is_absolute():
        if not allow_external_paths:
            raise ValueError("absolute annotation video paths are disabled")
        resolved = relative.expanduser().resolve()
        return resolved, resolved.parent
    if not portable or any(part == ".." for part in relative.parts):
        raise ValueError(f"unsafe relative video path: {video!r}")

    candidates = [root / relative]
    if root.name == "images" and relative.parts and relative.parts[0] == "images":
        candidates.insert(0, root.parent / relative)
        if len(relative.parts) > 1:
            candidates.append(root.joinpath(*relative.parts[1:]))
    anchor = root.parent if root.name == "images" else root
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not allow_external_paths:
            try:
                resolved.relative_to(anchor)
            except ValueError as error:
                raise ValueError(
                    f"annotation video path escapes trajectory root: {video!r}"
                ) from error
        if resolved not in unique:
            unique.append(resolved)
    for candidate in unique:
        if (candidate / "rgb").is_dir():
            return candidate, anchor
    return unique[0], anchor


def _annotation_path(configured: Path) -> Tuple[Path, Path]:
    root = Path(configured).expanduser().resolve()
    if root.is_file():
        return root, root.parent
    if (
        root.name == "images"
        and not (root / "annotations.json").is_file()
        and (root.parent / "annotations.json").is_file()
    ):
        return root.parent / "annotations.json", root
    return root / "annotations.json", root


def _validate_actions(item: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = item.get("actions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("annotation actions must be a non-empty list")
    actions = tuple(int(action) for action in raw)
    if actions[0] != -1:
        raise ValueError("canonical trajectory actions must begin with INIT (-1)")
    if any(action not in ACTION_NAMES for action in actions[1:]):
        raise ValueError("trajectory contains an unsupported primitive action")
    if actions[-1] != 0:
        raise ValueError("canonical trajectory actions must end with STOP (0)")
    steps = item.get("steps")
    if steps is not None and int(steps) != len(actions) - 1:
        raise ValueError("annotation steps does not match actions minus INIT")
    return actions


def load_annotation_records(
    configured: Path,
    *,
    max_episodes: Optional[int] = None,
    allow_external_paths: bool = False,
) -> Tuple[Path, List[AnnotationRecord]]:
    """Load annotations while enforcing path and schema boundaries."""

    max_episodes = _positive_optional(max_episodes, "max_episodes")
    annotation_path, source_root = _annotation_path(configured)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path} must contain a JSON list")
    records: List[AnnotationRecord] = []
    for item in payload:
        if max_episodes is not None and len(records) >= max_episodes:
            break
        if not isinstance(item, Mapping):
            raise ValueError("trajectory annotation must be an object")
        instructions = item.get("instructions")
        if isinstance(instructions, str):
            instructions = [instructions]
        if not isinstance(instructions, Sequence) or not instructions:
            raise ValueError("annotation instructions must be a non-empty list/string")
        video = item.get("video")
        if not isinstance(video, str) or not video:
            raise ValueError("annotation video must be a non-empty string")
        video_dir, anchor = resolve_video_directory(
            source_root, video, allow_external_paths=allow_external_paths
        )
        records.append(
            AnnotationRecord(
                source_root=source_root,
                path_anchor=anchor,
                video_dir=video_dir,
                instructions=tuple(str(value) for value in instructions),
                actions=_validate_actions(item),
                payload=dict(item),
                allow_external_paths=allow_external_paths,
            )
        )
    return annotation_path, records


def _sample_kept(
    record: AnnotationRecord,
    step_index: int,
    ratio: Optional[float],
    stride: Optional[int],
) -> bool:
    if stride is not None and stride > 1 and step_index % stride:
        return False
    if ratio is not None and ratio < 1.0:
        key = (
            f"{record.payload.get('id', '')}|"
            f"{record.payload.get('trajectory_id', '')}|{step_index}"
        )
        hashed = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        return (hashed / 0xFFFFFFFF) < ratio
    return True


def _head_stop_stride(actions: Sequence[int], head: int, stride: int) -> List[int]:
    last = len(actions) - 1
    kept = {
        index
        for index in range(1, min(head + 1, last + 1))
        if actions[index] in ACTION_NAMES
    }
    kept.add(last)
    consecutive_forward = 0
    for index in range(head + 1, last):
        action = actions[index]
        if action == 1:
            if consecutive_forward % max(stride, 1) == 0:
                kept.add(index)
            consecutive_forward += 1
        else:
            kept.add(index)
            consecutive_forward = 0
    return sorted(kept)


def build_sample_indices(
    records: Sequence[AnnotationRecord],
    *,
    max_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    sample_stride: Optional[int] = None,
    head_keep: Optional[int] = None,
    stop_repeat: int = 1,
) -> List[SampleIndex]:
    max_samples = _positive_optional(max_samples, "max_samples")
    if sample_ratio is not None and not 0.0 < float(sample_ratio) <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    if stop_repeat <= 0:
        raise ValueError("stop_repeat must be positive")
    samples = []
    for record_index, record in enumerate(records):
        if head_keep is None:
            steps = [
                index
                for index in range(1, len(record.actions))
                if _sample_kept(record, index, sample_ratio, sample_stride)
            ]
        else:
            steps = _head_stop_stride(
                record.actions, int(head_keep), int(sample_stride or 1)
            )
        for instruction_index, _ in enumerate(record.instructions):
            for step_index in steps:
                repeat = stop_repeat if record.actions[step_index] == 0 else 1
                for _ in range(repeat):
                    samples.append(
                        SampleIndex(record_index, instruction_index, step_index)
                    )
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    return samples


def frame_path(record: AnnotationRecord, frame_index: int) -> Path:
    path = (record.video_dir / "rgb" / f"{int(frame_index):03d}.jpg").resolve()
    if not record.allow_external_paths:
        try:
            path.relative_to(record.path_anchor)
        except ValueError as error:
            raise ValueError("RGB frame symlink escapes trajectory root") from error
    return path


class SatNavNaVILADataset:
    """Lazy NaVILA next-action supervision over SatNav trajectory exports."""

    def __init__(self, data_path, image_folder, tokenizer, data_args, training_args):
        del training_args
        configured = Path(data_path) if Path(data_path).is_dir() else Path(data_path)
        if Path(data_path).is_file():
            configured = Path(data_path)
        elif image_folder:
            configured = Path(image_folder)
        allow_external = os.environ.get(
            "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS", ""
        ).lower() in ("1", "true", "yes", "on")
        _, records = load_annotation_records(
            configured,
            max_episodes=_env_positive("SATNAV_MAX_EPISODES"),
            allow_external_paths=allow_external,
        )
        ratio_value = os.environ.get("SATNAV_SAMPLE_RATIO", "").strip()
        stride_value = os.environ.get("SATNAV_SAMPLE_STRIDE", "").strip()
        head_value = os.environ.get("SATNAV_HEAD_KEEP", "").strip()
        stop_value = os.environ.get("SATNAV_STOP_REPEAT", "1").strip()
        self.records = records
        self.samples = build_sample_indices(
            records,
            max_samples=_env_positive("SATNAV_MAX_SAMPLES"),
            sample_ratio=float(ratio_value) if ratio_value else None,
            sample_stride=int(stride_value) if stride_value else None,
            head_keep=int(head_value) if head_value else None,
            stop_repeat=int(stop_value),
        )
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.action_format = normalize_action_format(
            os.environ.get("SATNAV_ACTION_FORMAT", "compact")
        )
        self.action_counts = Counter(
            records[sample.record_index].actions[sample.step_index]
            for sample in self.samples
        )
        print(
            "SatNavNaVILADataset: "
            f"episodes={len(records)}, samples={len(self.samples)}, "
            f"action_format={self.action_format}, "
            f"action_counts={dict(self.action_counts)}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [sample.step_index + 100 for sample in self.samples]

    @property
    def modality_lengths(self):
        return self.lengths

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception as error:
            sample = self.samples[index]
            record = self.records[sample.record_index]
            raise RuntimeError(
                "Failed to load SatNav NaVILA sample "
                f"index={index}, episode_id={record.payload.get('id')}, "
                f"step_index={sample.step_index}"
            ) from error

    def _get_item(self, index) -> Dict[str, Any]:
        import torch
        from llava.data.dataset import preprocess
        from llava.mm_utils import process_image, vlnce_frame_sampling

        sample = self.samples[index]
        record = self.records[sample.record_index]
        paths = [
            str(frame_path(record, frame_index))
            for frame_index in range(1, sample.step_index + 1)
        ]
        sampled = vlnce_frame_sampling(
            paths, num_frames=int(self.data_args.num_video_frames)
        )
        images = torch.stack(
            [
                process_image(image, self.data_args, image_folder=None)
                for image in sampled
            ]
        )
        instruction = record.instructions[sample.instruction_index]
        if TRAINING_SENTINEL_TOKEN in instruction:
            raise ValueError("instruction contains the reserved training sentinel")
        prompt = build_prompt(
            instruction,
            max(int(self.data_args.num_video_frames) - 1, 0),
            self.action_format,
        )
        action = record.actions[sample.step_index]
        conversation = [
            [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": action_text(action, self.action_format)},
            ]
        ]
        tokenized = preprocess(
            copy.deepcopy(conversation), self.tokenizer, has_image=True
        )
        return {
            "input_ids": tokenized["input_ids"][0],
            "labels": tokenized["labels"][0],
            "image": images,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical SatNav trajectory_data for NaVILA"
    )
    parser.add_argument("trajectory_root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--strict-frames", action="store_true")
    parser.add_argument("--hash-frame-content", action="store_true")
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--allow-external-paths", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def validate_records(
    annotation_path: Path,
    records: Sequence[AnnotationRecord],
    *,
    strict_frames: bool,
    decode_samples: int,
    hash_frame_content: bool = False,
) -> Mapping[str, Any]:
    if hash_frame_content and not strict_frames:
        raise ValueError("--hash-frame-content requires --strict-frames")
    decoded = 0
    stride = max(len(records) // max(decode_samples, 1), 1)
    total_frames = 0
    total_frame_bytes = 0
    total_actions = 0
    frame_tree_digest = hashlib.sha256()
    for record_index, record in enumerate(records):
        expected = len(record.actions)
        total_actions += expected - 1
        rgb = record.video_dir / "rgb"
        if not rgb.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {rgb}")
        first = frame_path(record, 1)
        last = frame_path(record, expected)
        if not first.is_file() or not last.is_file():
            raise FileNotFoundError(
                f"trajectory {record.payload.get('id')} misses mapped first/last frame"
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
            names = [entry.name for entry in entries]
            expected_names = [f"{index:03d}.jpg" for index in range(1, expected + 1)]
            if names != expected_names:
                raise ValueError(
                    f"trajectory {record.payload.get('id')} frame/action mapping mismatch"
                )
            video = str(record.payload["video"]).replace("\\", "/").strip("/")
            for entry in entries:
                stat = entry.stat(follow_symlinks=False)
                total_frame_bytes += int(stat.st_size)
                frame_tree_digest.update(
                    f"{video}/rgb/{entry.name}\0{stat.st_size}\0".encode("utf-8")
                )
                if hash_frame_content:
                    with open(entry.path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            frame_tree_digest.update(chunk)
                    frame_tree_digest.update(b"\0")
                else:
                    frame_tree_digest.update(f"{stat.st_mtime_ns}\n".encode("utf-8"))
        total_frames += expected
        if (
            decode_samples > 0
            and record_index % stride == 0
            and decoded < decode_samples
        ):
            from PIL import Image

            with Image.open(last) as image:
                image.convert("RGB").load()
            decoded += 1
    result = {
        "status": "passed",
        "annotations": len(records),
        "primitive_actions_excluding_init": total_actions,
        "mapped_frames": total_frames,
        "decoded_samples": decoded,
        "strict_frames": strict_frames,
        "annotation_artifact": file_identity(annotation_path),
    }
    if strict_frames:
        result["frame_tree"] = {
            "algorithm": (
                "sha256-relative-path-size-content-v1"
                if hash_frame_content
                else "sha256-relative-path-size-mtime-v1"
            ),
            "digest": frame_tree_digest.hexdigest(),
            "file_count": total_frames,
            "total_bytes": total_frame_bytes,
        }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    annotation_path, records = load_annotation_records(
        args.trajectory_root, allow_external_paths=args.allow_external_paths
    )
    if args.expected_episodes is not None and len(records) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} annotations, found {len(records)}"
        )
    report = validate_records(
        annotation_path,
        records,
        strict_frames=args.strict_frames,
        decode_samples=args.decode_samples,
        hash_frame_content=args.hash_frame_content,
    )
    report = dict(
        report,
        allow_external_paths=bool(args.allow_external_paths),
        completed_at_ns=time.time_ns(),
        validation_run_id=os.environ.get("SATNAV_VALIDATION_RUN_ID", "standalone"),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

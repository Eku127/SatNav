"""Safe SatNav trajectory-data adapter for pinned Uni-NaVid training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from baselines.vlm.uninavid.actions import ACTION_NAMES, build_prompt, format_action_target
from baselines.vlm.uninavid.artifacts import file_identity
from baselines.vlm.uninavid.bootstrap import install_image_dataset_decord_stub


@dataclass(frozen=True)
class AnnotationRecord:
    source_root: Path
    path_anchor: Path
    video_dir: Path
    instruction: str
    actions: Tuple[int, ...]
    payload: Mapping[str, Any]
    allow_external_paths: bool = False


@dataclass(frozen=True)
class SampleIndex:
    record_index: int
    window_index: int
    start: int
    actions: Tuple[int, int, int, int]

    @property
    def history_frames(self) -> int:
        return self.start + 1


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean")


def _annotation_path(configured: Path) -> Tuple[Path, Path]:
    root = Path(configured).expanduser().resolve()
    if root.is_file():
        return root, root.parent
    return root / "annotations.json", root


def resolve_video_directory(
    trajectory_root: Path,
    video: str,
    *,
    allow_external_paths: bool = False,
) -> Tuple[Path, Path]:
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
    candidate = (root / relative).resolve()
    if not allow_external_paths:
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"annotation video path escapes root: {video!r}") from error
    return candidate, root


def _validated_actions(item: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = item.get("actions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        raise ValueError("annotation actions must contain INIT and at least one action")
    actions = tuple(int(action) for action in raw)
    if actions[0] != -1:
        raise ValueError("canonical trajectory actions must begin with INIT (-1)")
    if any(action not in ACTION_NAMES for action in actions[1:]):
        raise ValueError("trajectory contains an unsupported primitive action")
    if actions[-1] != 0:
        raise ValueError("canonical trajectory actions must end with STOP (0)")
    if "steps" in item and int(item["steps"]) != len(actions) - 1:
        raise ValueError("annotation steps does not match actions minus INIT")
    return actions


def load_annotation_records(
    configured: Path,
    *,
    max_episodes: Optional[int] = None,
    allow_external_paths: bool = False,
) -> Tuple[Path, List[AnnotationRecord]]:
    max_episodes = _positive_optional(max_episodes, "max_episodes")
    annotation_path, source_root = _annotation_path(configured)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path} must contain a JSON list")
    records = []
    for item in payload:
        if max_episodes is not None and len(records) >= max_episodes:
            break
        if not isinstance(item, Mapping):
            raise ValueError("trajectory annotation must be an object")
        if not str(item.get("id", "")):
            raise ValueError("annotation id must be non-empty")
        instructions = item.get("instructions")
        if not isinstance(instructions, Sequence) or isinstance(
            instructions, (str, bytes)
        ) or not instructions:
            raise ValueError("annotation instructions must be a non-empty list")
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
                instruction=str(instructions[0]),
                actions=_validated_actions(item),
                payload=dict(item),
                allow_external_paths=allow_external_paths,
            )
        )
    return annotation_path, records


def build_sample_indices(
    records: Sequence[AnnotationRecord],
    *,
    window_size: int = 4,
    max_samples: Optional[int] = None,
) -> List[SampleIndex]:
    if int(window_size) != 4:
        raise ValueError("the pinned Uni-NaVid contract requires window_size=4")
    max_samples = _positive_optional(max_samples, "max_samples")
    samples = []
    for record_index, record in enumerate(records):
        real = record.actions[1:]
        for start in range(0, len(real), window_size):
            window = list(real[start : start + window_size])
            window.extend([0] * (window_size - len(window)))
            samples.append(
                SampleIndex(
                    record_index=record_index,
                    window_index=start // window_size,
                    start=start,
                    actions=tuple(window),
                )
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


class SatNavUniNaVidDataset:
    """Four-action window supervision with the frozen prompt/target contract."""

    def __init__(self, data_path, tokenizer, data_args, window_size: int = 4):
        import torch

        self._torch = torch
        allow_external = _env_bool(
            "SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS", False
        )
        annotation_path, records = load_annotation_records(
            Path(data_path),
            max_episodes=_env_positive("SATNAV_MAX_EPISODES"),
            allow_external_paths=allow_external,
        )
        self.annotation_path = annotation_path
        self.records = records
        self.samples = build_sample_indices(
            records,
            window_size=window_size,
            max_samples=_env_positive("SATNAV_MAX_SAMPLES"),
        )
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.augmentation = _env_bool("SATNAV_UNINAVID_AUGMENTATION", True)
        self.action_counts = Counter(
            action for sample in self.samples for action in sample.actions
        )
        print(
            "SatNavUniNaVidDataset: "
            f"episodes={len(records)}, samples={len(self.samples)}, "
            f"augmentation={self.augmentation}, "
            f"action_counts={dict(self.action_counts)}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [sample.history_frames + 100 for sample in self.samples]

    @property
    def modality_lengths(self):
        return self.lengths

    def __getitem__(self, index):
        try:
            return self._load_sample(self.samples[index])
        except Exception as error:
            sample = self.samples[index]
            record = self.records[sample.record_index]
            raise RuntimeError(
                "Failed to load SatNav Uni-NaVid sample "
                f"index={index}, episode_id={record.payload.get('id')}, "
                f"window_index={sample.window_index}"
            ) from error

    def _load_sample(self, sample: SampleIndex):
        import numpy as np
        from PIL import Image
        install_image_dataset_decord_stub()
        from uninavid.constants import DEFAULT_IMAGE_TOKEN, NAVIGATION_IDENTIFIER
        from uninavid.train.train import (
            duplicate_with_probability,
            preprocess,
            preprocess_multimodal,
            random_color_jitter,
        )

        record = self.records[sample.record_index]
        frames = [
            np.array(Image.open(frame_path(record, index)).convert("RGB"))
            for index in range(1, sample.history_frames + 1)
        ]
        video = np.stack(frames)
        if self.augmentation and len(video) > 1:
            last = len(video) - 1
            kept = last - random.randint(0, math.ceil(0.1 * last))
            sampled = sorted(random.sample(range(last), max(kept, 0)))
            sampled.append(last)
            sampled = duplicate_with_probability(sampled, 0.03)
            video = random_color_jitter(video[sampled])
        image = self.data_args.image_processor.preprocess(
            video, return_tensors="pt"
        )["pixel_values"]
        prompt = build_prompt(record.instruction, NAVIGATION_IDENTIFIER, DEFAULT_IMAGE_TOKEN)
        conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": format_action_target(sample.actions)},
        ]
        sources = preprocess_multimodal(copy.deepcopy([conversation]), self.data_args)
        tokenized = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt,
            video_or_not=True,
        )
        input_ids = tokenized["input_ids"][0]
        labels = self.build_labels(input_ids)
        result = {"input_ids": input_ids, "labels": labels, "image": image}
        if tokenized.get("prompt") is not None:
            result["prompt"] = tokenized["prompt"]
        return result

    def build_labels(self, input_ids):
        ignore_index = -100
        labels = self._torch.full_like(input_ids, ignore_index)
        separator = self.tokenizer(
            "ASSISTANT:", return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        n, size = len(input_ids), len(separator)
        boundaries = []
        for index in range(n - size + 1):
            if (input_ids[index : index + size] == separator).all():
                boundaries.append(index + size)
        if not boundaries:
            raise ValueError("ASSISTANT separator not found in Uni-NaVid input_ids")
        start = boundaries[-1]
        labels[start:] = input_ids[start:]
        if int((labels != ignore_index).sum()) <= 0:
            raise ValueError("Uni-NaVid sample has no supervised response tokens")
        return labels


def validate_records(
    annotation_path: Path,
    records: Sequence[AnnotationRecord],
    *,
    strict_frames: bool,
    hash_frame_content: bool,
    decode_samples: int,
) -> Mapping[str, Any]:
    if hash_frame_content and not strict_frames:
        raise ValueError("--hash-frame-content requires --strict-frames")
    samples = build_sample_indices(records)
    used_frames = 0
    total_actions = 0
    decoded = 0
    decode_stride = max(len(records) // max(decode_samples, 1), 1)
    tree = hashlib.sha256()
    tree_files = 0
    tree_bytes = 0
    from PIL import Image

    for record_index, record in enumerate(records):
        total_actions += len(record.actions) - 1
        maximum = ((len(record.actions) - 2) // 4) * 4 + 1
        for index in range(1, maximum + 1):
            path = frame_path(record, index)
            if strict_frames and not path.is_file():
                raise FileNotFoundError(path)
            used_frames += 1
        if strict_frames and decoded < decode_samples and record_index % decode_stride == 0:
            with Image.open(frame_path(record, 1)) as image:
                image.convert("RGB").load()
            decoded += 1
        if hash_frame_content:
            rgb_dir = record.video_dir / "rgb"
            for path in sorted(rgb_dir.glob("*.jpg")):
                resolved = path.resolve()
                if not record.allow_external_paths:
                    resolved.relative_to(record.path_anchor)
                relative = resolved.relative_to(record.path_anchor).as_posix()
                size = resolved.stat().st_size
                tree.update(f"{relative}\0{size}\0".encode("utf-8"))
                with resolved.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        tree.update(chunk)
                tree.update(b"\0")
                tree_files += 1
                tree_bytes += size
    return {
        "status": "passed",
        "annotations": len(records),
        "actions": total_actions,
        "window_samples": len(samples),
        "required_history_frames": used_frames,
        "strict_frames": strict_frames,
        "decoded_samples": decoded,
        "annotation_artifact": file_identity(annotation_path),
        "frame_tree": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "digest": tree.hexdigest() if hash_frame_content else None,
            "file_count": tree_files,
            "total_bytes": tree_bytes,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SatNav data for Uni-NaVid")
    parser.add_argument("trajectory_root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--strict-frames", action="store_true")
    parser.add_argument("--hash-frame-content", action="store_true")
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--allow-external-paths", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    annotation_path, records = load_annotation_records(
        args.trajectory_root, allow_external_paths=args.allow_external_paths
    )
    report = dict(
        validate_records(
            annotation_path,
            records,
            strict_frames=args.strict_frames,
            hash_frame_content=args.hash_frame_content,
            decode_samples=args.decode_samples,
        ),
        allow_external_paths=args.allow_external_paths,
    )
    if args.expected_episodes is not None and len(records) != args.expected_episodes:
        raise ValueError(
            f"episode count {len(records)} != expected {args.expected_episodes}"
        )
    if (
        args.expected_samples is not None
        and int(report["window_samples"]) != args.expected_samples
    ):
        raise ValueError(
            f"sample count {report['window_samples']} != expected {args.expected_samples}"
        )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

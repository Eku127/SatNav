"""SatNav trajectory-data adapter for pinned StreamVLN training.

Pure annotation/path/index helpers stay importable without Torch or the
external StreamVLN checkout.  The heavyweight dataset imports are delayed
until ``SatNavActionDataset`` is instantiated by the training entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from baselines.vlm.streamvln.actions import format_action_symbols
from baselines.vlm.streamvln.history import history_indices


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_MEMORY_TOKEN = "<memory>"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class AnnotationRecord:
    """Normalized runtime view of one trajectory annotation."""

    source_root: Path
    video_dir: Path
    instructions: Tuple[str, ...]
    actions: Tuple[int, ...]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class SampleIndex:
    episode_index: int
    instruction_index: int
    start_index: int
    valid_index: int


def _positive_optional_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _env_positive_int(name: str) -> Optional[int]:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return _positive_optional_int(int(value), name)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error


def normalize_actions(actions: Sequence[Any]) -> Tuple[int, ...]:
    """Normalize current SatNav and pre-schema legacy action sequences.

    Current trajectory exports begin with ``-1`` (INIT) and already end in
    ``0`` (STOP).  Older exports may omit INIT and/or the final STOP; those are
    accepted so archived training data remains loadable.
    """

    normalized = [int(action) for action in actions]
    if normalized and normalized[0] == -1:
        normalized = normalized[1:]
    if any(action not in (0, 1, 2, 3) for action in normalized):
        raise ValueError(f"trajectory contains unsupported actions: {normalized}")
    if normalized and normalized[-1] != 0:
        normalized.append(0)
    return tuple(normalized)


def resolve_video_directory(
    trajectory_root: Path,
    video: str,
    *,
    allow_external_video_paths: bool = False,
) -> Path:
    """Resolve both canonical and legacy trajectory/image-root layouts.

    Canonical launchers pass ``trajectory_data`` and annotations store
    ``images/<episode>``.  Some legacy launchers passed
    ``trajectory_data/images`` instead; blindly joining those values produced
    ``images/images/<episode>``.  Candidate resolution handles both layouts
    while rejecting relative traversal.
    """

    root = Path(trajectory_root).expanduser().resolve()
    portable = str(video).replace("\\", "/")
    relative = Path(portable)
    if relative.is_absolute() or PureWindowsPath(portable).is_absolute():
        if not allow_external_video_paths:
            raise ValueError(
                "absolute annotation video paths are disabled; paths must be "
                "anchored to trajectory_root"
            )
        return relative.expanduser().resolve()
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
        if not allow_external_video_paths:
            try:
                resolved.relative_to(anchor)
            except ValueError as error:
                raise ValueError(
                    f"annotation video path escapes trajectory root: {video!r}"
                ) from error
        if resolved not in unique:
            unique.append(resolved)
    for candidate in unique:
        rgb_path = candidate / "rgb"
        if rgb_path.exists() or rgb_path.is_symlink():
            try:
                resolved_rgb = rgb_path.resolve(strict=True)
                resolved_rgb.relative_to(candidate)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(
                    f"annotation RGB directory escapes video directory: {video!r}"
                ) from error
            if resolved_rgb.is_dir():
                return candidate
    return unique[0]


def _annotation_path(root: Path) -> Tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return root, root.parent
    # Archived launchers sometimes configured trajectory_data/images even
    # though annotations.json remained one level above it.
    if (
        root.name == "images"
        and not (root / "annotations.json").is_file()
        and (root.parent / "annotations.json").is_file()
    ):
        return root.parent / "annotations.json", root
    return root / "annotations.json", root


def load_annotation_records(
    trajectory_roots: Iterable[Path],
    *,
    max_episodes: Optional[int] = None,
    allow_external_video_paths: bool = False,
) -> List[AnnotationRecord]:
    """Read and normalize one or more SatNav trajectory exports."""

    max_episodes = _positive_optional_int(max_episodes, "max_episodes")
    records: List[AnnotationRecord] = []
    for configured_root in trajectory_roots:
        annotation_path, data_root = _annotation_path(Path(configured_root))
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{annotation_path} must contain a JSON list")
        for item in payload:
            if max_episodes is not None and len(records) >= max_episodes:
                return records
            if not isinstance(item, Mapping):
                raise ValueError(f"annotation in {annotation_path} is not an object")
            raw_instructions = item.get("instructions", ())
            if isinstance(raw_instructions, str):
                raw_instructions = [raw_instructions]
            if not isinstance(raw_instructions, Sequence) or not raw_instructions:
                raise ValueError(
                    "annotation instructions must be a non-empty list/string"
                )
            instructions = tuple(str(value) for value in raw_instructions)
            actions = normalize_actions(item.get("actions", ()))
            if not actions:
                continue
            video = item.get("video")
            if not isinstance(video, str) or not video:
                raise ValueError("annotation video must be a non-empty string")
            records.append(
                AnnotationRecord(
                    source_root=data_root,
                    video_dir=resolve_video_directory(
                        data_root,
                        video,
                        allow_external_video_paths=allow_external_video_paths,
                    ),
                    instructions=instructions,
                    actions=actions,
                    payload=dict(item),
                )
            )
    return records


def _initial_rotation_count(actions: Sequence[int]) -> int:
    for index, action in enumerate(actions):
        if action == 1:
            return index
    return 0


def build_sample_indices(
    records: Sequence[AnnotationRecord],
    *,
    num_frames: int,
    remove_init_turns: bool = False,
    max_samples: Optional[int] = None,
) -> List[SampleIndex]:
    """Build legacy-compatible fixed-window training sample indices."""

    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    max_samples = _positive_optional_int(max_samples, "max_samples")
    samples: List[SampleIndex] = []
    for episode_index, record in enumerate(records):
        for instruction_index, instruction in enumerate(record.instructions):
            del instruction  # index identity only
            valid_index = (
                _initial_rotation_count(record.actions) if remove_init_turns else 0
            )
            remaining = len(record.actions) - valid_index
            if remaining < 4:
                continue
            for start_index in range(0, remaining, num_frames):
                samples.append(
                    SampleIndex(
                        episode_index=episode_index,
                        instruction_index=instruction_index,
                        start_index=start_index,
                        valid_index=valid_index,
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples


def _require_training_samples(samples: Sequence[SampleIndex]) -> None:
    if not samples:
        raise ValueError(
            "StreamVLN trajectory data produces no training samples; "
            "each selected trajectory needs at least four normalized actions"
        )


def frame_paths(video_dir: Path) -> List[Path]:
    try:
        video_root = Path(video_dir).resolve(strict=True)
        rgb_dir = (video_root / "rgb").resolve(strict=True)
        rgb_dir.relative_to(video_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"RGB frame directory escapes video directory: {video_dir}"
        ) from error
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB frame directory not found: {rgb_dir}")
    frames = []
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(video_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"RGB frame escapes video directory: {path}") from error
        if resolved.is_file():
            frames.append(resolved)
    if not frames:
        raise ValueError(f"No RGB frames found in {rgb_dir}")
    return frames


class SatNavActionDataset:
    """Torch dataset matching StreamVLN's interleaved action supervision."""

    def __init__(self, tokenizer: Any, data_args: Any, task_id: int = 0):
        from PIL import Image
        import numpy as np
        import torch

        from streamvln.dataset.vln_action_dataset import preprocess

        self._Image = Image
        self._np = np
        self._torch = torch
        self._preprocess = preprocess
        self.task_id = int(task_id)
        self.tokenizer = tokenizer
        self.transforms = getattr(data_args, "transform_train", None)
        self.num_frames = int(data_args.num_frames)
        self.num_history = int(data_args.num_history)
        self.num_future_steps = int(data_args.num_future_steps)
        self.remove_init_turns = bool(data_args.remove_init_turns)
        if self.num_history <= 0 or self.num_future_steps <= 0:
            raise ValueError("num_history and num_future_steps must be positive")

        processor = getattr(data_args, "image_processor", None)
        if processor is None:
            from llava.model.multimodal_encoder.siglip_encoder import (
                SigLipImageProcessor,
            )

            processor = SigLipImageProcessor()
        self.image_processor = processor

        roots = [
            Path(value.strip()) for value in str(data_args.video_folder).split(",")
        ]
        max_episodes = _env_positive_int("SATNAV_MAX_EPISODES")
        max_samples = _env_positive_int("SATNAV_MAX_SAMPLES")
        allow_external_video_paths = os.environ.get(
            "SATNAV_ALLOW_EXTERNAL_VIDEO_PATHS", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        self.records = load_annotation_records(
            roots,
            max_episodes=max_episodes,
            allow_external_video_paths=allow_external_video_paths,
        )
        self.samples = build_sample_indices(
            self.records,
            num_frames=self.num_frames,
            remove_init_turns=self.remove_init_turns,
            max_samples=max_samples,
        )
        _require_training_samples(self.samples)

        self.conjunctions = (
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        )
        prompt = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 10 meters, or STOP."
        )
        self.conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ""},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def task(self) -> int:
        return self.task_id

    def _prepare_conversation(
        self, conversation: List[Dict[str, str]], actions: Sequence[int]
    ) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        for start in range(0, len(actions), self.num_future_steps):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[start : start + self.num_future_steps]
            source[0]["value"] = (
                source[0]["value"] + f" {prompt}." if start == 0 else f"{prompt}."
            )
            source[1]["value"] = format_action_symbols(step_actions)
            sources.extend(source)
        return sources

    def __getitem__(self, index: int):
        sample = self.samples[index]
        record = self.records[sample.episode_index]
        effective_actions = record.actions[sample.valid_index :]
        start = sample.start_index
        end = min(start + self.num_frames, len(effective_actions))
        if start >= end:
            raise IndexError(f"empty StreamVLN sample at index {index}")
        actions = effective_actions[start:end]
        time_ids = self._np.arange(start, end, dtype=self._np.int64)

        frames = frame_paths(record.video_dir)
        frame_start = start + sample.valid_index
        frame_end = end + sample.valid_index
        sample_ids = range(frame_start, frame_end, self.num_future_steps)
        history_ids = (
            tuple(
                value + sample.valid_index
                for value in history_indices(
                    start,
                    num_history=self.num_history,
                    future_stride=self.num_future_steps,
                )
            )
            if start != 0
            else ()
        )
        required_ids = list(history_ids) + list(sample_ids)
        if required_ids and max(required_ids) >= len(frames):
            raise ValueError(
                f"trajectory {record.payload.get('id')} has {len(frames)} frames; "
                f"sample requires frame index {max(required_ids)}"
            )

        images = []
        for frame_index in required_ids:
            with self._Image.open(frames[frame_index]) as handle:
                image = handle.convert("RGB")
                if self.transforms is not None:
                    image = self.transforms(image)
                processed = self.image_processor.preprocess(
                    images=image, return_tensors="pt"
                )["pixel_values"][0]
            images.append(processed)
        image_tensor = self._torch.stack(images)

        sources = copy.deepcopy(self.conversation)
        if start != 0:
            sources[0]["value"] += (
                f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
            )
        sources[0]["value"] = sources[0]["value"].replace(
            "<instruction>.", record.instructions[sample.instruction_index]
        )
        interleaved = self._prepare_conversation(sources, actions)
        tokenized = self._preprocess([interleaved], self.tokenizer, has_image=True)
        return (
            tokenized["input_ids"][0],
            tokenized["labels"][0],
            image_tensor,
            self._torch.tensor(time_ids),
            self.task,
        )


def _pad_tensors(torch_module: Any, tensors: Sequence[Any], lengths: Sequence[int]):
    max_length = max(lengths)
    output = torch_module.zeros(
        len(tensors),
        max_length,
        *tensors[0].shape[1:],
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for index, (tensor, length) in enumerate(zip(tensors, lengths)):
        output[index, :length] = tensor
    return output


def satnav_collate_fn(batch: Sequence[Any], tokenizer: Any) -> Mapping[str, Any]:
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids, labels, images, time_ids, task_types = zip(*batch)
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )[:, : tokenizer.model_max_length]
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)[
        :, : tokenizer.model_max_length
    ]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    image_lengths = np.asarray([image.size(0) for image in images])
    time_ids = pad_sequence(time_ids, batch_first=True, padding_value=-1)
    return {
        "images": _pad_tensors(torch, images, image_lengths),
        "time_ids": time_ids,
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "labels": labels,
        "task_type": task_types,
    }


def make_satnav_collate_fn(tokenizer: Any):
    from functools import partial

    return partial(satnav_collate_fn, tokenizer=tokenizer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize StreamVLN SatNav trajectory annotations"
    )
    parser.add_argument("trajectory_root", type=Path)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--strict-frames", action="store_true")
    parser.add_argument(
        "--allow-external-video-paths",
        action="store_true",
        help="explicit legacy opt-in for absolute/out-of-root annotation video paths",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_annotation_records(
        [args.trajectory_root],
        max_episodes=args.max_episodes,
        allow_external_video_paths=args.allow_external_video_paths,
    )
    samples = build_sample_indices(
        records,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
    )
    _require_training_samples(samples)
    if args.strict_frames:
        for record in records:
            frames = frame_paths(record.video_dir)
            if len(frames) < len(record.actions):
                raise ValueError(
                    f"trajectory {record.payload.get('id')} has fewer frames than actions"
                )
    print(
        json.dumps({"episodes": len(records), "samples": len(samples)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NaVILA prompt, natural-language action parsing, and frame sampling."""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple


ACTION_NAMES: Mapping[int, str] = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
ACTION_TEXT: Mapping[str, Mapping[int, str]] = {
    "compact": ACTION_NAMES,
    "sentence": {
        0: "The next action is stop.",
        1: "The next action is move forward 10 meters.",
        2: "The next action is turn left 15 degree.",
        3: "The next action is turn right 15 degree.",
    },
}
ACTION_PATTERNS = (
    (0, re.compile(r"\bstop\b", re.IGNORECASE)),
    (1, re.compile(r"\b(?:move forward|forward)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(?:turn left|left)\b", re.IGNORECASE)),
    (3, re.compile(r"\b(?:turn right|right)\b", re.IGNORECASE)),
)
DISTANCE_CHOICES_METERS = (10, 20, 30)
TURN_CHOICES_DEGREES = (15, 30, 45)

PROMPT_SENTENCE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a "
    "video of historical observations {history_tokens}, and current observation "
    '<image>\n. Your assigned task is: "{instruction}" Analyze this series of '
    "images to decide your next action, which could be turning left or right by a "
    "specific degree, moving forward a certain distance, or stop if the task is completed."
)
PROMPT_COMPACT = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a "
    "video of historical observations {history_tokens}, and current observation "
    '<image>\n. Your assigned task is: "{instruction}" Analyze this series of '
    "images and reply with exactly one word for the next action: stop, forward, left, or right."
)


def normalize_action_format(value: Optional[str] = None) -> str:
    normalized = str(value or "compact").strip().lower().replace("-", "_")
    aliases = {
        "compact": "compact",
        "default": "compact",
        "token": "compact",
        "word": "compact",
        "sentence": "sentence",
        "natural": "sentence",
        "legacy": "sentence",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported NaVILA action format: {value!r}")
    return aliases[normalized]


def action_text(action: int, action_format: Optional[str] = None) -> str:
    try:
        return ACTION_TEXT[normalize_action_format(action_format)][int(action)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported SatNav action: {action!r}") from error


def normalize_instruction(text: str) -> str:
    """Match the normalization used by the pinned NaVILA adapter."""

    normalized = str(text).replace("\r\n", " ").replace("\n", " ").strip()
    normalized = re.sub(r"\s+\.", ".", normalized)
    normalized = re.sub(r"\s+", " ", normalized).capitalize()
    return re.sub(r"(?<=\.\s)([a-z])", lambda match: match.group().upper(), normalized)


def episode_instruction(episode: Any) -> str:
    instruction = getattr(episode, "instruction", "")
    if isinstance(instruction, Mapping):
        value = instruction.get("text", instruction.get("instruction_text", ""))
    elif hasattr(instruction, "text"):
        value = instruction.text
    elif hasattr(instruction, "instruction_text"):
        value = instruction.instruction_text
    else:
        value = instruction
    return normalize_instruction(str(value))


def build_prompt(
    instruction: str,
    history_count: int,
    action_format: Optional[str] = None,
) -> str:
    template = (
        PROMPT_COMPACT
        if normalize_action_format(action_format) == "compact"
        else PROMPT_SENTENCE
    )
    return template.format(
        history_tokens="<image>\n" * max(int(history_count), 0),
        instruction=normalize_instruction(instruction),
    )


def parse_action_text(text: str) -> Optional[int]:
    if not isinstance(text, str):
        raise TypeError(f"model output must be str, got {type(text).__name__}")
    for action, pattern in ACTION_PATTERNS:
        if pattern.search(text):
            return action
    return None


def _snap(value: int, base: int, choices: Sequence[int]) -> int:
    if value <= 0:
        return base
    if value % base:
        return min(choices, key=lambda candidate: abs(candidate - value))
    return value


def parse_action_queue(text: str) -> Tuple[int, List[int]]:
    """Parse one generated action and any primitive continuation queue."""

    action = parse_action_text(text)
    if action is None:
        action = 1
    queue: List[int] = []
    if action == 1:
        match = re.search(r"move forward (\d+) meters?", text, re.IGNORECASE)
        if match:
            distance = _snap(int(match.group(1)), 10, DISTANCE_CHOICES_METERS)
            queue.extend([1] * max(distance // 10 - 1, 0))
    elif action in (2, 3):
        direction = "left" if action == 2 else "right"
        match = re.search(rf"turn {direction} (\d+) degrees?", text, re.IGNORECASE)
        degrees = _snap(int(match.group(1)) if match else 15, 15, TURN_CHOICES_DEGREES)
        queue.extend([action] * max(degrees // 15 - 1, 0))
    return action, queue


def sample_and_pad_images(
    images: Sequence[Any], num_frames: int = 8, width: int = 512, height: int = 512
) -> List[Any]:
    """Retain the pinned evaluator's history sampling and black-frame padding."""

    import numpy as np
    from PIL import Image

    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    frames = list(images)
    if not frames:
        raise ValueError("at least one current observation is required")
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    latest = frames[-1]
    indices = np.linspace(
        0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int
    )
    return [frames[int(index)] for index in indices] + [latest]

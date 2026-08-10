"""Dependency-light Uni-NaVid prompt, target, and action behavior."""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Sequence


ACTION_NAMES: Mapping[int, str] = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
ACTION_TEXT_MAP = {value: key for key, value in ACTION_NAMES.items()}
ACTIONS_PER_QUERY = 4

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given {nav_id} {img_token}. "
    "Your assigned task is: '{instruction}'. "
    "Analyze this series of images to determine your next four actions. "
    "The predicted action should be one of the following: "
    "forward, left, right, or stop."
)


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
    return str(value)


def build_prompt(instruction: str, nav_id: str, image_token: str) -> str:
    return PROMPT_TEMPLATE.format(
        nav_id=nav_id,
        img_token=image_token,
        instruction=str(instruction),
    )


def format_action_target(actions: Sequence[int]) -> str:
    if len(actions) != ACTIONS_PER_QUERY:
        raise ValueError(f"expected four actions, got {len(actions)}")
    try:
        return " ".join(
            f"{index}. {ACTION_NAMES[int(action)]}"
            for index, action in enumerate(actions, start=1)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported action sequence: {actions!r}") from error


def parse_action_chunk(text: str) -> List[int]:
    if not isinstance(text, str):
        raise TypeError(f"model output must be str, got {type(text).__name__}")
    words = re.findall(r"\b(forward|left|right|stop)\b", text.lower())
    return [ACTION_TEXT_MAP[word] for word in words[:ACTIONS_PER_QUERY]]


def action_chunk_or_stop(text: str) -> tuple[List[int], bool]:
    parsed = parse_action_chunk(text)
    return (parsed, False) if parsed else ([0], True)

"""Frozen Llama-2 prompt contract shared by OpenFly train and evaluation."""

from __future__ import annotations

import os
from typing import Optional, Sequence


SYS_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
ASSISTANT_BOUNDARY = "[/INST]"
DEFAULT_ACTION_HISTORY_LIMIT = 16
_ACTION_NAME_BY_ID = {0: "stop", 1: "forward", 2: "left", 3: "right"}


def resolve_history_limit(value: Optional[int] = None) -> int:
    if value is not None:
        parsed = int(value)
    else:
        configured = os.environ.get("OPENFLY_ACTION_HISTORY_LIMIT", "").strip()
        parsed = int(configured) if configured else DEFAULT_ACTION_HISTORY_LIMIT
    if parsed < 0:
        raise ValueError("OpenFly action history limit must be non-negative")
    return parsed


def normalize_instruction(text: str) -> str:
    return " ".join(str(text).replace("\r\n", " ").replace("\n", " ").split()).lower()


def normalize_action_history(action_history: Optional[Sequence[int]]) -> list[str]:
    names: list[str] = []
    for raw in action_history or ():
        try:
            action = int(raw)
        except (TypeError, ValueError):
            continue
        if action == -1:
            continue
        name = _ACTION_NAME_BY_ID.get(action)
        if name is not None:
            names.append(name)
    return names


def format_action_history(
    action_history: Optional[Sequence[int]], history_limit: Optional[int] = None
) -> str:
    names = normalize_action_history(action_history)
    if not names:
        return "Past actions: none."
    limit = resolve_history_limit(history_limit)
    truncated = limit > 0 and len(names) > limit
    kept = names[-limit:] if limit > 0 else names
    if truncated:
        prefix = f"Past actions (last {len(kept)} of {len(names)})"
    else:
        prefix = f"Past actions ({len(names)} so far)"
    return f"{prefix}: {', '.join(kept)}."


def _system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class LLaMa2ChatPromptBuilder:
    """Minimal prompt builder retained from the frozen SwiftVLN adapter."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt = _system_prompt(SYS_PROMPT if system_prompt is None else system_prompt)
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        expected = "human" if self.turn_count % 2 == 0 else "gpt"
        if role != expected:
            raise ValueError(
                f"Unexpected role={role!r} at turn={self.turn_count}, expected={expected!r}"
            )
        cleaned = str(message).replace("<image>", "").strip()
        if self.turn_count == 0:
            wrapped = f"[INST] {self.system_prompt}{cleaned} [/INST] "
        elif role == "human":
            wrapped = f"[INST] {cleaned} [/INST] "
        else:
            wrapped = f"{cleaned if cleaned else ' '}</s>"
        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        return self.prompt.removeprefix("<s>").rstrip()


def _instruction_message(
    instruction: str,
    action_history: Optional[Sequence[int]],
    history_limit: Optional[int],
) -> str:
    question = f"What action should the robot take to {normalize_instruction(instruction)}?"
    return f"{question} {format_action_history(action_history, history_limit)}"


def build_openfly_prompt(
    instruction: str,
    action_history: Optional[Sequence[int]] = None,
    history_limit: Optional[int] = None,
) -> str:
    builder = LLaMa2ChatPromptBuilder()
    builder.add_turn(
        "human", _instruction_message(instruction, action_history, history_limit)
    )
    return builder.get_prompt()


def build_openfly_prompt_with_answer(
    instruction: str,
    answer: str,
    action_history: Optional[Sequence[int]] = None,
    history_limit: Optional[int] = None,
) -> str:
    builder = LLaMa2ChatPromptBuilder()
    builder.add_turn(
        "human", _instruction_message(instruction, action_history, history_limit)
    )
    builder.add_turn("gpt", str(answer))
    return builder.get_prompt()

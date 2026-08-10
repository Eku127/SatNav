"""OpenFly action formats and deterministic SatNav parsing."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


COMPACT = "compact"
ORIGINAL = "original"
AUTO = "auto"
SUPPORTED_ACTION_FORMATS = (COMPACT, ORIGINAL)

ACTION_TO_NAME = {0: "stop", 1: "forward", 2: "left", 3: "right"}
NAME_TO_ACTION = {name: action for action, name in ACTION_TO_NAME.items()}
ACTION_PATTERNS = {
    action: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for action, name in ACTION_TO_NAME.items()
}

ORIGINAL_ACTION_DIM = 8
ORIGINAL_UNNORM_KEY = "satnav_original"
ORIGINAL_ACTIVE_MASK = np.array(
    [True, True, True, True, False, False, False, False], dtype=bool
)
ORIGINAL_Q01 = np.zeros(ORIGINAL_ACTION_DIM, dtype=np.float32)
ORIGINAL_Q99 = np.array(
    [1.0, 10.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)
ORIGINAL_ACTION_VECTORS = {
    0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    3: np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}


def resolve_action_format(requested: str, fallback: Optional[str] = None) -> str:
    """Resolve ``auto`` without accepting an unknown checkpoint declaration."""

    value = str(requested).strip().lower()
    if value == AUTO:
        value = str(fallback or "").strip().lower()
        if not value:
            raise ValueError("action_format=auto requires checkpoint metadata")
    if value not in SUPPORTED_ACTION_FORMATS:
        raise ValueError(f"Unsupported OpenFly action format: {value!r}")
    return value


def action_to_text(action: int) -> str:
    try:
        return ACTION_TO_NAME[int(action)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported SatNav action: {action!r}") from error


def parse_action_text(text: str) -> Optional[int]:
    """Return the first canonical action by stable action-id priority."""

    rendered = str(text)
    for action in sorted(ACTION_PATTERNS):
        if ACTION_PATTERNS[action].search(rendered):
            return action
    return None


def get_original_action_vector(action: int) -> np.ndarray:
    try:
        return ORIGINAL_ACTION_VECTORS[int(action)].copy()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Unsupported SatNav action for original OpenFly format: {action!r}"
        ) from error


def normalize_original_action(raw_action: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape != (ORIGINAL_ACTION_DIM,):
        raise ValueError(f"OpenFly original action must have shape ({ORIGINAL_ACTION_DIM},)")
    normalized = np.zeros_like(raw, dtype=np.float32)
    active = ORIGINAL_ACTIVE_MASK
    normalized[active] = (
        2.0
        * (raw[active] - ORIGINAL_Q01[active])
        / (ORIGINAL_Q99[active] - ORIGINAL_Q01[active])
        - 1.0
    )
    return normalized


def get_normalized_original_action_vector(action: int) -> np.ndarray:
    return normalize_original_action(get_original_action_vector(action))


def get_original_norm_stats(
    unnorm_key: str = ORIGINAL_UNNORM_KEY,
    num_transitions: Optional[int] = None,
) -> dict:
    stats = {
        str(unnorm_key): {
            "action": {
                "q01": ORIGINAL_Q01.tolist(),
                "q99": ORIGINAL_Q99.tolist(),
                "mask": ORIGINAL_ACTIVE_MASK.astype(bool).tolist(),
            }
        }
    }
    if num_transitions is not None:
        stats[str(unnorm_key)]["num_transitions"] = int(num_transitions)
    return stats


def get_original_action_templates() -> dict[str, list[float]]:
    return {
        str(action): vector.tolist()
        for action, vector in ORIGINAL_ACTION_VECTORS.items()
    }


def convert_original_action_vector_to_action(
    action_vector: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    raw = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    if raw.shape != (ORIGINAL_ACTION_DIM,) or not np.isfinite(raw).all():
        raise ValueError("model returned an invalid OpenFly original action vector")
    rounded = np.round(raw).astype(np.int32)
    for action, template in ORIGINAL_ACTION_VECTORS.items():
        if np.array_equal(rounded, template.astype(np.int32)):
            return action, rounded, 0.0

    active = np.where(ORIGINAL_ACTIVE_MASK)[0]
    distances = {
        action: float(np.linalg.norm(raw[active] - template[active]))
        for action, template in ORIGINAL_ACTION_VECTORS.items()
    }
    action = min(distances, key=lambda value: (distances[value], value))
    return int(action), rounded, distances[action]


def validate_tokenizer_model_contract(model, tokenizer, action_format: str) -> None:
    resolved = resolve_action_format(action_format)
    tokenizer_vocab = int(tokenizer.vocab_size)
    model_vocab = int(model.config.text_config.vocab_size)
    padded = int(getattr(model.config, "pad_to_multiple_of", 0))
    if padded < 0 or model_vocab <= 0 or tokenizer_vocab <= 0:
        raise ValueError("OpenFly tokenizer/model vocabulary metadata is invalid")
    if resolved == ORIGINAL:
        if int(getattr(model.config, "n_action_bins", -1)) != 256:
            raise ValueError("OpenFly original action tokenizer requires n_action_bins=256")
        if model_vocab - padded != tokenizer_vocab:
            raise ValueError(
                "OpenFly original action vocabulary requires "
                "model_vocab - pad_to_multiple_of == tokenizer_vocab"
            )
    elif model_vocab not in {tokenizer_vocab, tokenizer_vocab + padded}:
        raise ValueError("OpenFly compact tokenizer/model vocabulary sizes are incompatible")

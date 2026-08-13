"""JSON helpers for evaluation results."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


def to_jsonable(value: Any) -> Any:
    """Convert common scalar/container objects without importing ML frameworks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float cannot be serialized: {value!r}")
        return value
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    # Public domain objects such as VLNEpisode deliberately exclude runtime
    # fields (notably absolute scene_path) from their default to_dict output.
    # Respect that contract before dataclasses.asdict recursively exposes every
    # field.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict(include_runtime=False))
        except TypeError:
            return to_jsonable(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    # NumPy and scalar tensor objects expose item(); arrays expose tolist().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass

    # AgentState-like values are useful diagnostics and have a stable shape.
    if hasattr(value, "position") and hasattr(value, "rotation"):
        return {
            "position": to_jsonable(value.position),
            "rotation": to_jsonable(value.rotation),
        }

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a compact, deterministic JSON value."""

    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace ``path`` with formatted JSON and fsync the file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            to_jsonable(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def read_json_object(path: Path) -> Dict[str, Any]:
    """Read a JSON object and reject other top-level shapes."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value

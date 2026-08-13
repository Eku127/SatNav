"""Content-bound manifests used only by training/checkpoint workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from satnav.evaluation._json import canonical_json_bytes, read_json_object, to_jsonable


class ManifestError(RuntimeError):
    """Base class for malformed training manifests."""


class ManifestMismatchError(ManifestError):
    """Raised when immutable training facts changed during resume."""


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def manifest_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = to_jsonable(payload)
    return {"digest": payload_digest(normalized), "payload": normalized}


def _write_once(path: Path, envelope: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            envelope,
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
        try:
            os.link(str(temporary_path), str(path))
            return True
        except FileExistsError:
            return False
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        envelope = read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read manifest {path}: {error}") from error
    payload = envelope.get("payload")
    stored_digest = envelope.get("digest")
    if not isinstance(payload, dict) or not isinstance(stored_digest, str):
        raise ManifestError(f"Malformed manifest envelope: {path}")
    actual_digest = payload_digest(payload)
    if actual_digest != stored_digest:
        raise ManifestError(
            f"Manifest digest is invalid for {path}: "
            f"stored={stored_digest}, actual={actual_digest}"
        )
    return envelope


def ensure_manifest(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = manifest_envelope(payload)
    if _write_once(Path(path), expected):
        return expected
    existing = read_manifest(Path(path))
    if existing["digest"] != expected["digest"]:
        raise ManifestMismatchError(
            f"Refusing to resume with a different manifest at {path}: "
            f"existing={existing['digest']}, requested={expected['digest']}"
        )
    return existing


__all__ = [
    "ManifestError",
    "ManifestMismatchError",
    "ensure_manifest",
    "manifest_envelope",
    "payload_digest",
    "read_manifest",
]

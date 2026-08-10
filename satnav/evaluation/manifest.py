"""Versioned benchmark and run manifests for reproducible evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from satnav.evaluation._json import canonical_json_bytes, read_json_object, to_jsonable


MANIFEST_SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """Base class for malformed or incompatible manifests."""


class ManifestMismatchError(ManifestError):
    """Raised before resume when immutable run facts changed."""


def payload_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a manifest payload."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class BenchmarkManifest:
    """Small benchmark contract containing only evaluation-critical facts."""

    benchmark_id: str
    dataset_version: str
    split: str
    kind: str = "smoke"
    episode_count: Optional[int] = None
    episode_digest: Optional[str] = None
    dataset_digest: Optional[str] = None
    action_space: Tuple[str, ...] = ()
    forward_step_size: Optional[float] = None
    turn_angle: Optional[float] = None
    observation: Mapping[str, Any] = field(default_factory=dict)
    max_episode_steps: Optional[int] = None
    success_threshold: Any = None
    required_metrics: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkManifest":
        """Parse either the concise config shape or a persisted payload."""

        if value.get("manifest_type") == "benchmark":
            dataset = value.get("dataset", {})
            return cls(
                benchmark_id=str(value["benchmark_id"]),
                dataset_version=str(dataset["version"]),
                split=str(dataset["split"]),
                kind=str(value.get("kind", "smoke")),
                episode_count=dataset.get("episode_count"),
                episode_digest=dataset.get("episode_digest"),
                dataset_digest=dataset.get("artifact_digest"),
                action_space=tuple(value.get("action_space", ())),
                forward_step_size=value.get("forward_step_size"),
                turn_angle=value.get("turn_angle"),
                observation=value.get("observation", {}),
                max_episode_steps=value.get("max_episode_steps"),
                success_threshold=value.get("success_threshold"),
                required_metrics=tuple(value.get("required_metrics", ())),
                metadata=value.get("metadata", {}),
            )
        return cls(
            benchmark_id=str(value["benchmark_id"]),
            dataset_version=str(value["dataset_version"]),
            split=str(value["split"]),
            kind=str(value.get("kind", "smoke")),
            episode_count=value.get("episode_count"),
            episode_digest=value.get("episode_digest"),
            dataset_digest=value.get("dataset_digest"),
            action_space=tuple(value.get("action_space", ())),
            forward_step_size=value.get("forward_step_size"),
            turn_angle=value.get("turn_angle"),
            observation=value.get("observation", {}),
            max_episode_steps=value.get("max_episode_steps"),
            success_threshold=value.get("success_threshold"),
            required_metrics=tuple(value.get("required_metrics", ())),
            metadata=value.get("metadata", {}),
        )

    def resolved(
        self,
        *,
        split: str,
        episode_count: int,
        episode_digest: str,
        max_episode_steps: int,
    ) -> "BenchmarkManifest":
        """Fill computed fields while rejecting contradictory declarations."""

        if self.split and self.split != split:
            raise ManifestMismatchError(
                f"Benchmark split {self.split!r} does not match run split {split!r}"
            )
        if self.episode_count is not None and int(self.episode_count) != episode_count:
            raise ManifestMismatchError(
                "Benchmark episode_count does not match the loaded dataset: "
                f"{self.episode_count} != {episode_count}"
            )
        if self.episode_digest is not None and self.episode_digest != episode_digest:
            raise ManifestMismatchError(
                "Benchmark episode_digest does not match the loaded dataset"
            )
        if (
            self.max_episode_steps is not None
            and int(self.max_episode_steps) != max_episode_steps
        ):
            raise ManifestMismatchError(
                "Benchmark max_episode_steps does not match the environment: "
                f"{self.max_episode_steps} != {max_episode_steps}"
            )
        return replace(
            self,
            split=split,
            episode_count=episode_count,
            episode_digest=episode_digest,
            max_episode_steps=max_episode_steps,
        )

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "manifest_type": "benchmark",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "dataset": {
                "version": self.dataset_version,
                "split": self.split,
                "episode_count": self.episode_count,
                "episode_digest": self.episode_digest,
                "artifact_digest": self.dataset_digest,
            },
            "action_space": list(self.action_space),
            "forward_step_size": self.forward_step_size,
            "turn_angle": self.turn_angle,
            "observation": self.observation,
            "max_episode_steps": self.max_episode_steps,
            "success_threshold": self.success_threshold,
            "required_metrics": list(self.required_metrics),
            "kind": self.kind,
            "metadata": self.metadata,
        }


def build_run_manifest(
    *,
    benchmark_digest: str,
    policy_id: str,
    policy_metadata: Mapping[str, Any],
    split: str,
    offset: int,
    limit: Optional[int],
    selected_keys: Sequence[str],
    selected_digest: str,
    world_size: int,
    base_seed: int,
    run_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the immutable, rank-independent run payload."""

    metadata = to_jsonable(run_metadata)
    return {
        "manifest_type": "run",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_digest": benchmark_digest,
        "policy": {
            "id": str(policy_id),
            "metadata": to_jsonable(policy_metadata),
        },
        "selection": {
            "split": split,
            "offset": int(offset),
            "limit": limit,
            "episode_count": len(selected_keys),
            "episode_digest": selected_digest,
            "episode_keys": list(selected_keys),
        },
        "sharding": {"method": "stride", "world_size": int(world_size)},
        "seed": {"base_seed": int(base_seed), "method": "sha256-key-v1"},
        "configuration": {
            "digest": hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
            "values": metadata,
        },
    }


def manifest_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Wrap a payload with its self-verifying digest."""

    normalized = to_jsonable(payload)
    return {"digest": payload_digest(normalized), "payload": normalized}


def _write_once(path: Path, envelope: Mapping[str, Any]) -> bool:
    """Atomically create ``path`` without replacing another rank's file."""

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
    """Read an envelope and verify its digest."""

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
    """Create a manifest once, or verify an identical existing manifest.

    This is safe for ranks starting concurrently: each writes a complete temp
    file and only one hard-links it into the shared output directory.
    """

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


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    """Load a concise tracked JSON benchmark specification."""

    try:
        value = read_json_object(Path(path))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read benchmark config {path}: {error}") from error
    try:
        return BenchmarkManifest.from_mapping(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"Invalid benchmark config {path}: {error}") from error

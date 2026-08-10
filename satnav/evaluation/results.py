"""Rank-local JSONL persistence, resume, validation, and aggregation."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from satnav.evaluation._json import (
    atomic_write_json,
    canonical_json_bytes,
    read_json_object,
    to_jsonable,
)
from satnav.evaluation.episodes import episode_keys_digest
from satnav.evaluation.manifest import ManifestError, read_manifest


RESULT_SCHEMA_VERSION = 1


class ResultError(RuntimeError):
    """Base class for evaluation result persistence failures."""


class ResumeRequiredError(ResultError):
    """Raised rather than silently appending to a prior non-resume run."""


class ResultValidationError(ResultError):
    """Raised when expected-count, duplicate, missing, or rank checks fail."""

    def __init__(self, message: str, summary: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.summary = summary


def rank_directory(output_dir: Path, rank: int) -> Path:
    return Path(output_dir) / f"rank_{int(rank):05d}"


class RankResultStore:
    """Append-only result store owned by exactly one rank."""

    def __init__(self, output_dir: Path, rank: int, run_manifest_digest: str):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.run_manifest_digest = str(run_manifest_digest)
        self.directory = rank_directory(self.output_dir, self.rank)
        self.records_path = self.directory / "episodes.jsonl"
        self.done_path = self.directory / "done.json"

    def prepare(self, *, resume: bool) -> List[Mapping[str, Any]]:
        """Create the rank directory and load resumable records."""

        self.directory.mkdir(parents=True, exist_ok=True)
        records_exist = self.records_path.exists() and self.records_path.stat().st_size > 0
        done_exists = self.done_path.exists()
        if (records_exist or done_exists) and not resume:
            raise ResumeRequiredError(
                f"Rank {self.rank} already has output in {self.directory}; "
                "set resume=True or choose a fresh output directory"
            )
        if done_exists:
            self.load_done()
        return self.load_records(recover_truncated_tail=resume)

    def _validate_record(self, record: Mapping[str, Any], line_number: int) -> None:
        if record.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ResultError(
                f"Unsupported result schema at {self.records_path}:{line_number}"
            )
        if record.get("run_manifest_digest") != self.run_manifest_digest:
            raise ResultError(
                f"Run manifest digest mismatch at {self.records_path}:{line_number}"
            )
        if int(record.get("rank", -1)) != self.rank:
            raise ResultError(
                f"Rank mismatch at {self.records_path}:{line_number}"
            )
        if not isinstance(record.get("episode_key"), str):
            raise ResultError(
                f"Missing episode_key at {self.records_path}:{line_number}"
            )
        if record.get("status") not in ("ok", "error"):
            raise ResultError(
                f"Invalid result status at {self.records_path}:{line_number}"
            )

    def load_records(
        self, *, recover_truncated_tail: bool = False
    ) -> List[Mapping[str, Any]]:
        """Load records and optionally remove a crash-truncated final line."""

        if not self.records_path.exists():
            return []
        raw = self.records_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        records: List[Mapping[str, Any]] = []
        valid_bytes = 0
        for index, line in enumerate(lines, start=1):
            complete_line = line.endswith(b"\n")
            try:
                decoded = json.loads(line.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("result record is not a JSON object")
                self._validate_record(decoded, index)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, ResultError):
                is_tail = index == len(lines) and not complete_line
                if recover_truncated_tail and is_tail:
                    with self.records_path.open("r+b") as handle:
                        handle.truncate(valid_bytes)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                raise ResultError(
                    f"Invalid JSONL record at {self.records_path}:{index}"
                )
            records.append(decoded)
            valid_bytes += len(line)
            if not complete_line:
                # A valid crash-tail record is retained and terminated before a
                # future append, preventing two JSON objects from concatenating.
                with self.records_path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return records

    def append(self, record: Mapping[str, Any]) -> None:
        """Durably append exactly one compact JSON object."""

        normalized = to_jsonable(record)
        if not isinstance(normalized, dict):
            raise ResultError("Result record must be a mapping")
        self._validate_record(normalized, 0)
        encoded = canonical_json_bytes(normalized) + b"\n"
        descriptor = os.open(
            str(self.records_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load_done(self) -> Optional[Mapping[str, Any]]:
        if not self.done_path.exists():
            return None
        try:
            done = read_json_object(self.done_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ResultError(f"Cannot read done marker {self.done_path}: {error}") from error
        if done.get("run_manifest_digest") != self.run_manifest_digest:
            raise ResultError(f"Done marker manifest mismatch: {self.done_path}")
        if int(done.get("rank", -1)) != self.rank:
            raise ResultError(f"Done marker rank mismatch: {self.done_path}")
        return done

    def mark_done(
        self,
        *,
        expected_keys: Sequence[str],
        records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Validate the local shard and atomically publish its done marker."""

        keys = [str(record["episode_key"]) for record in records]
        counts = Counter(keys)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        expected = set(expected_keys)
        observed = set(keys)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if duplicates or missing or unexpected or len(records) != len(expected_keys):
            raise ResultValidationError(
                f"Rank {self.rank} cannot be marked done: "
                f"records={len(records)}, expected={len(expected_keys)}, "
                f"duplicates={len(duplicates)}, missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        error_count = sum(record.get("status") == "error" for record in records)
        done = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_manifest_digest": self.run_manifest_digest,
            "rank": self.rank,
            "status": "completed_with_errors" if error_count else "complete",
            "expected_count": len(expected_keys),
            "record_count": len(records),
            "error_count": error_count,
            "episode_digest": episode_keys_digest(expected_keys),
        }
        atomic_write_json(self.done_path, done)
        return done


def _numeric_metrics(records: Iterable[Mapping[str, Any]]) -> Mapping[str, float]:
    values: Dict[str, List[float]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for name, value in metrics.items():
            if isinstance(value, bool):
                numeric = float(value)
            elif isinstance(value, (int, float)):
                numeric = float(value)
            else:
                continue
            if math.isfinite(numeric):
                values.setdefault(str(name), []).append(numeric)
    return {
        name: sum(metric_values) / len(metric_values)
        for name, metric_values in sorted(values.items())
        if metric_values
    }


def aggregate_run(
    output_dir: Path,
    *,
    strict: bool = True,
    require_done: bool = True,
    fail_on_episode_error: bool = False,
) -> Mapping[str, Any]:
    """Deduplicate and validate every rank against the run manifest.

    ``summary.json`` is written even for structurally invalid runs so automated
    jobs have a machine-readable explanation before ``ResultValidationError``
    is raised.
    """

    output_dir = Path(output_dir)
    run_envelope = read_manifest(output_dir / "run_manifest.json")
    benchmark_envelope = read_manifest(output_dir / "benchmark_manifest.json")
    run_digest = str(run_envelope["digest"])
    run_payload = run_envelope["payload"]
    benchmark_payload = benchmark_envelope["payload"]
    if run_payload.get("benchmark_digest") != benchmark_envelope["digest"]:
        raise ManifestError("Run manifest references a different benchmark manifest")

    selection = run_payload.get("selection", {})
    expected_keys = [str(key) for key in selection.get("episode_keys", [])]
    expected_count = int(selection.get("episode_count", -1))
    if expected_count != len(expected_keys):
        raise ManifestError("Run manifest episode_count does not match episode_keys")
    if episode_keys_digest(expected_keys) != selection.get("episode_digest"):
        raise ManifestError("Run manifest episode digest is invalid")

    world_size = int(run_payload.get("sharding", {}).get("world_size", 0))
    if world_size <= 0:
        raise ManifestError("Run manifest has invalid world_size")
    expected_rank = {key: index % world_size for index, key in enumerate(expected_keys)}

    all_records: List[Mapping[str, Any]] = []
    missing_done_ranks: List[int] = []
    invalid_done_ranks: List[int] = []
    for rank in range(world_size):
        store = RankResultStore(output_dir, rank, run_digest)
        records = store.load_records(recover_truncated_tail=False)
        all_records.extend(records)
        done = store.load_done()
        rank_expected_keys = expected_keys[rank::world_size]
        rank_error_count = sum(
            record.get("status") == "error" for record in records
        )
        if done is None:
            if require_done:
                missing_done_ranks.append(rank)
        elif (
            int(done.get("expected_count", -1)) != len(rank_expected_keys)
            or int(done.get("record_count", -1)) != len(records)
            or int(done.get("error_count", -1)) != rank_error_count
            or done.get("episode_digest") != episode_keys_digest(rank_expected_keys)
            or done.get("status")
            != ("completed_with_errors" if rank_error_count else "complete")
        ):
            invalid_done_ranks.append(rank)

    counts = Counter(str(record["episode_key"]) for record in all_records)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    records_by_key: Dict[str, Mapping[str, Any]] = {}
    for record in all_records:
        records_by_key.setdefault(str(record["episode_key"]), record)
    observed_keys = set(records_by_key)
    expected_key_set = set(expected_keys)
    missing_keys = sorted(expected_key_set - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_key_set)
    rank_mismatches = sorted(
        key
        for key, record in records_by_key.items()
        if key in expected_rank and int(record["rank"]) != expected_rank[key]
    )

    required_metrics = [str(name) for name in benchmark_payload.get("required_metrics", [])]
    missing_required_metrics: Dict[str, List[str]] = {}
    for key, record in records_by_key.items():
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics", {})
        missing = [name for name in required_metrics if name not in metrics]
        if missing:
            missing_required_metrics[key] = missing

    error_records = [
        record for record in records_by_key.values() if record.get("status") == "error"
    ]
    structural_errors = bool(
        len(all_records) != expected_count
        or duplicate_keys
        or missing_keys
        or unexpected_keys
        or rank_mismatches
        or missing_done_ranks
        or invalid_done_ranks
        or missing_required_metrics
    )
    error_policy_failed = fail_on_episode_error and bool(error_records)
    if structural_errors:
        status = "invalid"
    elif error_records:
        status = "completed_with_errors"
    else:
        status = "complete"

    summary: Mapping[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_manifest_digest": run_digest,
        "benchmark_manifest_digest": benchmark_envelope["digest"],
        "status": status,
        "expected_episode_count": expected_count,
        "record_count": len(all_records),
        "unique_record_count": len(records_by_key),
        "ok_episode_count": sum(
            record.get("status") == "ok" for record in records_by_key.values()
        ),
        "error_episode_count": len(error_records),
        "metrics": _numeric_metrics(records_by_key.values()),
        "validation": {
            "duplicate_keys": duplicate_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "rank_mismatches": rank_mismatches,
            "missing_done_ranks": missing_done_ranks,
            "invalid_done_ranks": invalid_done_ranks,
            "missing_required_metrics": missing_required_metrics,
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if strict and (structural_errors or error_policy_failed):
        raise ResultValidationError(
            "Evaluation result validation failed; inspect summary.json", summary
        )
    return summary

"""Rank-local JSONL persistence, resume, and metric aggregation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from satnav.evaluation._json import (
    atomic_write_json,
    canonical_json_bytes,
    read_json_object,
    to_jsonable,
)


RESULT_SCHEMA_VERSION = 1


class ResultError(RuntimeError):
    """Base class for evaluation result persistence failures."""


class ResumeRequiredError(ResultError):
    """Raised rather than silently appending to a prior non-resume run."""


def rank_directory(output_dir: Path, rank: int) -> Path:
    return Path(output_dir) / f"rank_{int(rank):05d}"


class RankResultStore:
    """Append-only result store owned by exactly one rank."""

    def __init__(self, output_dir: Path, rank: int):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
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
        return self.load_records(recover_truncated_tail=resume)

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
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                is_tail = index == len(lines) and not complete_line
                if recover_truncated_tail and is_tail:
                    with self.records_path.open("r+b") as handle:
                        handle.truncate(valid_bytes)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                raise ResultError(
                    f"Invalid JSONL record at {self.records_path}:{index}"
                ) from error
            records.append(decoded)
            valid_bytes += len(line)
            if not complete_line:
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
            return read_json_object(self.done_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ResultError(f"Cannot read done marker {self.done_path}: {error}") from error

    def mark_done(
        self,
        *,
        expected_keys: Sequence[str],
        records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Publish the local rank completion marker."""

        error_count = sum(record.get("status") == "error" for record in records)
        done = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rank": self.rank,
            "status": "completed_with_errors" if error_count else "complete",
            "expected_count": len(expected_keys),
            "record_count": len(records),
            "error_count": error_count,
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


def _rank_directories(output_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in Path(output_dir).glob("rank_[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )


def aggregate_run(
    output_dir: Path,
    *,
    fail_on_episode_error: bool = False,
) -> Mapping[str, Any]:
    """Merge rank-local records and write ``summary.json``."""

    output_dir = Path(output_dir)
    all_records: List[Mapping[str, Any]] = []
    done_markers: List[Mapping[str, Any]] = []
    for directory in _rank_directories(output_dir):
        rank = int(directory.name[len("rank_") :])
        store = RankResultStore(output_dir, rank)
        all_records.extend(store.load_records(recover_truncated_tail=False))
        done = store.load_done()
        if done is not None:
            done_markers.append(done)

    records_by_key: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(all_records):
        key = str(record.get("episode_key") or f"record-{index}")
        records_by_key[key] = record
    records = list(records_by_key.values())
    error_records = [record for record in records if record.get("status") == "error"]
    status = "completed_with_errors" if error_records else "complete"
    expected_count = sum(int(done.get("expected_count", 0)) for done in done_markers)
    summary: Mapping[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "expected_episode_count": expected_count or len(records),
        "record_count": len(all_records),
        "unique_record_count": len(records),
        "ok_episode_count": sum(record.get("status") == "ok" for record in records),
        "error_episode_count": len(error_records),
        "metrics": _numeric_metrics(records),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if fail_on_episode_error and error_records:
        raise ResultError("Evaluation contains episode errors; inspect summary.json")
    return summary

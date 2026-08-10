#!/usr/bin/env python3
"""Normalize two known metadata defects in the official decord 0.6.0 wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from packaging.tags import Tag, sys_tags
from packaging.utils import canonicalize_name


DECORD_VERSION = "0.6.0"
LEGACY_INTERNAL_TAG = "cp36-cp36m-manylinux2010_x86_64"
PUBLISHED_WHEEL_TAG = "py3-none-manylinux2010_x86_64"
REQUIRED_PLATFORM_TAG = Tag("py3", "none", "manylinux2010_x86_64")

# PyPI artifact: decord-0.6.0-py3-none-manylinux2010_x86_64.whl
OFFICIAL_WHEEL_ARCHIVE_SIZE = 13_602_299
OFFICIAL_WHEEL_ARCHIVE_SHA256 = (
    "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
)
OFFICIAL_PAYLOAD_RECORD_COUNT = 54
OFFICIAL_PAYLOAD_RECORD_SIZE = 4_686
OFFICIAL_PAYLOAD_RECORD_SHA256 = (
    "d150393947fd254588462c73496305c19963691c770e40b48f9e88cc20c6f1ad"
)
OFFICIAL_LIBDECORD_SHA256 = (
    "98b260c5812106648ba299279916fbe98439893e346d4efdcf5cde66ba8973da"
)
OFFICIAL_METADATA = (
    b"Metadata-Version: 2.1\n"
    b"Name: decord\n"
    b"Version: 0.6.0\n"
    b"Summary: Decord Video Loader\n"
    b"Home-page: https://github.com/dmlc/decord\n"
    b"Maintainer: Decord committers\n"
    b"Maintainer-email: cheungchih@gmail.com\n"
    b"License: APACHE\n"
    b"Platform: UNKNOWN\n"
    b"Classifier: Development Status :: 3 - Alpha\n"
    b"Classifier: Programming Language :: Python :: 3\n"
    b"Classifier: License :: OSI Approved :: Apache Software License\n"
    b"Requires-Dist: numpy (>=1.14.0)\n"
    b"\n"
    b"UNKNOWN\n"
    b"\n"
    b"\n"
)
OFFICIAL_TOP_LEVEL = b"decord\n"
OFFICIAL_STALE_TOP_LEVEL_RECORD = [
    "sha256=8TBMC8W9caRfSBphoy47j2wFImKqCOgWKD3JVELo5e0",
    "17",
]

ORIGINAL_WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: bdist_wheel (0.36.2)\n"
    b"Root-Is-Purelib: false\n"
    b"Tag: cp36-cp36m-manylinux2010_x86_64\n"
    b"\n"
)
NORMALIZED_WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: bdist_wheel (0.36.2)\n"
    b"Root-Is-Purelib: false\n"
    b"Tag: py3-none-manylinux2010_x86_64\n"
    b"\n"
)

_DIST_INFO_NAME = "decord-0.6.0.dist-info"
_ALLOWED_RECORD_ROOTS = {"decord", "decord.libs", _DIST_INFO_NAME}
_ALLOWED_DIST_INFO_FILES = {
    "INSTALLER",
    "METADATA",
    "RECORD",
    "REQUESTED",
    "WHEEL",
    "direct_url.json",
    "top_level.txt",
}
_PYC_PATH = re.compile(r"decord/(?:[^/]+/)*__pycache__/[^/]+\.pyc\Z")


def _hash_field(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return f"sha256={digest.rstrip(b'=').decode('ascii')}"


def _stage_file(path: Path, content: bytes, mode: int) -> Path:
    staging_directory = (
        path.parent.parent if path.parent.name == _DIST_INFO_NAME else path.parent
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(staging_directory),
        prefix=f".{_DIST_INFO_NAME}.{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write(path: Path, content: bytes, mode: int) -> None:
    temporary = _stage_file(path, content, mode)
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_metadata_pair(
    wheel_path: Path,
    wheel_content: bytes,
    record_path: Path,
    record_content: bytes,
) -> None:
    """Replace both files, restoring the first if the second replace fails."""

    original_wheel = wheel_path.read_bytes()
    original_record = record_path.read_bytes()
    wheel_stat = wheel_path.stat()
    record_stat = record_path.stat()
    wheel_mode = stat.S_IMODE(wheel_stat.st_mode)
    record_mode = stat.S_IMODE(record_stat.st_mode)
    wheel_temporary = None
    record_temporary = None
    wheel_replaced = False
    try:
        wheel_temporary = _stage_file(wheel_path, wheel_content, wheel_mode)
        record_temporary = _stage_file(record_path, record_content, record_mode)
        os.replace(wheel_temporary, wheel_path)
        wheel_replaced = True
        os.replace(record_temporary, record_path)
    except BaseException:
        if wheel_replaced:
            _atomic_write(wheel_path, original_wheel, wheel_mode)
            os.utime(
                wheel_path,
                ns=(wheel_stat.st_atime_ns, wheel_stat.st_mtime_ns),
                follow_symlinks=False,
            )
        if record_path.read_bytes() != original_record:
            _atomic_write(record_path, original_record, record_mode)
            os.utime(
                record_path,
                ns=(record_stat.st_atime_ns, record_stat.st_mtime_ns),
                follow_symlinks=False,
            )
        raise
    finally:
        for temporary in (wheel_temporary, record_temporary):
            if temporary is None:
                continue
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _record_rows(record_content: bytes) -> Tuple[List[List[str]], str]:
    try:
        text = record_content.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("decord RECORD is not valid UTF-8 CSV") from error
    if not rows or any(len(row) != 3 or not row[0] for row in rows):
        raise ValueError("every decord RECORD entry must have exactly three columns")
    for row in rows:
        pure = _validate_record_path(row[0])
        if pure.parts[0] not in _ALLOWED_RECORD_ROOTS:
            raise ValueError(f"unexpected decord RECORD root: {row[0]!r}")
        if pure.parts[0] in {"decord", "decord.libs"} and len(pure.parts) < 2:
            raise ValueError(f"incomplete decord RECORD path: {row[0]!r}")
        if pure.parts[0] == _DIST_INFO_NAME and (
            len(pure.parts) != 2 or pure.parts[1] not in _ALLOWED_DIST_INFO_FILES
        ):
            raise ValueError(f"unexpected decord dist-info entry: {row[0]!r}")
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("decord RECORD contains duplicate paths")
    if b"\r\n" in record_content:
        if record_content.replace(b"\r\n", b"").find(b"\n") >= 0:
            raise ValueError("decord RECORD contains mixed newlines")
        newline = "\r\n"
    else:
        newline = "\n"
    return rows, newline


def _validate_record_path(relative: str) -> PurePosixPath:
    parts = relative.split("/")
    if (
        not relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"unsafe decord RECORD path: {relative!r}")
    return PurePosixPath(*parts)


def _safe_payload_path(site_packages: Path, relative: str) -> Path:
    pure = _validate_record_path(relative)
    candidate = site_packages
    for part in pure.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"decord payload path must not be a symlink: {relative}")
    return candidate


def _verify_recorded_file(site_packages: Path, row: Sequence[str]) -> bytes:
    path, recorded_hash, recorded_size = row
    if not recorded_hash.startswith("sha256="):
        raise ValueError(f"decord RECORD must use sha256 for {path}")
    if not recorded_size.isdecimal() or str(int(recorded_size)) != recorded_size:
        raise ValueError(f"decord RECORD has a non-canonical size for {path}")
    candidate = _safe_payload_path(site_packages, path)
    if not candidate.is_file():
        raise ValueError(f"decord RECORD payload is missing: {path}")
    content = candidate.read_bytes()
    if _hash_field(content) != recorded_hash or len(content) != int(recorded_size):
        raise ValueError(f"decord RECORD hash or size mismatch for {path}")
    return content


def _canonical_payload_record(rows: Iterable[Sequence[str]]) -> bytes:
    return (
        "".join(",".join(row) + "\n" for row in sorted(rows, key=lambda row: row[0]))
    ).encode("utf-8")


def _is_allowed_pyc(path: str) -> bool:
    return _PYC_PATH.fullmatch(path) is not None


def _verify_payload(site_packages: Path, rows: Sequence[Sequence[str]]) -> None:
    payload_rows: List[Sequence[str]] = []
    for row in rows:
        path, recorded_hash, recorded_size = row
        if not (path.startswith("decord/") or path.startswith("decord.libs/")):
            continue
        if recorded_hash and recorded_size:
            _verify_recorded_file(site_packages, row)
            payload_rows.append(row)
        elif recorded_hash or recorded_size or not _is_allowed_pyc(path):
            raise ValueError(f"unexpected unhashed decord payload entry: {path}")

    canonical = _canonical_payload_record(payload_rows)
    if (
        len(payload_rows) != OFFICIAL_PAYLOAD_RECORD_COUNT
        or len(canonical) != OFFICIAL_PAYLOAD_RECORD_SIZE
        or hashlib.sha256(canonical).hexdigest() != OFFICIAL_PAYLOAD_RECORD_SHA256
    ):
        raise ValueError("decord payload RECORD does not match the official PyPI wheel")

    libdecord = site_packages / "decord" / "libdecord.so"
    if hashlib.sha256(libdecord.read_bytes()).hexdigest() != OFFICIAL_LIBDECORD_SHA256:
        raise ValueError("decord/libdecord.so does not match the official PyPI wheel")

    recorded_paths = {row[0] for row in payload_rows}
    for package_name in ("decord", "decord.libs"):
        package_root = site_packages / package_name
        if package_root.is_symlink() or not package_root.is_dir():
            raise ValueError(f"missing or linked decord payload directory: {package_name}")
        for root, directories, files in os.walk(str(package_root), followlinks=False):
            root_path = Path(root)
            for name in directories:
                if (root_path / name).is_symlink():
                    raise ValueError("decord payload directories must not be symlinks")
            for name in files:
                candidate = root_path / name
                relative = candidate.relative_to(site_packages).as_posix()
                if candidate.is_symlink():
                    raise ValueError(f"decord payload path must not be a symlink: {relative}")
                if not _is_allowed_pyc(relative) and relative not in recorded_paths:
                    raise ValueError(f"unrecorded decord payload file: {relative}")


def _verify_direct_url(content: bytes) -> None:
    try:
        payload = json.loads(content.decode("utf-8"))
        archive_info = payload["archive_info"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("decord direct_url.json is not valid archive metadata") from error
    declared_hashes = []
    if "hash" in archive_info:
        declared_hashes.append(archive_info["hash"])
    hashes = archive_info.get("hashes", {})
    if "sha256" in hashes:
        declared_hashes.append(f"sha256={hashes['sha256']}")
    expected = f"sha256={OFFICIAL_WHEEL_ARCHIVE_SHA256}"
    if not declared_hashes or any(value != expected for value in declared_hashes):
        raise ValueError("decord direct_url.json does not identify the official wheel")


def _verify_dist_info(
    dist_info: Path,
    rows: Sequence[Sequence[str]],
    original_wheel: bytes,
) -> Tuple[str, str]:
    by_path = {row[0]: row for row in rows}
    prefix = f"{dist_info.name}/"
    required_names = {"INSTALLER", "METADATA", "RECORD", "WHEEL", "top_level.txt"}
    recorded_names = {
        path[len(prefix) :]
        for path in by_path
        if path.startswith(prefix)
    }
    if not required_names.issubset(recorded_names):
        missing = sorted(required_names - recorded_names)
        raise ValueError(f"decord RECORD is missing dist-info entries: {missing}")

    actual_names = set()
    for path in dist_info.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("decord dist-info may contain only regular files")
        if path.name not in _ALLOWED_DIST_INFO_FILES:
            raise ValueError(f"unexpected decord dist-info file: {path.name}")
        actual_names.add(path.name)
    if actual_names != recorded_names:
        raise ValueError("decord dist-info files and RECORD entries differ")

    record_relative = f"{prefix}RECORD"
    if by_path[record_relative] != [record_relative, "", ""]:
        raise ValueError("decord RECORD must contain one unhashed self-entry")

    wheel_relative = f"{prefix}WHEEL"
    wheel_row = by_path[wheel_relative]
    legacy_wheel_record = [_hash_field(ORIGINAL_WHEEL), str(len(ORIGINAL_WHEEL))]
    normalized_wheel_record = [
        _hash_field(NORMALIZED_WHEEL),
        str(len(NORMALIZED_WHEEL)),
    ]
    if wheel_row[1:] == legacy_wheel_record:
        wheel_record_status = "legacy"
    elif wheel_row[1:] == normalized_wheel_record:
        wheel_record_status = "normalized"
    else:
        raise ValueError("decord RECORD has an unexpected WHEEL hash or size")
    if (dist_info / "WHEEL").read_bytes() != original_wheel:
        raise ValueError("decord WHEEL changed while reading")

    top_level_relative = f"{prefix}top_level.txt"
    top_level_row = by_path[top_level_relative]
    normalized_top_level_record = [
        _hash_field(OFFICIAL_TOP_LEVEL),
        str(len(OFFICIAL_TOP_LEVEL)),
    ]
    if top_level_row[1:] == OFFICIAL_STALE_TOP_LEVEL_RECORD:
        top_level_record_status = "legacy"
    elif top_level_row[1:] == normalized_top_level_record:
        top_level_record_status = "normalized"
    else:
        raise ValueError("decord RECORD has an unexpected top_level.txt hash or size")
    if (dist_info / "top_level.txt").read_bytes() != OFFICIAL_TOP_LEVEL:
        raise ValueError("decord top_level.txt does not match the official wheel")

    policies = {
        "INSTALLER": b"pip\n",
        "METADATA": OFFICIAL_METADATA,
        "REQUESTED": b"",
    }
    for name in sorted(recorded_names - {"RECORD", "WHEEL", "top_level.txt"}):
        relative = f"{prefix}{name}"
        content = _verify_recorded_file(dist_info.parent, by_path[relative])
        expected = policies.get(name)
        if expected is not None and content != expected:
            raise ValueError(f"decord {name} does not match the expected pip artifact")
        if name == "direct_url.json":
            _verify_direct_url(content)
    return wheel_record_status, top_level_record_status


def _verify_platform() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("decord metadata normalization requires Linux x86_64")
    if REQUIRED_PLATFORM_TAG not in set(sys_tags()):
        raise ValueError(f"active Python does not support {PUBLISHED_WHEEL_TAG}")


def _render_record(
    rows: Sequence[Sequence[str]],
    wheel_relative: str,
    top_level_relative: str,
    newline: str,
) -> bytes:
    updated = [list(row) for row in rows]
    wheel_rows = [row for row in updated if row[0] == wheel_relative]
    if len(wheel_rows) != 1:
        raise ValueError("decord RECORD must contain exactly one WHEEL entry")
    wheel_rows[0][1] = _hash_field(NORMALIZED_WHEEL)
    wheel_rows[0][2] = str(len(NORMALIZED_WHEEL))
    top_level_rows = [row for row in updated if row[0] == top_level_relative]
    if len(top_level_rows) != 1:
        raise ValueError("decord RECORD must contain exactly one top_level.txt entry")
    top_level_rows[0][1] = _hash_field(OFFICIAL_TOP_LEVEL)
    top_level_rows[0][2] = str(len(OFFICIAL_TOP_LEVEL))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator=newline)
    writer.writerows(updated)
    return output.getvalue().encode("utf-8")


def normalize_decord_metadata(dist_info: Path) -> Dict[str, Any]:
    """Correct the verified WHEEL tag and stale top-level RECORD entry."""

    _verify_platform()
    supplied_dist_info = Path(dist_info)
    if supplied_dist_info.is_symlink():
        raise ValueError("decord dist-info must not be a symlink")
    dist_info = supplied_dist_info.resolve(strict=True)
    if dist_info.name != _DIST_INFO_NAME or not dist_info.is_dir():
        raise ValueError(f"expected {_DIST_INFO_NAME}")

    wheel_path = dist_info / "WHEEL"
    record_path = dist_info / "RECORD"
    installer_path = dist_info / "INSTALLER"
    for metadata_path in (wheel_path, record_path, installer_path):
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("decord metadata files must be regular, non-symlink files")
    if installer_path.read_bytes() != b"pip\n":
        raise ValueError("decord metadata normalization only supports pip installations")

    original_wheel = wheel_path.read_bytes()
    if original_wheel == ORIGINAL_WHEEL:
        wheel_status = "legacy"
    elif original_wheel == NORMALIZED_WHEEL:
        wheel_status = "normalized"
    else:
        raise ValueError("unexpected decord WHEEL content")

    original_record = record_path.read_bytes()
    rows, newline = _record_rows(original_record)
    wheel_relative = f"{dist_info.name}/WHEEL"
    top_level_relative = f"{dist_info.name}/top_level.txt"
    wheel_record_status, top_level_record_status = _verify_dist_info(
        dist_info, rows, original_wheel
    )
    _verify_payload(dist_info.parent, rows)
    desired_record = _render_record(
        rows, wheel_relative, top_level_relative, newline
    )
    if (
        wheel_status == "normalized"
        and wheel_record_status == "normalized"
        and top_level_record_status == "normalized"
    ):
        if desired_record != original_record:
            raise ValueError("normalized decord WHEEL has an inconsistent RECORD")
        status = "already_normalized"
    else:
        _replace_metadata_pair(
            wheel_path, NORMALIZED_WHEEL, record_path, desired_record
        )
        status = (
            "patched"
            if wheel_status == wheel_record_status == top_level_record_status == "legacy"
            else "recovered"
        )

    return {
        "distribution": "decord",
        "version": DECORD_VERSION,
        "status": status,
        "official_wheel_sha256": OFFICIAL_WHEEL_ARCHIVE_SHA256,
        "official_wheel_size": OFFICIAL_WHEEL_ARCHIVE_SIZE,
        "original_internal_tag": LEGACY_INTERNAL_TAG,
        "normalized_tag": PUBLISHED_WHEEL_TAG,
        "payload_record_sha256": OFFICIAL_PAYLOAD_RECORD_SHA256,
    }


def _installed_dist_info() -> Path:
    distributions = [
        distribution
        for distribution in importlib.metadata.distributions()
        if canonicalize_name(distribution.metadata.get("Name", "")) == "decord"
    ]
    if len(distributions) != 1:
        raise ValueError("expected exactly one installed decord distribution")
    distribution = distributions[0]
    if distribution.version != DECORD_VERSION:
        raise ValueError(
            f"expected decord {DECORD_VERSION}, found {distribution.version}"
        )
    if distribution.read_text("INSTALLER") != "pip\n":
        raise ValueError("decord metadata normalization only supports pip installations")
    wheel_files = [
        path
        for path in distribution.files or ()
        if path.name == "WHEEL" and path.parent.name == _DIST_INFO_NAME
    ]
    if len(wheel_files) != 1:
        raise ValueError("cannot resolve one installed decord WHEEL metadata file")
    located_wheel = Path(distribution.locate_file(wheel_files[0]))
    if located_wheel.is_symlink() or located_wheel.parent.is_symlink():
        raise ValueError("installed decord metadata must not use symlinks")
    dist_info = located_wheel.resolve(strict=True).parent
    environment = Path(sys.prefix).resolve(strict=True)
    try:
        dist_info.relative_to(environment)
    except ValueError as error:
        raise ValueError("refusing to patch decord outside the active environment") from error
    return dist_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = normalize_decord_metadata(_installed_dist_info())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

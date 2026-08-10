import base64
import csv
import hashlib
import io
from pathlib import Path

import pytest

from baselines.vlm.streamvln.scripts import normalize_decord_wheel as normalizer


def _hash_field(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return f"sha256={digest.rstrip(b'=').decode('ascii')}"


def _hashed_row(path: str, content: bytes):
    return [path, _hash_field(content), str(len(content))]


def _write_record(path: Path, rows) -> None:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerows(rows)
    path.write_bytes(output.getvalue().encode("utf-8"))


def _dist_info(tmp_path: Path, monkeypatch, normalized: bool = False) -> Path:
    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "decord-0.6.0.dist-info"
    dist_info.mkdir(parents=True)
    payload = {
        "decord/__init__.py": b"version = '0.6.0'\n",
        "decord/libdecord.so": b"test-native-library",
        "decord.libs/libavcodec.so": b"test-linked-library",
    }
    payload_rows = []
    for relative, content in payload.items():
        path = site_packages / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        payload_rows.append(_hashed_row(relative, content))
    pyc = site_packages / "decord" / "__pycache__" / "__init__.cpython-39.pyc"
    pyc.parent.mkdir()
    pyc.write_bytes(b"generated-bytecode")

    wheel = normalizer.NORMALIZED_WHEEL if normalized else normalizer.ORIGINAL_WHEEL
    (dist_info / "WHEEL").write_bytes(wheel)
    (dist_info / "INSTALLER").write_bytes(b"pip\n")
    (dist_info / "METADATA").write_bytes(normalizer.OFFICIAL_METADATA)
    (dist_info / "top_level.txt").write_bytes(normalizer.OFFICIAL_TOP_LEVEL)
    rows = sorted(payload_rows, key=lambda row: row[0])
    top_level_record = (
        _hashed_row(f"{dist_info.name}/top_level.txt", normalizer.OFFICIAL_TOP_LEVEL)
        if normalized
        else [
            f"{dist_info.name}/top_level.txt",
            *normalizer.OFFICIAL_STALE_TOP_LEVEL_RECORD,
        ]
    )
    rows.extend(
        [
            ["decord/__pycache__/__init__.cpython-39.pyc", "", ""],
            _hashed_row(f"{dist_info.name}/INSTALLER", b"pip\n"),
            _hashed_row(f"{dist_info.name}/METADATA", normalizer.OFFICIAL_METADATA),
            [f"{dist_info.name}/RECORD", "", ""],
            _hashed_row(f"{dist_info.name}/WHEEL", wheel),
            top_level_record,
        ]
    )
    _write_record(dist_info / "RECORD", rows)
    for path in (dist_info / "WHEEL", dist_info / "RECORD"):
        path.chmod(0o644)

    canonical = normalizer._canonical_payload_record(payload_rows)
    monkeypatch.setattr(
        normalizer, "OFFICIAL_PAYLOAD_RECORD_COUNT", len(payload_rows)
    )
    monkeypatch.setattr(normalizer, "OFFICIAL_PAYLOAD_RECORD_SIZE", len(canonical))
    monkeypatch.setattr(
        normalizer,
        "OFFICIAL_PAYLOAD_RECORD_SHA256",
        hashlib.sha256(canonical).hexdigest(),
    )
    monkeypatch.setattr(
        normalizer,
        "OFFICIAL_LIBDECORD_SHA256",
        hashlib.sha256(payload["decord/libdecord.so"]).hexdigest(),
    )
    return dist_info


def _metadata_state(dist_info: Path):
    return {
        path.name: (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_mode & 0o777,
        )
        for path in (dist_info / "WHEEL", dist_info / "RECORD")
    }


def test_published_artifact_constants_are_exact():
    assert len(normalizer.ORIGINAL_WHEEL) == 112
    assert hashlib.sha256(normalizer.ORIGINAL_WHEEL).hexdigest() == (
        "d24f09731316657ed32488ba245812cbb5047342b148776a371264e69966d856"
    )
    assert len(normalizer.NORMALIZED_WHEEL) == 110
    assert hashlib.sha256(normalizer.NORMALIZED_WHEEL).hexdigest() == (
        "c920ad946c1010397bd118151dac11b8d2282bac1e7196b29ca34c23f1e86a8e"
    )
    assert normalizer.OFFICIAL_WHEEL_ARCHIVE_SIZE == 13_602_299
    assert normalizer.OFFICIAL_WHEEL_ARCHIVE_SHA256 == (
        "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
    )
    assert len(normalizer.OFFICIAL_METADATA) == 422
    assert _hash_field(normalizer.OFFICIAL_METADATA) == (
        "sha256=xxbfHI0xijGKytw1rRFgdw4P8XdqGaS34ptvwQzNGaY"
    )
    assert normalizer.OFFICIAL_TOP_LEVEL == b"decord\n"
    assert _hash_field(normalizer.OFFICIAL_TOP_LEVEL) == (
        "sha256=2gcXRGxvur2Z1iLmIE4fxL6cmywVZYD__8rzDGIEZDM"
    )


def test_normalize_decord_metadata_corrects_tag_record_and_preserves_mode(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)

    result = normalizer.normalize_decord_metadata(dist_info)

    assert result["status"] == "patched"
    wheel_path = dist_info / "WHEEL"
    record_path = dist_info / "RECORD"
    assert wheel_path.read_bytes() == normalizer.NORMALIZED_WHEEL
    rows = list(csv.reader(record_path.read_text().splitlines()))
    wheel_row = next(row for row in rows if row[0].endswith("/WHEEL"))
    assert wheel_row == [
        "decord-0.6.0.dist-info/WHEEL",
        _hash_field(normalizer.NORMALIZED_WHEEL),
        "110",
    ]
    top_level_row = next(row for row in rows if row[0].endswith("/top_level.txt"))
    assert top_level_row == [
        "decord-0.6.0.dist-info/top_level.txt",
        _hash_field(normalizer.OFFICIAL_TOP_LEVEL),
        "7",
    ]
    assert wheel_path.stat().st_mode & 0o777 == 0o644
    assert record_path.stat().st_mode & 0o777 == 0o644


def test_normalize_decord_metadata_is_fully_idempotent(tmp_path, monkeypatch):
    dist_info = _dist_info(tmp_path, monkeypatch)
    normalizer.normalize_decord_metadata(dist_info)
    before = _metadata_state(dist_info)

    result = normalizer.normalize_decord_metadata(dist_info)

    assert result["status"] == "already_normalized"
    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_recovers_interrupted_pair_update(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info / "WHEEL").write_bytes(normalizer.NORMALIZED_WHEEL)

    result = normalizer.normalize_decord_metadata(dist_info)

    assert result["status"] == "recovered"
    assert (dist_info / "WHEEL").read_bytes() == normalizer.NORMALIZED_WHEEL
    rows = list(csv.reader((dist_info / "RECORD").read_text().splitlines()))
    wheel_row = next(row for row in rows if row[0].endswith("/WHEEL"))
    assert wheel_row[1:] == [_hash_field(normalizer.NORMALIZED_WHEEL), "110"]


def test_normalize_decord_metadata_recovers_with_external_staging_leftover(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info / "WHEEL").write_bytes(normalizer.NORMALIZED_WHEEL)
    leftover = dist_info.parent / ".decord-0.6.0.dist-info.RECORD.interrupted.tmp"
    leftover.write_bytes(b"incomplete staging data")

    result = normalizer.normalize_decord_metadata(dist_info)

    assert result["status"] == "recovered"
    assert leftover.read_bytes() == b"incomplete staging data"


def test_normalize_decord_metadata_rolls_back_failed_second_replace(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    before = _metadata_state(dist_info)
    real_replace = normalizer.os.replace
    failed = False

    def fail_record_once(source, destination):
        nonlocal failed
        if not failed and Path(destination).name == "RECORD":
            failed = True
            raise OSError("injected RECORD replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(normalizer.os, "replace", fail_record_once)

    with pytest.raises(OSError, match="injected RECORD replace failure"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before
    assert not list(dist_info.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    "replacement",
    [
        normalizer.ORIGINAL_WHEEL.replace(
            b"cp36-cp36m", b"cp39-cp39"
        ),
        normalizer.ORIGINAL_WHEEL + b"Tag: py3-none-manylinux2010_x86_64\n",
    ],
)
def test_normalize_decord_metadata_rejects_unexpected_wheel_bytes(
    tmp_path, monkeypatch, replacement
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info / "WHEEL").write_bytes(replacement)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unexpected decord WHEEL content"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_unsupported_platform(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    monkeypatch.setattr(normalizer, "sys_tags", lambda: iter(()))
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="does not support"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_tampered_payload(tmp_path, monkeypatch):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info.parent / "decord" / "__init__.py").write_bytes(b"tampered")
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="hash or size mismatch"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_unrecorded_payload(tmp_path, monkeypatch):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info.parent / "decord" / "unrecorded.py").write_bytes(b"unexpected")
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unrecorded decord payload file"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_dependency_metadata_tamper(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    metadata_path = dist_info / "METADATA"
    tampered = normalizer.OFFICIAL_METADATA.replace(
        b"Requires-Dist: numpy (>=1.14.0)\n", b""
    )
    metadata_path.write_bytes(tampered)
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    metadata_row = next(row for row in rows if row[0].endswith("/METADATA"))
    metadata_row[1:] = _hashed_row(metadata_row[0], tampered)[1:]
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="METADATA does not match"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_tampered_top_level(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    (dist_info / "top_level.txt").write_bytes(b"malicious\n")
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="top_level.txt does not match"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_unexpected_top_level_record(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    top_level_row = next(row for row in rows if row[0].endswith("/top_level.txt"))
    top_level_row[1:] = ["sha256=unexpected", "7"]
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unexpected top_level.txt hash or size"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


@pytest.mark.parametrize("unexpected_path", ["evil.pth", "other-package/code.py"])
def test_normalize_decord_metadata_rejects_unexpected_record_root(
    tmp_path, monkeypatch, unexpected_path
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    path = dist_info.parent / unexpected_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"import os\n")
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    rows.append(_hashed_row(unexpected_path, path.read_bytes()))
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unexpected decord RECORD root"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_unknown_dist_info_entry(
    tmp_path, monkeypatch
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    injected = dist_info / "entry_points.txt"
    injected.write_bytes(b"[console_scripts]\n")
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    rows.append(_hashed_row(f"{dist_info.name}/{injected.name}", injected.read_bytes()))
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unexpected decord dist-info entry"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


@pytest.mark.parametrize("root_only", ["decord", "decord.libs"])
def test_normalize_decord_metadata_rejects_incomplete_payload_path(
    tmp_path, monkeypatch, root_only
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    rows.append([root_only, "", ""])
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="incomplete decord RECORD path"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


@pytest.mark.parametrize("case", ["malformed", "duplicate", "missing"])
def test_normalize_decord_metadata_rejects_invalid_record_without_writing(
    tmp_path, monkeypatch, case
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    wheel_index = next(index for index, row in enumerate(rows) if row[0].endswith("/WHEEL"))
    if case == "malformed":
        record_path.write_bytes(record_path.read_bytes() + b"only,two\r\n")
    elif case == "duplicate":
        rows.append(rows[wheel_index])
        _write_record(record_path, rows)
    else:
        del rows[wheel_index]
        _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "decord/../__pycache__/escape.pyc",
        "../../outside-target",
        "decord-0.6.0.dist-info/../../outside-target",
    ],
)
def test_normalize_decord_metadata_rejects_every_unsafe_record_path(
    tmp_path, monkeypatch, unsafe_path
):
    dist_info = _dist_info(tmp_path, monkeypatch)
    record_path = dist_info / "RECORD"
    rows = list(csv.reader(record_path.read_text().splitlines()))
    rows.append([unsafe_path, "", ""])
    _write_record(record_path, rows)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="unsafe decord RECORD path"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_payload_symlink(tmp_path, monkeypatch):
    dist_info = _dist_info(tmp_path, monkeypatch)
    payload = dist_info.parent / "decord" / "__init__.py"
    target = tmp_path / "target.py"
    payload.rename(target)
    payload.symlink_to(target)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="must not be a symlink"):
        normalizer.normalize_decord_metadata(dist_info)

    assert _metadata_state(dist_info) == before


def test_normalize_decord_metadata_rejects_dist_info_symlink(tmp_path, monkeypatch):
    dist_info = _dist_info(tmp_path, monkeypatch)
    linked = tmp_path / "linked-dist-info"
    linked.symlink_to(dist_info, target_is_directory=True)
    before = _metadata_state(dist_info)

    with pytest.raises(ValueError, match="dist-info must not be a symlink"):
        normalizer.normalize_decord_metadata(linked)

    assert _metadata_state(dist_info) == before


def test_installed_dist_info_rejects_wrong_version(monkeypatch):
    class Distribution:
        metadata = {"Name": "decord"}
        version = "0.7.0"

    monkeypatch.setattr(
        normalizer.importlib.metadata, "distributions", lambda: [Distribution()]
    )

    with pytest.raises(ValueError, match="expected decord 0.6.0"):
        normalizer._installed_dist_info()

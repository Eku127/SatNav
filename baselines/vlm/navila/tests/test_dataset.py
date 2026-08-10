import json
import os
from pathlib import Path

import pytest

from baselines.vlm.navila.dataset import (
    build_sample_indices,
    frame_path,
    load_annotation_records,
    resolve_video_directory,
    validate_records,
)


def _trajectory_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectory_data"
    rgb = root / "images" / "scene_satnav_000001" / "rgb"
    rgb.mkdir(parents=True)
    for index in range(1, 5):
        (rgb / f"{index:03d}.jpg").write_bytes(b"jpeg")
    annotations = [
        {
            "id": 1,
            "trajectory_id": "t1",
            "steps": 3,
            "video": "images/scene_satnav_000001",
            "instructions": ["go north"],
            "actions": [-1, 1, 2, 0],
        }
    ]
    (root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    return root


def test_current_schema_and_sample_mapping(tmp_path):
    root = _trajectory_root(tmp_path)
    annotation_path, records = load_annotation_records(root)
    assert annotation_path == root / "annotations.json"
    assert records[0].actions == (-1, 1, 2, 0)
    assert (
        frame_path(records[0], 3)
        == root / "images" / "scene_satnav_000001" / "rgb" / "003.jpg"
    )
    assert [sample.step_index for sample in build_sample_indices(records)] == [1, 2, 3]


def test_absolute_traversal_and_symlink_escape_are_rejected(tmp_path):
    external = tmp_path / "external"
    (external / "rgb").mkdir(parents=True)
    with pytest.raises(ValueError, match="absolute"):
        resolve_video_directory(tmp_path / "trajectory_data", str(external))
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video_directory(tmp_path / "trajectory_data", "../external")
    root = tmp_path / "trajectory_data"
    root.mkdir()
    (root / "linked").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        resolve_video_directory(root, "linked")


def test_frame_symlink_escape_is_rejected(tmp_path):
    root = _trajectory_root(tmp_path)
    _, records = load_annotation_records(root)
    external = tmp_path / "outside.jpg"
    external.write_bytes(b"jpeg")
    target = root / "images" / "scene_satnav_000001" / "rgb" / "001.jpg"
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="symlink escapes"):
        frame_path(records[0], 1)


def test_sampling_cap_and_stop_repeat(tmp_path):
    _, records = load_annotation_records(_trajectory_root(tmp_path))
    samples = build_sample_indices(records, stop_repeat=3)
    assert [sample.step_index for sample in samples] == [1, 2, 3, 3, 3]
    assert len(build_sample_indices(records, max_samples=2)) == 2


def test_content_tree_digest_detects_same_size_same_mtime_replacement(tmp_path):
    root = _trajectory_root(tmp_path)
    annotation_path, records = load_annotation_records(root)
    before = validate_records(
        annotation_path,
        records,
        strict_frames=True,
        decode_samples=0,
        hash_frame_content=True,
    )
    frame = root / "images" / "scene_satnav_000001" / "rgb" / "002.jpg"
    stat = frame.stat()
    frame.write_bytes(b"evil")
    os.utime(frame, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = validate_records(
        annotation_path,
        records,
        strict_frames=True,
        decode_samples=0,
        hash_frame_content=True,
    )
    assert before["frame_tree"]["digest"] != after["frame_tree"]["digest"]
    assert before["frame_tree"]["algorithm"].endswith("content-v1")

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.vlm.uninavid.dataset import (
    SatNavUniNaVidDataset,
    build_sample_indices,
    frame_path,
    load_annotation_records,
    resolve_video_directory,
    validate_records,
)


def _trajectory_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectory_data"
    rgb = root / "images" / "sample" / "rgb"
    rgb.mkdir(parents=True)
    from PIL import Image

    for index in range(1, 6):
        Image.new("RGB", (2, 2), (index, 0, 0)).save(rgb / f"{index:03d}.jpg")
    annotations = [
        {
            "id": 7,
            "trajectory_id": "t7",
            "steps": 5,
            "video": "images/sample",
            "instructions": ["go north"],
            "actions": [-1, 1, 2, 3, 1, 0],
        }
    ]
    (root / "annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    return root


def test_window_mapping_schema_and_paths(tmp_path):
    root = _trajectory_root(tmp_path)
    annotation, records = load_annotation_records(root)
    samples = build_sample_indices(records)
    assert annotation == root / "annotations.json"
    assert [(sample.start, sample.actions) for sample in samples] == [
        (0, (1, 2, 3, 1)),
        (4, (0, 0, 0, 0)),
    ]
    assert [sample.history_frames for sample in samples] == [1, 5]
    assert frame_path(records[0], 5) == root / "images/sample/rgb/005.jpg"


def test_traversal_absolute_and_frame_symlink_escape_are_rejected(tmp_path):
    root = _trajectory_root(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    with pytest.raises(ValueError, match="absolute"):
        resolve_video_directory(root, str(external))
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video_directory(root, "../external")
    _, records = load_annotation_records(root)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    frame = root / "images/sample/rgb/001.jpg"
    frame.unlink()
    frame.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink escapes"):
        frame_path(records[0], 1)


def test_full_content_tree_detects_same_size_replacement(tmp_path):
    root = _trajectory_root(tmp_path)
    annotation, records = load_annotation_records(root)
    before = validate_records(
        annotation,
        records,
        strict_frames=True,
        hash_frame_content=True,
        decode_samples=0,
    )
    frame = root / "images/sample/rgb/003.jpg"
    stat = frame.stat()
    value = bytearray(frame.read_bytes())
    value[-3] ^= 1
    frame.write_bytes(value)
    os.utime(frame, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = validate_records(
        annotation,
        records,
        strict_frames=True,
        hash_frame_content=True,
        decode_samples=0,
    )
    assert before["frame_tree"]["digest"] != after["frame_tree"]["digest"]


def test_frame_tree_matches_navila_canonical_known_vector(tmp_path):
    from baselines.vlm.navila.dataset import (
        load_annotation_records as load_navila_records,
    )
    from baselines.vlm.navila.dataset import validate_records as validate_navila_records

    root = tmp_path / "trajectory_data"
    rgb = root / "images" / "sample" / "rgb"
    rgb.mkdir(parents=True)
    (rgb / "001.jpg").write_bytes(b"abc")
    (rgb / "002.jpg").write_bytes(b"defg")
    annotations = [
        {
            "id": 1,
            "steps": 1,
            "video": "images/sample",
            "instructions": ["stop"],
            "actions": [-1, 0],
        }
    ]
    (root / "annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    annotation, records = load_annotation_records(root)
    report = validate_records(
        annotation,
        records,
        strict_frames=True,
        hash_frame_content=True,
        decode_samples=0,
    )
    # Repeated sha256(relative-path + NUL + decimal-size + NUL + raw-content
    # + NUL) is the cross-baseline SatNav/NaVILA content-v1 canonical stream.
    assert report["frame_tree"] == {
        "algorithm": "sha256-relative-path-size-content-v1",
        "digest": "fccc68d143e552cf1d14bbb93f406a37fc1b50620f9a911e89e3fd8095c51d98",
        "file_count": 2,
        "total_bytes": 7,
    }
    navila_annotation, navila_records = load_navila_records(root)
    navila_report = validate_navila_records(
        navila_annotation,
        navila_records,
        strict_frames=True,
        hash_frame_content=True,
        decode_samples=0,
    )
    assert navila_report["frame_tree"] == report["frame_tree"]


def test_labels_use_last_assistant_boundary():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(input_ids=torch.tensor([[7, 8]]))

    dataset = object.__new__(SatNavUniNaVidDataset)
    dataset._torch = torch
    dataset.tokenizer = Tokenizer()
    labels = dataset.build_labels(torch.tensor([7, 8, 3, 7, 8, 9, 10]))
    assert labels.tolist() == [-100, -100, -100, -100, -100, 9, 10]


def test_getitem_fails_same_sample_without_random_substitution():
    dataset = object.__new__(SatNavUniNaVidDataset)
    dataset.samples = [SimpleNamespace(record_index=0, window_index=2)]
    dataset.records = [SimpleNamespace(payload={"id": "episode"})]
    dataset._load_sample = lambda sample: (_ for _ in ()).throw(ValueError("bad"))
    with pytest.raises(RuntimeError, match="index=0, episode_id=episode"):
        dataset[0]

import json
from pathlib import Path

import pytest

from baselines.vlm.streamvln.dataset import (
    build_sample_indices,
    frame_paths,
    load_annotation_records,
    main,
    normalize_actions,
    resolve_video_directory,
)


def _trajectory_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectory_data"
    rgb = root / "images" / "scene_satnav_000001" / "rgb"
    rgb.mkdir(parents=True)
    annotations = [
        {
            "id": 1,
            "trajectory_id": "t1",
            "steps": 4,
            "video": "images/scene_satnav_000001",
            "instructions": ["go north"],
            "actions": [-1, 1, 2, 3, 0],
        }
    ]
    (root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    return root


def test_current_schema_skips_init_and_keeps_existing_stop(tmp_path):
    root = _trajectory_root(tmp_path)
    records = load_annotation_records([root])
    assert records[0].actions == (1, 2, 3, 0)
    assert records[0].video_dir == root / "images" / "scene_satnav_000001"
    assert build_sample_indices(records, num_frames=32)[0].start_index == 0


def test_preflight_rejects_an_empty_training_sample_set(tmp_path):
    root = _trajectory_root(tmp_path)
    annotation_path = root / "annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations[0].update(steps=1, actions=[-1, 0])
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    records = load_annotation_records([root])
    assert build_sample_indices(records, num_frames=32) == []
    with pytest.raises(ValueError, match="no training samples"):
        main([str(root), "--num-frames", "32"])


def test_legacy_images_root_finds_parent_annotations_without_double_images(tmp_path):
    root = _trajectory_root(tmp_path)
    records = load_annotation_records([root / "images"])
    assert records[0].video_dir == root / "images" / "scene_satnav_000001"


def test_archived_action_sequence_gets_terminal_stop():
    assert normalize_actions([1, 1, 3]) == (1, 1, 3, 0)


def test_relative_video_traversal_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video_directory(tmp_path, "../outside")


def test_absolute_video_path_requires_explicit_legacy_opt_in(tmp_path):
    external = tmp_path / "external"
    (external / "rgb").mkdir(parents=True)
    with pytest.raises(ValueError, match="absolute"):
        resolve_video_directory(tmp_path / "trajectory_data", str(external))
    assert (
        resolve_video_directory(
            tmp_path / "trajectory_data",
            str(external),
            allow_external_video_paths=True,
        )
        == external
    )


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "trajectory_data"
    external = tmp_path / "external"
    (external / "rgb").mkdir(parents=True)
    root.mkdir()
    (root / "linked").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        resolve_video_directory(root, "linked")


def test_rgb_directory_symlink_escape_is_rejected(tmp_path):
    root = _trajectory_root(tmp_path)
    rgb = root / "images" / "scene_satnav_000001" / "rgb"
    rgb.rmdir()
    external = tmp_path / "external-rgb"
    external.mkdir()
    rgb.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="RGB directory escapes"):
        load_annotation_records([root])


def test_frame_symlink_escape_is_rejected(tmp_path):
    root = _trajectory_root(tmp_path)
    record = load_annotation_records([root])[0]
    rgb = record.video_dir / "rgb"
    external = tmp_path / "external.jpg"
    external.touch()
    (rgb / "001.jpg").symlink_to(external)

    with pytest.raises(ValueError, match="RGB frame escapes"):
        frame_paths(record.video_dir)


def test_chunk_index_count_and_optional_sample_cap(tmp_path):
    root = _trajectory_root(tmp_path)
    record = load_annotation_records([root])[0]
    expanded = type(record)(
        source_root=record.source_root,
        video_dir=record.video_dir,
        instructions=("one", "two"),
        actions=tuple([1] * 64 + [0]),
        payload=record.payload,
    )
    assert len(build_sample_indices([expanded], num_frames=32)) == 6
    assert len(build_sample_indices([expanded], num_frames=32, max_samples=2)) == 2

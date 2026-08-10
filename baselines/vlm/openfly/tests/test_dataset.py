from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baselines.vlm.openfly.dataset import (
    IGNORE_INDEX,
    OpenFlyDataCollator,
    SamplingPolicy,
    SatNavOpenFlyDataset,
    load_annotation_records,
    resolve_video_directory,
    validate_records,
)
from baselines.vlm.openfly.trainer import (
    _require_dataset_files_unchanged,
    _validate_dataset_against_report,
)


def _fixture(root: Path) -> Path:
    trajectory = root / "trajectory_data"
    video = trajectory / "images" / "scene_a_satnav_episode"
    rgb = video / "rgb"
    rgb.mkdir(parents=True)
    for index in range(1, 5):
        Image.new("RGB", (4, 4), (index, 0, 0)).save(rgb / f"{index:03d}.jpg")
    annotation = [
        {
            "id": "a",
            "video": "images/scene_a_satnav_episode",
            "trajectory_id": "same-id",
            "instructions": [" Go   Home "],
            "actions": [-1, 1, 2, 0],
            "steps": 3,
        }
    ]
    (trajectory / "annotations.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    episode_dir = root / "episodes" / "train"
    episode_dir.mkdir(parents=True)
    episodes = {
        "episodes": [
            {
                "scene_id": "/maps/scene_a.tif",
                "trajectory_id": "same-id",
                "trajectory_type": "LandmarkSet",
                "instruction": {"instruction_text": "go home"},
            }
        ]
    }
    (episode_dir / "all_episodes.json").write_text(
        json.dumps(episodes), encoding="utf-8"
    )
    return trajectory


def test_exact_join_sampling_and_full_frame_validation(tmp_path: Path) -> None:
    trajectory = _fixture(tmp_path)
    annotation, episodes, records = load_annotation_records(trajectory)
    assert records[0].trajectory_type == "LandmarkSet"
    dataset = SatNavOpenFlyDataset(trajectory, policy=SamplingPolicy())
    assert len(dataset) == 4  # three steps, with STOP repeated twice
    sample = dataset[0]
    assert len(sample["images"]) == 3
    report = validate_records(
        annotation,
        episodes,
        records,
        policy=SamplingPolicy(),
        strict_frames=True,
        hash_frame_content=True,
        decode_samples=1,
    )
    assert report["status"] == "passed"
    assert report["frame_tree"]["file_count"] == 4


def test_dotted_logical_scene_survives_video_and_source_join(tmp_path: Path) -> None:
    trajectory = _fixture(tmp_path)
    old_video = trajectory / "images" / "scene_a_satnav_episode"
    new_video = trajectory / "images" / "Rome.v2_satnav_000000"
    old_video.rename(new_video)
    annotation_path = trajectory / "annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations[0]["video"] = "images/Rome.v2_satnav_000000"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    episode_path = tmp_path / "episodes" / "train" / "all_episodes.json"
    episodes = json.loads(episode_path.read_text(encoding="utf-8"))
    episodes["episodes"][0]["scene_id"] = "/maps/Rome.v2.tif"
    episode_path.write_text(json.dumps(episodes), encoding="utf-8")

    _, _, records = load_annotation_records(trajectory)

    assert records[0].logical_scene == "Rome.v2"


def _validation_fixture(
    tmp_path: Path, *, max_samples: int | None = None
) -> tuple[SatNavOpenFlyDataset, dict]:
    trajectory = _fixture(tmp_path)
    annotation, episodes, records = load_annotation_records(trajectory)
    policy = SamplingPolicy()
    report = dict(
        validate_records(
            annotation,
            episodes,
            records,
            policy=policy,
            strict_frames=True,
            hash_frame_content=True,
            decode_samples=0,
            frame_identity_index=tmp_path / "frame-identities.sha256",
        ),
        allow_external_paths=False,
    )
    frame_index = report["frame_identity_index"]
    return (
        SatNavOpenFlyDataset(
            trajectory,
            policy=policy,
            max_samples=max_samples,
            frame_identity_index=tmp_path / frame_index["artifact"]["id"],
            frame_identity_index_artifact=frame_index["artifact"],
        ),
        report,
    )


def test_validation_report_crosscheck_binds_files_and_full_counts(
    tmp_path: Path,
) -> None:
    dataset, report = _validation_fixture(tmp_path, max_samples=1)
    evidence = _validate_dataset_against_report(
        dataset,
        report,
        max_episodes=None,
        max_samples=1,
    )
    assert evidence["full_record_crosscheck"] is True
    assert evidence["selected_samples"] == 1
    payload = json.loads(dataset.annotation_path.read_text(encoding="utf-8"))
    dataset.annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="annotation"):
        _require_dataset_files_unchanged(dataset, evidence)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("annotations", 2),
        ("mapped_frames", 5),
        ("primitive_actions_excluding_init", 4),
        ("samples", 5),
        ("action_counts", {}),
        ("trajectory_type_counts", {}),
        ("sampling_policy", {}),
        ("episode_trajectory_type_counts", {}),
    ],
)
def test_validation_report_crosscheck_rejects_count_contradictions(
    tmp_path: Path, field: str, replacement: object
) -> None:
    dataset, report = _validation_fixture(tmp_path)
    contradictory = copy.deepcopy(report)
    contradictory[field] = replacement
    with pytest.raises((RuntimeError, ValueError), match="validation|frame-tree"):
        _validate_dataset_against_report(
            dataset,
            contradictory,
            max_episodes=None,
            max_samples=None,
        )


def test_validation_report_crosscheck_rejects_weak_frame_identity(
    tmp_path: Path,
) -> None:
    dataset, report = _validation_fixture(tmp_path)
    report["frame_tree"]["algorithm"] = "sha256-relative-path-size-mtime-v1"
    with pytest.raises(ValueError, match="frame-tree"):
        _validate_dataset_against_report(
            dataset,
            report,
            max_episodes=None,
            max_samples=None,
        )


def test_consumed_frame_rejects_post_preflight_content_mutation(
    tmp_path: Path,
) -> None:
    dataset, _ = _validation_fixture(tmp_path)
    frame = dataset.records[0].video_dir / "rgb" / "001.jpg"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(frame)
    with pytest.raises(RuntimeError, match="failed to load exact") as caught:
        dataset[0]
    assert "preflight identity" in str(caught.value.__cause__)


def test_frame_index_snapshot_rejects_coordinated_index_and_frame_mutation(
    tmp_path: Path,
) -> None:
    dataset, _ = _validation_fixture(tmp_path)
    frame = dataset.records[0].video_dir / "rgb" / "001.jpg"
    Image.new("RGB", (4, 4), (0, 255, 0)).save(frame)
    replacement = frame.read_bytes()
    assert dataset.frame_identity_index_path is not None
    with dataset.frame_identity_index_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(hashlib.sha256(replacement).digest())
    with pytest.raises(RuntimeError, match="failed to load exact") as caught:
        dataset[0]
    assert "preflight identity" in str(caught.value.__cause__)


def test_frame_index_artifact_mutation_is_rejected_after_preflight(
    tmp_path: Path,
) -> None:
    dataset, report = _validation_fixture(tmp_path)
    evidence = _validate_dataset_against_report(
        dataset,
        report,
        max_episodes=None,
        max_samples=None,
    )
    assert dataset.frame_identity_index_path is not None
    with dataset.frame_identity_index_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
    with pytest.raises(RuntimeError, match="frame identity index"):
        _require_dataset_files_unchanged(dataset, evidence)


def test_bounded_episode_dataset_uses_prefix_of_full_frame_index(
    tmp_path: Path,
) -> None:
    trajectory = _fixture(tmp_path)
    second_rgb = trajectory / "images" / "scene_b_satnav_episode" / "rgb"
    second_rgb.mkdir(parents=True)
    for index in range(1, 4):
        Image.new("RGB", (4, 4), (0, index, 0)).save(second_rgb / f"{index:03d}.jpg")
    annotations = json.loads((trajectory / "annotations.json").read_text())
    annotations.append(
        {
            "id": "b",
            "video": "images/scene_b_satnav_episode",
            "trajectory_id": "second-id",
            "instructions": ["Turn home"],
            "actions": [-1, 3, 0],
            "steps": 2,
        }
    )
    (trajectory / "annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    episode_path = tmp_path / "episodes" / "train" / "all_episodes.json"
    episodes = json.loads(episode_path.read_text())
    episodes["episodes"].append(
        {
            "scene_id": "/maps/scene_b.tif",
            "trajectory_id": "second-id",
            "trajectory_type": "Road",
            "instruction": {"instruction_text": "turn home"},
        }
    )
    episode_path.write_text(json.dumps(episodes), encoding="utf-8")
    annotation, episode, records = load_annotation_records(trajectory)
    policy = SamplingPolicy()
    index_path = tmp_path / "full-frame-identities.sha256"
    report = dict(
        validate_records(
            annotation,
            episode,
            records,
            policy=policy,
            strict_frames=True,
            hash_frame_content=True,
            decode_samples=0,
            frame_identity_index=index_path,
        ),
        allow_external_paths=False,
    )
    frame_index = report["frame_identity_index"]
    dataset = SatNavOpenFlyDataset(
        trajectory,
        policy=policy,
        max_episodes=1,
        frame_identity_index=index_path,
        frame_identity_index_artifact=frame_index["artifact"],
    )
    evidence = _validate_dataset_against_report(
        dataset,
        report,
        max_episodes=1,
        max_samples=None,
    )
    assert evidence["full_record_crosscheck"] is False
    assert dataset.frame_identity_index_file_count == 7
    assert dataset[0]["annotation_id"] == "a"


def test_join_does_not_fall_back_to_trajectory_id(tmp_path: Path) -> None:
    trajectory = _fixture(tmp_path)
    annotation = json.loads((trajectory / "annotations.json").read_text())
    annotation[0]["video"] = "images/wrong_scene_satnav_episode"
    (trajectory / "annotations.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no exact"):
        load_annotation_records(trajectory)


def test_safe_annotation_paths(tmp_path: Path) -> None:
    root = tmp_path / "trajectory"
    root.mkdir()
    with pytest.raises(ValueError, match="absolute"):
        resolve_video_directory(root, "/tmp/video")
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video_directory(root, "../video")


def test_action_schema_never_coerces_json_values(tmp_path: Path) -> None:
    trajectory = _fixture(tmp_path)
    payload = json.loads((trajectory / "annotations.json").read_text())
    payload[0]["actions"][1] = 1.9
    (trajectory / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON integers"):
        load_annotation_records(trajectory)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return SimpleNamespace(
            input_ids=[ord(char) + 1 for char in text],
            offset_mapping=[(index, index + 1) for index in range(len(text))],
        )


class _ImageProcessor:
    def __call__(self, images, return_tensors):
        assert len(images) == 3 and return_tensors == "pt"
        return {"pixel_values": torch.zeros(3, 6, 2, 2)}


def test_collator_supervises_only_last_assistant_answer() -> None:
    processor = SimpleNamespace(
        tokenizer=_Tokenizer(), image_processor=_ImageProcessor()
    )
    collator = OpenFlyDataCollator(
        processor=processor,
        model_max_length=2048,
        pad_token_id=0,
        action_format="compact",
    )
    instance = {
        "full_text": "[INST] human [/INST] forward</s>",
        "images": [Image.new("RGB", (2, 2)) for _ in range(3)],
    }
    batch = collator([instance])
    labels = batch["labels"][0]
    boundary_end = instance["full_text"].rfind("[/INST]") + len("[/INST]")
    assert torch.all(labels[:boundary_end] == IGNORE_INDEX)
    assert torch.any(labels[boundary_end:] != IGNORE_INDEX)
    assert batch["attention_mask"].all()


def test_original_action_subsequence_is_exact() -> None:
    positions = OpenFlyDataCollator._exact_action_positions(
        [1, 9, 8, 7, 6, 2], [1, 2, 3, 4, 5], [9, 8, 7, 6]
    )
    assert positions == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="absent or truncated"):
        OpenFlyDataCollator._exact_action_positions([1, 9, 8], [1, 2], [9, 8, 7])

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from baselines.vlm.openfly.artifacts import (
    OpenFlyArtifactError,
    openfly_model_identity,
    processor_identity,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_model(
    root: Path, *, backend: str = "continue", action: str = "compact"
) -> Path:
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "model_type": "openvla",
            "architectures": ["OpenVLAForActionPrediction"],
            "action_format": action,
            "grid_size": 16,
            "use_cache": False,
        },
    )
    _write_json(root / "preprocessor_config.json", {"image_processor_type": "x"})
    _write_json(root / "tokenizer_config.json", {"tokenizer_class": "x"})
    _write_json(root / "tokenizer.json", {"version": "1.0"})
    _write_json(
        root / "backend_meta.json", {"backend": backend, "action_format": action}
    )
    _write_json(
        root / "dataset_statistics.json", {"action_format": action, "num_samples": 3}
    )
    (root / "model.safetensors").write_bytes(b"weight-content")
    return root


def test_model_identity_is_content_not_basename(tmp_path: Path) -> None:
    left = make_model(tmp_path / "left")
    right = tmp_path / "completely-different-name"
    shutil.copytree(left, right)
    left_id = openfly_model_identity(left)
    right_id = openfly_model_identity(right)
    assert left_id["digest"] == right_id["digest"]
    (right / "model.safetensors").write_bytes(b"changed")
    assert openfly_model_identity(right)["digest"] != left_id["digest"]


def test_model_identity_excludes_training_state_and_evidence(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    before = openfly_model_identity(root)
    checkpoint = root / "checkpoint-1" / "global_step1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer_states.pt").write_bytes(b"large optimizer state")
    _write_json(root / "data_validation.json", {"status": "passed"})
    (root / "train.log").write_text("evidence", encoding="utf-8")
    assert openfly_model_identity(root)["digest"] == before["digest"]


def test_model_identity_binds_declared_inference_metadata(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    before = openfly_model_identity(root)
    _write_json(
        root / "dataset_statistics.json",
        {"action_format": "compact", "num_samples": 4},
    )
    assert openfly_model_identity(root)["digest"] != before["digest"]


def test_chat_template_mutates_model_and_processor_identities(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    model_before = openfly_model_identity(root)
    processor_before = processor_identity(root)
    _write_json(root / "chat_template.json", {"default": "{{ messages }}"})
    model_with_template = openfly_model_identity(root)
    processor_with_template = processor_identity(root)
    assert model_with_template["digest"] != model_before["digest"]
    assert processor_with_template["digest"] != processor_before["digest"]
    _write_json(root / "chat_template.json", {"default": "changed"})
    assert openfly_model_identity(root)["digest"] != model_with_template["digest"]
    assert processor_identity(root)["digest"] != processor_with_template["digest"]


def test_identity_rejects_contradictory_metadata(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    _write_json(
        root / "backend_meta.json", {"backend": "scratch", "action_format": "original"}
    )
    with pytest.raises(OpenFlyArtifactError, match="contradicts"):
        openfly_model_identity(root)


def test_identity_rejects_unsafe_index_shard(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    (root / "model.safetensors").unlink()
    _write_json(
        root / "model.safetensors.index.json",
        {"metadata": {"total_size": 1}, "weight_map": {"x": "../escape.safetensors"}},
    )
    with pytest.raises(OpenFlyArtifactError, match="unsafe"):
        openfly_model_identity(root)


def test_identity_rejects_unreferenced_weight_file(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    (root / "model.safetensors").unlink()
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (root / "unreferenced.safetensors").write_bytes(b"ambiguous")
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": {"x": "model-00001-of-00001.safetensors"},
        },
    )
    with pytest.raises(OpenFlyArtifactError, match="unreferenced"):
        openfly_model_identity(root)


def test_processor_identity_ignores_weight_files(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    before = processor_identity(root)
    (root / "model.safetensors").write_bytes(b"other")
    assert processor_identity(root)["digest"] == before["digest"]


def test_processor_identity_rejects_optional_symlink(tmp_path: Path) -> None:
    root = make_model(tmp_path / "model")
    outside = tmp_path / "template.json"
    _write_json(outside, {"chat_template": "outside"})
    (root / "chat_template.json").symlink_to(outside)
    with pytest.raises(OpenFlyArtifactError, match="symlink"):
        processor_identity(root)

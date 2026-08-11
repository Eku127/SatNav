"""Regression tests for portable validation reports and environment traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from scripts.validation import capture_env_trace
from scripts.validation import compare_env_traces
from scripts.validation import data_validation


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data_validation_report_uses_only_portable_input_identities(
    tmp_path, monkeypatch
):
    private_root = tmp_path / "private-host-root"
    scenes_dir = private_root / "scene-assets"
    scenes_dir.mkdir(parents=True)
    dataset_path = private_root / "episodes.json"
    missing_scene = scenes_dir / "SecretScene.tif"
    source_row = {
        "episode_id": 7,
        "trajectory_id": 11,
        "scene_id": str(missing_scene),
    }
    dataset_path.write_text(
        json.dumps({"episodes": [source_row, source_row]}),
        encoding="utf-8",
    )

    class FakeEpisode:
        episode_id = "7"
        trajectory_id = "11"
        scene_id = str(missing_scene)
        scene_path = str(missing_scene)
        episode_key = f"val_seen::{missing_scene}::7"

        @staticmethod
        def to_dict():
            return {
                "episode_id": "7",
                "trajectory_id": "11",
                "scene_id": str(missing_scene),
            }

    monkeypatch.setattr(
        data_validation,
        "SatNavDataset",
        lambda _config: SimpleNamespace(episodes=[FakeEpisode(), FakeEpisode()]),
    )
    report_path = private_root / "release-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "data_validation.py",
            "--split",
            f"val_seen={dataset_path}",
            "--expected",
            "val_seen=2",
            "--scenes-dir",
            str(scenes_dir),
            "--report",
            str(report_path),
        ],
    )

    assert data_validation.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    assert str(private_root) not in serialized
    assert payload["scene_dir"] == {"id": "scene-assets"}
    split = payload["splits"]["val_seen"]
    assert split["artifact"] == {
        "id": "episodes.json",
        "size": dataset_path.stat().st_size,
    }
    assert split["missing_scene_assets"] == [
        {"index": 0, "scene_id": "SecretScene"},
        {"index": 1, "scene_id": "SecretScene"},
    ]
    assert split["duplicate_keys"] == ["val_seen::SecretScene::7"]


def test_data_validation_runs_without_writing_a_report(
    tmp_path, monkeypatch, capsys
):
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    scene_asset_path = scenes_dir / "Scene-1.tif"
    scene_asset_path.touch()
    dataset_path = tmp_path / "episodes.json"
    source_row = {
        "episode_id": "7",
        "trajectory_id": "11",
        "scene_id": "Scene-1",
    }
    dataset_path.write_text(
        json.dumps({"episodes": [source_row]}),
        encoding="utf-8",
    )

    class FakeEpisode:
        episode_id = "7"
        trajectory_id = "11"
        scene_id = "Scene-1"
        scene_path = str(scene_asset_path)
        episode_key = "val_seen::Scene-1::7"

        @staticmethod
        def to_dict():
            return dict(source_row)

    monkeypatch.setattr(
        data_validation,
        "SatNavDataset",
        lambda _config: SimpleNamespace(episodes=[FakeEpisode()]),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "data_validation.py",
            "--split",
            f"val_seen={dataset_path}",
            "--expected",
            "val_seen=1",
            "--scenes-dir",
            str(scenes_dir),
        ],
    )

    assert data_validation.main() == 0
    output = capsys.readouterr().out
    assert "val_seen: passed (1 episodes)" in output
    assert "SatNav data configuration is complete." in output
    assert "Report:" not in output


def test_capture_trace_removes_checkout_and_scene_asset_paths(
    tmp_path, monkeypatch
):
    private_root = tmp_path / "machine-private-root"
    repo = private_root / "SatNav-checkout"
    repo.mkdir(parents=True)
    config_path = private_root / "task.yaml"
    config_path.write_text("TASK: {}\n", encoding="utf-8")
    dataset_path = private_root / "episodes.json"
    dataset_path.write_text('{"episodes": []}\n', encoding="utf-8")
    scenes_dir = private_root / "scene-assets"
    scenes_dir.mkdir()
    scene_asset = scenes_dir / "Amsterdam-1.tif"

    episode = SimpleNamespace(
        episode_id="7",
        trajectory_id="11",
        scene_id=str(scene_asset),
        episode_key=f"val_seen::{scene_asset}::7",
    )

    class FakeEnv:
        def __init__(self, _config, dataset):
            assert dataset.episodes == [episode]
            self.agent_state = {
                "position": [1.0, 2.0, 3.0],
                "asset": str(scene_asset),
                str(private_root / "state-key"): 1,
            }
            self.last_step_info = None
            self.closed = False

        def reset_to_episode(self, selected):
            assert selected is episode
            return {
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "scene": "data/scene_datasets/Amsterdam-1.tif",
                str(private_root / "sensor-key"): 3,
            }

        def get_metrics(self):
            return {"distance_to_goal": 1.25}

        def step(self, action):
            assert action == "STOP"
            self.last_step_info = {"scene_path": str(scene_asset)}
            return {"rgb": np.ones((1,), dtype=np.uint8)}, True, self.last_step_info

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        capture_env_trace.OmegaConf,
        "load",
        lambda _path: OmegaConf.create(
            {"DATASET": {}, "TASK": {"MEASUREMENTS": []}}
        ),
    )
    monkeypatch.setattr(
        capture_env_trace,
        "SatNavDataset",
        lambda _config: SimpleNamespace(episodes=[episode]),
    )
    monkeypatch.setattr(capture_env_trace, "Env", FakeEnv)
    monkeypatch.setattr(capture_env_trace, "_commit", lambda _repo: "a" * 40)
    output = private_root / "trace.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture_env_trace.py",
            "--config",
            str(config_path),
            "--data-path",
            str(dataset_path),
            "--scenes-dir",
            str(scenes_dir),
            "--split",
            "val_seen",
            "--actions",
            "STOP",
            "--repo",
            str(repo),
            "--output",
            str(output),
        ],
    )

    assert capture_env_trace.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    assert str(private_root) not in serialized
    assert "data/scene_datasets/" not in serialized
    assert payload["source"]["repo"] == {
        "id": "SatNav-checkout",
        "commit": "a" * 40,
    }
    assert payload["source"]["config"] == {
        "id": "task.yaml",
        "sha256": _json_sha256(config_path),
        "size": config_path.stat().st_size,
    }
    assert payload["source"]["dataset"] == {
        "id": "episodes.json",
        "sha256": _json_sha256(dataset_path),
        "size": dataset_path.stat().st_size,
    }
    assert payload["episode"]["scene_id"] == "Amsterdam-1"
    assert payload["episode"]["episode_key"] == "val_seen::Amsterdam-1::7"
    assert payload["steps"][0]["observation"]["scene"] == {
        "id": "Amsterdam-1.tif"
    }
    assert payload["steps"][1]["last_step_info"]["scene_path"] == {
        "id": "Amsterdam-1.tif"
    }


def test_compare_report_uses_trace_identities_and_redacts_path_differences(
    tmp_path, monkeypatch
):
    private_root = tmp_path / "comparison-private-root"
    before_root = private_root / "before-machine"
    after_root = private_root / "after-machine"
    before_root.mkdir(parents=True)
    after_root.mkdir(parents=True)
    before_scene = before_root / "scenes" / "Amsterdam-1.tif"
    after_scene = "Amsterdam-1"

    def trace(scene_id, episode_key, asset):
        return {
            "source": {"split": "val_seen"},
            "episode": {
                "episode_id": "7",
                "scene_id": str(scene_id),
                "episode_key": episode_key,
            },
            "actions_requested": ["STOP"],
            "steps": [
                {
                    "step": 0,
                    "action": None,
                    "observation": {},
                    "agent_state": {
                        "asset": str(asset),
                        str(Path(asset).parent / "private-key"): 1,
                    },
                    "metrics": {},
                    "done": False,
                }
            ],
        }

    before_path = before_root / "before-trace.json"
    after_path = after_root / "after-trace.json"
    before_path.write_text(
        json.dumps(
            trace(
                before_scene,
                f"val_seen::{before_scene}::7",
                before_root / "private-state.bin",
            )
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            trace(
                after_scene,
                "val_seen::Amsterdam-1::7",
                after_root / "changed-state.bin",
            )
        ),
        encoding="utf-8",
    )
    output = private_root / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_env_traces.py",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--output",
            str(output),
        ],
    )

    assert compare_env_traces.main() == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    assert str(private_root) not in serialized
    assert payload["before"] == {
        "id": "before-trace.json",
        "sha256": _json_sha256(before_path),
        "size": before_path.stat().st_size,
    }
    assert payload["after"] == {
        "id": "after-trace.json",
        "sha256": _json_sha256(after_path),
        "size": after_path.stat().st_size,
    }
    difference = next(
        item
        for item in payload["unexpected_differences"]
        if item["path"].endswith(".asset")
    )
    assert difference["before"] == {"id": "private-state.bin"}
    assert difference["after"] == {"id": "changed-state.bin"}
    scene_change = next(
        item
        for item in payload["identity_changes"]
        if item["field"] == "episode.scene_id"
    )
    assert scene_change["before"] == "Amsterdam-1"
    assert scene_change["after"] == "Amsterdam-1"

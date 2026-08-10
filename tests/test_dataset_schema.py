"""Tests for lossless SatNav episode loading and scene resolution."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from satnav.dataset import SatNavDataset, SceneResolver


def episode_payload(episode_id, scene_id):
    return {
        "episode_id": episode_id,
        "trajectory_id": 30,
        "trajectory_type": "Road",
        "trajectory_subtype": "Highway",
        "scene_id": scene_id,
        "start_position": [1.0, 2.0, 50.0],
        "start_rotation": 12.5,
        "goals": [{"position": [3.0, 4.0, 50.0], "radius": 7.0}],
        "instruction": {
            "instruction_text": "go straight",
            "instruction_type": "academic",
            "difficulty_level": 2,
            "language": "en",
        },
        "waypoints": [[1.0, 2.0, 50.0], [3.0, 4.0, 50.0]],
        "reference_path": [[1.0, 2.0, 50.0], [3.0, 4.0, 50.0]],
        "aux_info": {"tile_id": "0_0", "task_num": 5},
        "benchmark_extension": {"preserve": True},
    }


class DatasetSchemaTests(unittest.TestCase):
    def write_dataset(self, directory, episodes):
        path = Path(directory) / "episodes.json"
        path.write_text(
            json.dumps(
                {
                    "instruction_vocab": {"word_list": ["go"]},
                    "benchmark": "v0.1",
                    "episodes": episodes,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loader_preserves_release_fields_and_stable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            scenes_dir = Path(directory) / "scenes"
            scenes_dir.mkdir()
            path = self.write_dataset(
                directory,
                [episode_payload(90, "data/scene_datasets/Amsterdam-1.tif")],
            )
            dataset = SatNavDataset(
                {
                    "DATA_PATH": str(path),
                    "SPLIT": "val_seen",
                    "SCENES_DIR": str(scenes_dir),
                }
            )

            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(dataset.split, "val_seen")
            self.assertEqual(dataset.metadata, {"benchmark": "v0.1"})
            self.assertEqual(dataset.instruction_vocab, {"word_list": ["go"]})

            episode = dataset.episodes[0]
            self.assertEqual(episode.episode_id, "90")
            self.assertEqual(episode.trajectory_id, "30")
            self.assertEqual(episode.scene_id, "Amsterdam-1")
            self.assertEqual(
                episode.scene_path,
                os.path.join(str(scenes_dir), "Amsterdam-1.tif"),
            )
            self.assertEqual(episode.episode_key, "val_seen::Amsterdam-1::90")
            self.assertEqual(episode.trajectory_subtype, "Highway")
            self.assertEqual(episode.waypoints[1], [3.0, 4.0, 50.0])
            self.assertEqual(episode.aux_info["tile_id"], "0_0")
            self.assertEqual(episode.instruction.instruction_type, "academic")
            self.assertEqual(episode.instruction.difficulty_level, 2)
            self.assertEqual(episode.instruction.extras, {"language": "en"})
            self.assertEqual(episode.goals[0].extras, {"radius": 7.0})
            self.assertEqual(
                episode.extras,
                {"benchmark_extension": {"preserve": True}},
            )

            serialized = episode.to_dict()
            self.assertEqual(serialized["scene_id"], "Amsterdam-1")
            self.assertEqual(serialized["instruction"]["language"], "en")
            self.assertEqual(serialized["goals"][0]["radius"], 7.0)
            self.assertTrue(serialized["benchmark_extension"]["preserve"])
            self.assertNotIn("scene_path", serialized)
            self.assertNotIn("episode_key", serialized)
            self.assertNotIn(str(scenes_dir), json.dumps(serialized))

    def test_same_raw_id_in_different_scenes_has_distinct_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    episode_payload(1, "Amsterdam-1"),
                    episode_payload(1, "Rome-1"),
                ],
            )
            dataset = SatNavDataset({"DATA_PATH": str(path), "SPLIT": "train"})
            self.assertEqual(
                {episode.episode_key for episode in dataset.episodes},
                {"train::Amsterdam-1::1", "train::Rome-1::1"},
            )

    def test_absolute_legacy_scene_path_is_separated_from_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path = str(Path(directory) / "LegacyScene.tif")
            path = self.write_dataset(directory, [episode_payload(1, scene_path)])
            dataset = SatNavDataset({"DATA_PATH": str(path), "SPLIT": "test"})
            episode = dataset.episodes[0]
            self.assertEqual(episode.scene_id, "LegacyScene")
            self.assertEqual(episode.scene_path, scene_path)

    def test_scene_resolver_keeps_explicit_relative_paths_compatible(self):
        resolver = SceneResolver("/scene-root")
        self.assertEqual(resolver.resolve("Amsterdam-1"), "/scene-root/Amsterdam-1")
        self.assertEqual(
            resolver.resolve("relative/maps/Amsterdam-1.tif"),
            "relative/maps/Amsterdam-1.tif",
        )
        self.assertEqual(
            resolver.logical_scene_id("relative/maps/Amsterdam-1.tif"),
            "Amsterdam-1",
        )
        self.assertEqual(
            resolver.logical_scene_id("relative/maps/Rome.v2.tif"),
            "Rome.v2",
        )
        self.assertEqual(resolver.logical_scene_id("Rome.v2"), "Rome.v2")


if __name__ == "__main__":
    unittest.main()

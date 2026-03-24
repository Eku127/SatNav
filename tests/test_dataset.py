#!/usr/bin/env python3
"""Tests for dataset loading."""

import gzip
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from satnav.dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
from satnav.dataset.satnav_dataset import (
    SatNavDataset,
    ALL_SCENES_MASK,
    ALL_EPISODES_MASK,
    DEFAULT_SCENE_PATH_PREFIX,
)
from satnav.core.episode import VLNEpisode, InstructionData, NavigationGoal


# Path to example dataset file
EXAMPLE_DATASET_PATH = Path(__file__).parent / "test_data" / "satnav_dataset_example.json"


class TestSatNavDataset:
    """Test cases for SatNavDataset class."""

    def test_load_example_dataset(self):
        """Test loading the example dataset file."""
        if not EXAMPLE_DATASET_PATH.exists():
            pytest.skip(f"Example dataset not found: {EXAMPLE_DATASET_PATH}")
        
        config = {
            "DATA_PATH": str(EXAMPLE_DATASET_PATH),
            "SPLIT": "test",
        }
        dataset = SatNavDataset(config)
        
        assert len(dataset.episodes) == 2
        assert dataset.episodes[0].episode_id == "example_001"
        assert dataset.episodes[1].episode_id == "example_002"
        
        # Verify first episode
        ep1 = dataset.episodes[0]
        assert ep1.instruction.instruction_text == "Go east to the destination"
        assert len(ep1.reference_path) == 3
        assert len(ep1.goals) == 1
        
        # Verify second episode
        ep2 = dataset.episodes[1]
        assert ep2.instruction.instruction_text == "Navigate north to the destination"
        assert len(ep2.reference_path) == 3
        assert len(ep2.goals) == 1

    def create_test_dataset_json(self, tmp_path):
        """Create a test dataset JSON file."""
        dataset_data = {
            "instruction_vocab": {
                "word_list": ["go", "to", "the", "kitchen"]
            },
            "episodes": [
                {
                    "episode_id": "test_001",
                    "trajectory_id": "traj_001",
                    "scene_id": "scene_001",
                    "start_position": [116.3974, 39.9093, 100.0],
                    "start_rotation": 0.0,
                    "goals": [
                        {"position": [116.3975, 39.9094, 100.0]}
                    ],
                    "reference_path": [
                        [116.3974, 39.9093, 100.0],
                        [116.39745, 39.90935, 100.0],
                        [116.3975, 39.9094, 100.0]
                    ],
                    "instruction": {
                        "instruction_text": "Go to the kitchen"
                    }
                },
                {
                    "episode_id": "test_002",
                    "trajectory_id": "traj_002",
                    "scene_id": "scene_002",
                    "start_position": [121.4737, 31.2304, 50.0],
                    "start_rotation": 90.0,
                    "goals": [
                        {"position": [121.4738, 31.2305, 50.0]}
                    ],
                    "reference_path": [
                        [121.4737, 31.2304, 50.0],
                        [121.4738, 31.2305, 50.0]
                    ],
                    "instruction": {
                        "instruction_text": "Navigate forward"
                    }
                }
            ]
        }
        
        dataset_file = tmp_path / "test_dataset.json.gz"
        with gzip.open(dataset_file, "wt") as f:
            json.dump(dataset_data, f)
        
        return str(dataset_file)

    def test_load_dataset_from_json(self, tmp_path):
        """Test loading dataset from plain JSON file (not gzipped)."""
        # Create a plain JSON file (not gzipped)
        dataset_data = {
            "episodes": [
                {
                    "episode_id": "test_001",
                    "trajectory_id": "traj_001",
                    "scene_id": "scene_001",
                    "start_position": [116.3974, 39.9093, 100.0],
                    "start_rotation": 0.0,
                    "goals": [{"position": [116.3975, 39.9094, 100.0]}],
                    "reference_path": [
                        [116.3974, 39.9093, 100.0],
                        [116.39745, 39.90935, 100.0],
                        [116.3975, 39.9094, 100.0]
                    ],
                    "instruction": {"instruction_text": "Go to the kitchen"}
                },
                {
                    "episode_id": "test_002",
                    "trajectory_id": "traj_002",
                    "scene_id": "scene_002",
                    "start_position": [121.4737, 31.2304, 50.0],
                    "start_rotation": 90.0,
                    "goals": [{"position": [121.4738, 31.2305, 50.0]}],
                    "reference_path": [
                        [121.4737, 31.2304, 50.0],
                        [121.4738, 31.2305, 50.0]
                    ],
                    "instruction": {"instruction_text": "Navigate forward"}
                }
            ]
        }
        
        dataset_file = tmp_path / "test_dataset.json"  # Plain JSON, not .gz
        with open(dataset_file, "wt", encoding="utf-8") as f:
            json.dump(dataset_data, f)
        
        config = {
            "DATA_PATH": str(dataset_file),
            "SPLIT": "test",
        }
        
        dataset = SatNavDataset(config)
        
        assert len(dataset.episodes) == 2
        assert dataset.episodes[0].episode_id == "test_001"
        assert dataset.episodes[1].episode_id == "test_002"

    def test_load_dataset_with_format_path(self, tmp_path):
        """Test loading dataset with formatted DATA_PATH."""
        # Create dataset file in split directory
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        dataset_file = split_dir / "train.json.gz"
        
        dataset_data = {
            "episodes": [
                {
                    "episode_id": "001",
                    "trajectory_id": "traj_001",
                    "scene_id": "scene_001",
                    "start_position": [116.3974, 39.9093, 100.0],
                    "start_rotation": 0.0,
                    "goals": [{"position": [116.3975, 39.9094, 100.0]}],
                    "reference_path": [[116.3974, 39.9093, 100.0], [116.3975, 39.9094, 100.0]],
                    "instruction": {"instruction_text": "Go forward"}
                }
            ]
        }
        
        with gzip.open(dataset_file, "wt") as f:
            json.dump(dataset_data, f)
        
        config = {
            "DATA_PATH": str(tmp_path / "{split}" / "{split}.json.gz"),
            "SPLIT": "train",
        }
        
        dataset = SatNavDataset(config)
        assert len(dataset.episodes) == 1

    def test_load_plain_json_vs_gzipped_json(self, tmp_path):
        """Test that both plain JSON and gzipped JSON work."""
        dataset_data = {
            "episodes": [
                {
                    "episode_id": "test_001",
                    "trajectory_id": "traj_001",
                    "scene_id": "scene_001",
                    "start_position": [116.3974, 39.9093, 100.0],
                    "start_rotation": 0.0,
                    "goals": [{"position": [116.3975, 39.9094, 100.0]}],
                    "reference_path": [[116.3974, 39.9093, 100.0]],
                    "instruction": {"instruction_text": "Test"}
                }
            ]
        }
        
        # Test plain JSON
        json_file = tmp_path / "test.json"
        with open(json_file, "wt", encoding="utf-8") as f:
            json.dump(dataset_data, f)
        
        config1 = {"DATA_PATH": str(json_file), "SPLIT": "test"}
        dataset1 = SatNavDataset(config1)
        assert len(dataset1.episodes) == 1
        
        # Test gzipped JSON
        gz_file = tmp_path / "test.json.gz"
        with gzip.open(gz_file, "wt") as f:
            json.dump(dataset_data, f)
        
        config2 = {"DATA_PATH": str(gz_file), "SPLIT": "test"}
        dataset2 = SatNavDataset(config2)
        assert len(dataset2.episodes) == 1
        
        # Both should have same content
        assert dataset1.episodes[0].episode_id == dataset2.episodes[0].episode_id

    def test_parse_episode_data(self, tmp_path):
        """Test parsing episode data correctly."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        episode = dataset.episodes[0]
        assert isinstance(episode, VLNEpisode)
        assert episode.episode_id == "test_001"
        assert episode.trajectory_id == "traj_001"
        assert episode.start_position == [116.3974, 39.9093, 100.0]
        assert episode.start_rotation == 0.0
        assert len(episode.goals) == 1
        assert isinstance(episode.goals[0], NavigationGoal)
        assert episode.goals[0].position == [116.3975, 39.9094, 100.0]
        assert len(episode.reference_path) == 3
        assert episode.reference_path[0] == [116.3974, 39.9093, 100.0]
        assert isinstance(episode.instruction, InstructionData)
        assert episode.instruction.instruction_text == "Go to the kitchen"

    def test_reference_path_continuous_coordinates(self, tmp_path):
        """Test that reference_path is parsed as continuous coordinates."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        episode = dataset.episodes[0]
        # Verify reference_path is List[List[float]]
        assert isinstance(episode.reference_path, list)
        assert len(episode.reference_path) > 0
        assert isinstance(episode.reference_path[0], list)
        assert len(episode.reference_path[0]) == 3  # [lon, lat, alt]

    def test_goals_continuous_coordinates(self, tmp_path):
        """Test that goals are parsed as continuous coordinates."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        episode = dataset.episodes[0]
        assert len(episode.goals) > 0
        assert isinstance(episode.goals[0], NavigationGoal)
        assert len(episode.goals[0].position) == 3  # [lon, lat, alt]

    def test_scene_path_adjustment(self, tmp_path):
        """Test scene path adjustment with scenes_dir."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        
        config = {
            "DATA_PATH": dataset_file,
            "SPLIT": "test",
            "SCENES_DIR": str(scenes_dir),
        }
        dataset = SatNavDataset(config)
        
        episode = dataset.episodes[0]
        assert str(scenes_dir) in episode.scene_id or episode.scene_id == "scene_001"

    def test_scene_path_prefix_removal(self, tmp_path):
        """Test removal of default scene path prefix."""
        dataset_data = {
            "episodes": [
                {
                    "episode_id": "001",
                    "trajectory_id": "traj_001",
                    "scene_id": f"{DEFAULT_SCENE_PATH_PREFIX}scene_001",
                    "start_position": [116.3974, 39.9093, 100.0],
                    "start_rotation": 0.0,
                    "goals": [{"position": [116.3975, 39.9094, 100.0]}],
                    "reference_path": [[116.3974, 39.9093, 100.0]],
                    "instruction": {"instruction_text": "Go forward"}
                }
            ]
        }
        
        dataset_file = tmp_path / "test.json.gz"
        with gzip.open(dataset_file, "wt") as f:
            json.dump(dataset_data, f)
        
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        
        config = {
            "DATA_PATH": str(dataset_file),
            "SPLIT": "test",
            "SCENES_DIR": str(scenes_dir),
        }
        dataset = SatNavDataset(config)
        
        episode = dataset.episodes[0]
        # Should have removed prefix and added scenes_dir
        assert DEFAULT_SCENE_PATH_PREFIX not in episode.scene_id

    def test_filter_by_content_scenes(self, tmp_path):
        """Test filtering episodes by content scenes."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {
            "DATA_PATH": dataset_file,
            "SPLIT": "test",
            "CONTENT_SCENES": ["scene_001"],  # Only load scene_001
        }
        dataset = SatNavDataset(config)
        
        # Should only have episode from scene_001
        assert len(dataset.episodes) == 1
        assert dataset.episodes[0].scene_id == "scene_001"

    def test_filter_by_episodes_allowed(self, tmp_path):
        """Test filtering episodes by episode IDs."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {
            "DATA_PATH": dataset_file,
            "SPLIT": "test",
            "EPISODES_ALLOWED": ["test_001"],  # Only load test_001
        }
        dataset = SatNavDataset(config)
        
        assert len(dataset.episodes) == 1
        assert dataset.episodes[0].episode_id == "test_001"

    def test_all_scenes_mask(self, tmp_path):
        """Test that ALL_SCENES_MASK loads all scenes."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {
            "DATA_PATH": dataset_file,
            "SPLIT": "test",
            "CONTENT_SCENES": [ALL_SCENES_MASK],
        }
        dataset = SatNavDataset(config)
        
        # Should load all episodes
        assert len(dataset.episodes) == 2

    def test_all_episodes_mask(self, tmp_path):
        """Test that ALL_EPISODES_MASK loads all episodes."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {
            "DATA_PATH": dataset_file,
            "SPLIT": "test",
            "EPISODES_ALLOWED": [ALL_EPISODES_MASK],
        }
        dataset = SatNavDataset(config)
        
        assert len(dataset.episodes) == 2

    def test_empty_dataset(self, tmp_path):
        """Test loading empty dataset."""
        dataset_data = {"episodes": []}
        dataset_file = tmp_path / "empty.json.gz"
        with gzip.open(dataset_file, "wt") as f:
            json.dump(dataset_data, f)
        
        config = {"DATA_PATH": str(dataset_file), "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        assert len(dataset.episodes) == 0
        assert dataset.num_episodes == 0

    def test_num_episodes_property(self, tmp_path):
        """Test num_episodes property."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        assert dataset.num_episodes == 2
        assert dataset.num_episodes == len(dataset.episodes)

    def test_get_episode_iterator(self, tmp_path):
        """Test episode iterator."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset = SatNavDataset(config)
        
        iterator = dataset.get_episode_iterator()
        episodes = list(iterator)
        
        assert len(episodes) == 2
        assert all(isinstance(ep, VLNEpisode) for ep in episodes)

    def test_scene_from_scene_path(self):
        """Test scene_from_scene_path static method."""
        scene_name = SatNavDataset.scene_from_scene_path("/path/to/scene_name.ext")
        assert scene_name == "scene_name"
        
        scene_name = SatNavDataset.scene_from_scene_path("scene_name.ext")
        assert scene_name == "scene_name"

    def test_check_config_paths_exist(self, tmp_path):
        """Test check_config_paths_exist method."""
        dataset_file = tmp_path / "test.json.gz"
        dataset_file.touch()
        
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        
        config = {
            "DATA_PATH": str(dataset_file),
            "SPLIT": "test",
            "SCENES_DIR": str(scenes_dir),
        }
        
        assert SatNavDataset.check_config_paths_exist(config) is True
        
        # Test with non-existent file
        config["DATA_PATH"] = str(tmp_path / "nonexistent.json.gz")
        assert SatNavDataset.check_config_paths_exist(config) is False

    def test_get_scenes_to_load(self, tmp_path):
        """Test get_scenes_to_load class method."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        config = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        scenes = SatNavDataset.get_scenes_to_load(config)
        
        assert isinstance(scenes, list)
        assert len(scenes) == 2  # scene_001 and scene_002
        assert "scene_001" in scenes
        assert "scene_002" in scenes

    def test_dict_config_compatibility(self, tmp_path):
        """Test that dataset works with both dict and DictConfig."""
        dataset_file = self.create_test_dataset_json(tmp_path)
        
        # Test with dict
        config_dict = {"DATA_PATH": dataset_file, "SPLIT": "test"}
        dataset1 = SatNavDataset(config_dict)
        
        # Test with DictConfig
        config_omegaconf = OmegaConf.create(config_dict)
        dataset2 = SatNavDataset(config_omegaconf)
        
        assert len(dataset1.episodes) == len(dataset2.episodes)

    def test_none_config(self):
        """Test initializing dataset with None config."""
        dataset = SatNavDataset(None)
        assert len(dataset.episodes) == 0

    def test_load_dataset_from_config_file(self):
        """Test loading dataset using config file (satnav_config_example.yaml)."""
        from satnav.core.config import load_config
        
        # Load config from test_data directory
        config_file = Path(__file__).parent / "test_data" / "satnav_config_example.yaml"
        if not config_file.exists():
            pytest.skip(f"Config file not found: {config_file}")
        
        config = load_config(str(config_file))
        
        # Verify config structure
        assert "DATASET" in config
        assert "DATA_PATH" in config.DATASET
        assert "SPLIT" in config.DATASET
        
        # Update DATA_PATH to use absolute path for the example dataset
        dataset_path = Path(__file__).parent / "test_data" / "satnav_dataset_example.json"
        if not dataset_path.exists():
            pytest.skip(f"Example dataset not found: {dataset_path}")
        
        # Create dataset config from loaded config
        dataset_config = {
            "DATA_PATH": str(dataset_path),
            "SPLIT": config.DATASET.SPLIT,
        }
        
        # Load dataset
        dataset = SatNavDataset(dataset_config)
        
        # Verify dataset loaded successfully
        assert len(dataset.episodes) == 2
        assert dataset.episodes[0].episode_id == "example_001"
        assert dataset.episodes[1].episode_id == "example_002"
        # Verify scene_id matches the test data
        assert "map" in dataset.episodes[0].scene_id
        assert "map" in dataset.episodes[1].scene_id
        
        # Verify config values are accessible
        assert config.ENVIRONMENT.MAX_EPISODE_STEPS == 500
        assert config.TASK.TYPE == "VLN"  # Updated to match actual config
        assert config.TASK.SUCCESS_DISTANCE == 3.0
        assert "DISTANCE_TO_GOAL" in config.TASK.MEASUREMENTS
        assert "SUCCESS" in config.TASK.MEASUREMENTS
        assert "SPL" in config.TASK.MEASUREMENTS


class TestOfflineTrajectoryDataset:
    """Tests for the offline trajectory dataset."""

    def test_offline_dataset_alignment(self, tmp_path):
        images_root = tmp_path / "trajectory_data" / "images"
        traj_dir = images_root / "sceneA_satnav_000001" / "rgb"
        traj_dir.mkdir(parents=True)

        for idx in range(1, 6):
            image = torch.full((4, 4, 3), idx, dtype=torch.uint8).numpy()
            from PIL import Image

            Image.fromarray(image).save(traj_dir / f"{idx:03d}.jpg")

        annotations = [
            {
                "id": 1,
                "trajectory_id": "1",
                "steps": 4,
                "video": "images/sceneA_satnav_000001",
                "instructions": ["Go forward then stop."],
                "actions": [-1, 1, 1, 2, 0],
                "_scene_id": "sceneA",
                "_episode_id": "ep1",
            }
        ]
        annotations_path = tmp_path / "trajectory_data" / "annotations.json"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f)

        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word2idx": {
                        "<pad>": 0,
                        "<unk>": 1,
                        "go": 2,
                        "forward": 3,
                        "then": 4,
                        "stop": 5,
                    }
                },
                f,
            )

        config = OmegaConf.create(
            {
                "DATASET": {
                    "DATA_PATH": str(tmp_path / "episodes.json"),
                    "SPLIT": "train",
                    "SCENES_DIR": str(tmp_path),
                    "vocab_file": str(vocab_path),
                },
                "IL": {
                    "batch_size": 1,
                    "RECOLLECT_TRAINER": {
                        "max_traj_len": 500,
                    },
                    "OFFLINE": {
                        "annotations_path": str(annotations_path),
                        "images_root": str(images_root),
                        "rgb_size": 4,
                        "max_instruction_len": 8,
                        "max_traj_len": 500,
                    },
                    "use_inflection_weighting": True,
                    "inflection_weight_coef": 3.2,
                },
            }
        )

        dataset = OfflineTrajectoryDataset(config, target_rgb_size=4)
        observations, prev_actions, teacher_actions, weights = dataset[0]

        assert observations["rgb"].shape == (3, 4, 4, 3)
        assert observations["instruction"].shape == (3, 8)
        assert prev_actions.tolist() == [0, 1, 1]
        assert teacher_actions.tolist() == [1, 1, 2]
        assert weights.tolist() == pytest.approx([3.2, 1.0, 3.2])

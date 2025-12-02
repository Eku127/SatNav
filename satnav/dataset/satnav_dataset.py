#!/usr/bin/env python3
"""Dataset loader for SatNav VLN tasks in continuous space."""

import gzip
import json
import os
from pathlib import Path
from typing import List, Optional, Set, Union

from omegaconf import DictConfig

from satnav.core.episode import (
    InstructionData,
    NavigationGoal,
    VLNEpisode,
)


# Constants for filtering
ALL_SCENES_MASK = "*"
ALL_EPISODES_MASK = "*"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


class SatNavDataset:
    """Dataset loader for SatNav VLN tasks in continuous space.
    
    This class loads VLN episodes from gzipped JSON files, supporting continuous
    space navigation with geographic coordinates (longitude, latitude, altitude)
    and continuous reference paths.
    
    Attributes:
        episodes: List of VLNEpisode objects loaded from the dataset.
    """
    
    def __init__(self, config: Optional[Union[DictConfig, dict]] = None) -> None:
        """Initialize the dataset.
        
        Args:
            config: Configuration dictionary containing:
                - DATA_PATH: Path template for dataset file (e.g., "data/datasets/{split}/{split}.json.gz")
                - SPLIT: Dataset split name (e.g., "train", "val_seen")
                - SCENES_DIR: Directory containing scene data (optional)
                - CONTENT_SCENES: List of scene names to load, or ["*"] for all (optional)
                - EPISODES_ALLOWED: List of episode IDs to load, or ["*"] for all (optional)
        """
        self.episodes: List[VLNEpisode] = []
        
        if config is None:
            return
        
        # Handle both DictConfig and dict
        if isinstance(config, DictConfig):
            data_path = config.DATA_PATH
            split = config.SPLIT
            scenes_dir = getattr(config, "SCENES_DIR", None)
            content_scenes = getattr(config, "CONTENT_SCENES", [ALL_SCENES_MASK])
            episodes_allowed = getattr(config, "EPISODES_ALLOWED", [ALL_EPISODES_MASK])
        else:
            data_path = config.get("DATA_PATH")
            split = config.get("SPLIT")
            scenes_dir = config.get("SCENES_DIR")
            content_scenes = config.get("CONTENT_SCENES", [ALL_SCENES_MASK])
            episodes_allowed = config.get("EPISODES_ALLOWED", [ALL_EPISODES_MASK])
        
        # Load dataset file
        # Handle both format strings and direct paths
        try:
            dataset_filename = data_path.format(split=split)
        except (KeyError, AttributeError):
            # If format fails, use data_path directly (might be absolute path)
            dataset_filename = data_path
        
        if not os.path.exists(dataset_filename):
            raise FileNotFoundError(f"Dataset file not found: {dataset_filename}")
        
        # Support both .json.gz and .json files for debugging
        if dataset_filename.endswith('.gz'):
            # Gzipped JSON file
            with gzip.open(dataset_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)
        elif dataset_filename.endswith('.json'):
            # Plain JSON file (for debugging)
            with open(dataset_filename, "rt", encoding="utf-8") as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)
        else:
            # Try to detect file type by attempting to open as gzip first
            try:
                with gzip.open(dataset_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=scenes_dir)
            except (gzip.BadGzipFile, OSError):
                # Fall back to plain JSON
                with open(dataset_filename, "rt", encoding="utf-8") as f:
                    self.from_json(f.read(), scenes_dir=scenes_dir)
        
        # Filter by scenes if specified
        if ALL_SCENES_MASK not in content_scenes:
            scenes_to_load = set(content_scenes)
            self.episodes = [
                e for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]
        
        # Filter by episode IDs if specified
        if ALL_EPISODES_MASK not in episodes_allowed:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(episodes_allowed)
            self.episodes = [
                episode for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]
    
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        """Load episodes from JSON string.
        
        Args:
            json_str: JSON string containing dataset episodes.
            scenes_dir: Optional directory path to prepend to scene_id paths.
        """
        deserialized = json.loads(json_str)
        
        # Handle instruction_vocab if present (optional, for compatibility)
        if "instruction_vocab" in deserialized:
            # Store vocab if needed, but not required for minimal implementation
            pass
        
        for episode_data in deserialized["episodes"]:
            # Convert IDs to strings for consistency
            episode_data["episode_id"] = str(episode_data.get("episode_id", ""))
            episode_data["trajectory_id"] = str(episode_data.get("trajectory_id", ""))
            
            # Parse instruction
            instruction_data = episode_data.get("instruction", {})
            if isinstance(instruction_data, dict):
                instruction = InstructionData(
                    instruction_text=instruction_data.get("instruction_text", "")
                )
            else:
                instruction = instruction_data
            
            # Parse goals (continuous space coordinates)
            goals_data = episode_data.get("goals", [])
            goals = []
            if goals_data is not None:
                for goal_data in goals_data:
                    if isinstance(goal_data, dict):
                        goal = NavigationGoal(position=goal_data.get("position", []))
                    else:
                        goal = goal_data
                    goals.append(goal)
            
            # Parse reference_path (continuous path: List[List[float]])
            reference_path = episode_data.get("reference_path", [])
            if reference_path is None:
                reference_path = []
            
            # Handle scene_id path adjustment
            scene_id = episode_data.get("scene_id", "")
            if scenes_dir is not None and scene_id:
                if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    scene_id = scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
                scene_id = os.path.join(scenes_dir, scene_id)
            
            # Create VLNEpisode
            episode = VLNEpisode(
                episode_id=episode_data["episode_id"],
                scene_id=scene_id,
                start_position=episode_data.get("start_position", [0.0, 0.0, 0.0]),
                start_rotation=episode_data.get("start_rotation", 0.0),
                goals=goals,
                reference_path=reference_path,
                instruction=instruction,
                trajectory_id=episode_data["trajectory_id"],
            )
            
            self.episodes.append(episode)
    
    @staticmethod
    def scene_from_scene_path(scene_path: str) -> str:
        """Extract scene name from scene path.
        
        Args:
            scene_path: Path to scene file (e.g., "/path/to/scene_name.ext").
            
        Returns:
            Scene name without extension.
        """
        return os.path.splitext(os.path.basename(scene_path))[0]
    
    @classmethod
    def check_config_paths_exist(cls, config: Union[DictConfig, dict]) -> bool:
        """Check if dataset and scene paths exist.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            True if all required paths exist, False otherwise.
        """
        if isinstance(config, DictConfig):
            data_path = config.DATA_PATH
            split = config.SPLIT
            scenes_dir = getattr(config, "SCENES_DIR", None)
        else:
            data_path = config.get("DATA_PATH")
            split = config.get("SPLIT")
            scenes_dir = config.get("SCENES_DIR")
        
        dataset_file = data_path.format(split=split)
        if not os.path.exists(dataset_file):
            return False
        
        if scenes_dir is not None and not os.path.exists(scenes_dir):
            return False
        
        return True
    
    @classmethod
    def get_scenes_to_load(cls, config: Union[DictConfig, dict]) -> List[str]:
        """Get list of scene names that would be loaded.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Sorted list of unique scene names.
        """
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        scenes = {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        return sorted(scenes)
    
    @property
    def num_episodes(self) -> int:
        """Get number of episodes in the dataset."""
        return len(self.episodes)
    
    def get_episode_iterator(self, **kwargs):
        """Get iterator over episodes.
        
        Args:
            **kwargs: Additional arguments (for compatibility, currently unused).
            
        Returns:
            Iterator over episodes.
        """
        return iter(self.episodes)


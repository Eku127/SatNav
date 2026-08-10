#!/usr/bin/env python3
"""Dataset loader for SatNav VLN tasks in continuous space."""

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from satnav.core.episode import (
    InstructionData,
    NavigationGoal,
    VLNEpisode,
)
from satnav.dataset.scene_resolver import SceneResolver


# Constants for filtering
ALL_SCENES_MASK = "*"
ALL_EPISODES_MASK = "*"

_KNOWN_EPISODE_FIELDS = {
    "episode_id",
    "trajectory_id",
    "trajectory_type",
    "trajectory_subtype",
    "scene_id",
    "start_position",
    "start_rotation",
    "goals",
    "instruction",
    "waypoints",
    "reference_path",
    "aux_info",
}
_KNOWN_INSTRUCTION_FIELDS = {
    "instruction_text",
    "instruction_type",
    "difficulty_level",
}


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
        self.split: Optional[str] = None
        self.scenes_dir: Optional[str] = None
        self.scene_resolver = SceneResolver()
        self.instruction_vocab: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
        
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

        if data_path is None:
            raise ValueError("DATASET.DATA_PATH must be configured")

        self.split = str(split) if split is not None else None
        self.scenes_dir = str(scenes_dir) if scenes_dir is not None else None
        self.scene_resolver = SceneResolver(self.scenes_dir)
        content_scenes = content_scenes or [ALL_SCENES_MASK]
        episodes_allowed = episodes_allowed or [ALL_EPISODES_MASK]
        
        # Load dataset file
        # Handle both format strings and direct paths
        try:
            dataset_filename = str(data_path).format(split=split)
        except (KeyError, AttributeError):
            # If format fails, use data_path directly (might be absolute path)
            dataset_filename = str(data_path)
        
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
            scenes_to_load = {
                self.scene_from_scene_path(scene) for scene in content_scenes
            }
            self.episodes = [
                e for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]
        
        # Filter by episode IDs if specified
        if ALL_EPISODES_MASK not in episodes_allowed:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - {
                str(episode_id) for episode_id in episodes_allowed
            }
            self.episodes = [
                episode for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]
    
    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        """Load episodes from JSON string.
        
        Args:
            json_str: JSON string containing dataset episodes.
            scenes_dir: Optional local scene root.  This resolves
                ``episode.scene_path`` without changing logical ``scene_id``.
            split: Optional split override used for stable episode keys.
        """
        deserialized = json.loads(json_str)

        if not isinstance(deserialized, dict) or not isinstance(
            deserialized.get("episodes"), list
        ):
            raise ValueError("Dataset JSON must contain an 'episodes' list")

        if split is not None:
            self.split = str(split)
        if scenes_dir is not None:
            self.scenes_dir = str(scenes_dir)
            self.scene_resolver = SceneResolver(self.scenes_dir)

        self.instruction_vocab = deserialized.get("instruction_vocab")
        self.metadata = {
            key: value
            for key, value in deserialized.items()
            if key not in {"episodes", "instruction_vocab"}
        }
        
        for episode_data in deserialized["episodes"]:
            if not isinstance(episode_data, dict):
                raise ValueError("Every episode entry must be a JSON object")

            # Convert IDs to strings for compatibility with existing consumers.
            episode_id = str(episode_data.get("episode_id", ""))
            trajectory_id = str(episode_data.get("trajectory_id", ""))
            
            # Parse instruction
            instruction_data = episode_data.get("instruction", {})
            if isinstance(instruction_data, dict):
                instruction = InstructionData(
                    instruction_text=str(
                        instruction_data.get("instruction_text", "")
                    ),
                    instruction_type=instruction_data.get("instruction_type"),
                    difficulty_level=instruction_data.get("difficulty_level"),
                    extras={
                        key: value
                        for key, value in instruction_data.items()
                        if key not in _KNOWN_INSTRUCTION_FIELDS
                    },
                )
            elif isinstance(instruction_data, InstructionData):
                instruction = instruction_data
            else:
                instruction = InstructionData(
                    instruction_text=str(instruction_data or "")
                )
            
            # Parse goals (continuous space coordinates)
            goals_data = episode_data.get("goals", [])
            goals = []
            if goals_data is not None:
                for goal_data in goals_data:
                    if isinstance(goal_data, dict):
                        goal = NavigationGoal(
                            position=goal_data.get("position", []),
                            extras={
                                key: value
                                for key, value in goal_data.items()
                                if key != "position"
                            },
                        )
                    elif isinstance(goal_data, NavigationGoal):
                        goal = goal_data
                    else:
                        goal = NavigationGoal(position=goal_data)
                    goals.append(goal)
            
            # Parse reference_path (continuous path: List[List[float]])
            reference_path = episode_data.get("reference_path", [])
            if reference_path is None:
                reference_path = []
            
            raw_scene_reference = str(episode_data.get("scene_id", ""))
            scene_id = self.scene_resolver.logical_scene_id(raw_scene_reference)
            scene_path = (
                self.scene_resolver.resolve(raw_scene_reference)
                if raw_scene_reference
                else None
            )
            
            # Create VLNEpisode
            episode = VLNEpisode(
                episode_id=episode_id,
                scene_id=scene_id,
                start_position=episode_data.get("start_position", [0.0, 0.0, 0.0]),
                start_rotation=episode_data.get("start_rotation", 0.0),
                goals=goals,
                reference_path=reference_path,
                instruction=instruction,
                trajectory_id=trajectory_id,
                trajectory_type=episode_data.get("trajectory_type"),
                trajectory_subtype=episode_data.get("trajectory_subtype"),
                waypoints=episode_data.get("waypoints") or [],
                aux_info=episode_data.get("aux_info") or {},
                split=self.split,
                scene_path=scene_path,
                extras={
                    key: value
                    for key, value in episode_data.items()
                    if key not in _KNOWN_EPISODE_FIELDS
                },
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
        return SceneResolver.logical_scene_id(scene_path)

    def resolve_scene_path(self, scene: Union[str, VLNEpisode]) -> str:
        """Resolve an episode or logical scene ID to its local asset path."""
        if isinstance(scene, VLNEpisode):
            if scene.scene_path:
                return scene.scene_path
            scene = scene.scene_id
        return self.scene_resolver.resolve(scene)
    
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

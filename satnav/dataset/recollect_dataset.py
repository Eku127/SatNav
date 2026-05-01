#!/usr/bin/env python3
"""Recollection dataset for real-time trajectory collection.

This module provides a dataset class that collects trajectories from
the environment in real-time using teacher forcing with reference paths.

This is retained for future recollection / online imitation work. The current
release training path for Seq2Seq and CMA uses OfflineTrajectoryDataset with
OfflineTrainer instead.

Reference: VLN-CE vlnce_baselines/common/recollection_dataset.py
"""

import copy
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from satnav.core.config import get_success_distance_default
from satnav.navigation import ReferencePathFollower
from satnav.task.actions import Action
from satnav.training.distributed import get_rank, get_world_size
from satnav.utils.build_vocab import build_vocab_from_dataset, VocabDict


class RecollectionDataset(torch.utils.data.IterableDataset):
    """Dataset that collects trajectories in real-time using teacher forcing.
    
    This dataset:
    1. Extracts GT action sequences from reference_path using ReferencePathFollower
    2. Executes GT actions in the environment to collect observations
    3. Tokenizes instructions using vocabulary
    4. Resizes RGB images to 224x224 to reduce GPU memory usage
    5. Returns training samples: (observations, prev_actions, teacher_actions)
    
    The dataset automatically resizes RGB images to 224x224 during collection to:
    - Reduce CPU RAM usage in preload buffer (~5x reduction)
    - Reduce GPU memory usage during training (~4-5x reduction)
    - Speed up CPU→GPU data transfer
    - Match ResNet50's optimal input size (ImageNet pretrained)
    
    Attributes:
        config: Training configuration
        env: SatNav environment
        path_follower: ReferencePathFollower for GT trajectory extraction
        vocab: Vocabulary for instruction tokenization
        trajectories: Dict mapping episode_id to action sequences
        batch_size: Batch size for training
        target_rgb_size: Target size for RGB images (default: 224)
        observation_space: Observation space description
        action_space: Action space description
    """
    
    def __init__(self, config: DictConfig, target_rgb_size: int = 224):
        """Initialize the recollection dataset.
        
        Args:
            config: Training configuration containing:
                - DATASET config for environment
                - IL.batch_size for batch size
                - IL.RECOLLECT_TRAINER.preload_size for buffer size
                - IL.RECOLLECT_TRAINER.max_traj_len for trajectory length limit
            target_rgb_size: Target size for RGB images (default: 224).
                Images are resized to this size during collection to reduce memory usage.
        """
        super().__init__()
        self.config = config
        self.target_rgb_size = target_rgb_size
        self._preload = deque()
        self.rank = get_rank(config)
        self.world_size = get_world_size(config)
        
        # Filter out TOP_DOWN_MAP from measurements during training for performance
        # This avoids the overhead of updating the map on every step during data collection
        training_config = self._filter_topdown_map_for_training(config)
        
        # Create environment with cycle=True for training
        # Import here to avoid circular import (Env imports SatNavDataset)
        from satnav.core.env import Env
        print(
            "Creating environment..."
            f" rank={self.rank}"
            f" world_size={self.world_size}"
        )
        self.env = Env(training_config, cycle=True)
        
        # Initialize ReferencePathFollower
        # Get parameters from config or use defaults
        # goal_radius comes from TASK.SUCCESS_DISTANCE (supports both old and new format)
        goal_radius = get_success_distance_default(config, default=10.0)
        if isinstance(config, dict):
            sim_config = config.get("SIMULATOR", {})
            turn_angle = sim_config.get("TURN_ANGLE", 15.0) if isinstance(sim_config, dict) else getattr(sim_config, "TURN_ANGLE", 15.0)
        else:
            sim_config = getattr(config, "SIMULATOR", {})
            turn_angle = getattr(sim_config, "TURN_ANGLE", 15.0)
        
        print(f"ReferencePathFollower parameters: goal_radius={goal_radius}m, turn_angle={turn_angle}°")
        print(f"Target RGB size: {self.target_rgb_size}x{self.target_rgb_size} (resizing enabled for memory efficiency)")
        
        self.path_follower = ReferencePathFollower(
            goal_radius=goal_radius,
            turn_angle=turn_angle
        )
        
        # Load or build vocabulary
        print("Loading vocabulary...")
        self.vocab = self._load_vocabulary()
        self.max_instruction_len = 200  # Maximum instruction length

        all_episode_indices = list(range(len(self.env._dataset.episodes)))
        self._episode_indices = all_episode_indices[self.rank::self.world_size]
        self._current_episode_idx = 0
        print(
            f"Episode shard for rank {self.rank}: "
            f"{len(self._episode_indices)}/{len(all_episode_indices)} episodes"
        )
        
        # Extract GT trajectories from all episodes
        print("Extracting GT trajectories from reference paths...")
        self.trajectories = self._extract_trajectories()
        print(f"Extracted {len(self.trajectories)} trajectories on rank {self.rank}")
        
        # Get observation and action spaces from environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Batch size
        self.batch_size = config.IL.batch_size
        
        # Inflection weighting: weight timesteps where action changes
        # This helps the model learn critical decision points
        if isinstance(config, dict):
            il_config = config.get('IL', {})
            self.use_inflection_weighting = il_config.get('use_inflection_weighting', False)
            self.inflection_weight_coef = il_config.get('inflection_weight_coef', 3.2)
        else:
            il_config = getattr(config, 'IL', {})
            self.use_inflection_weighting = getattr(il_config, 'use_inflection_weighting', False)
            self.inflection_weight_coef = getattr(il_config, 'inflection_weight_coef', 3.2)
        
        if self.use_inflection_weighting:
            self.inflec_weights = torch.tensor([1.0, self.inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        
    def _filter_topdown_map_for_training(self, config: DictConfig) -> DictConfig:
        """Filter out TOP_DOWN_MAP from measurements during training for performance.
        
        This method creates a modified copy of the config that excludes TOP_DOWN_MAP
        from the MEASUREMENTS list. This significantly speeds up data collection during
        training since TOP_DOWN_MAP updates are computationally expensive.
        
        Args:
            config: Original configuration object
            
        Returns:
            Modified configuration with TOP_DOWN_MAP filtered out
        """
        if isinstance(config, DictConfig):
            # Make a mutable copy
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Filter out TOP_DOWN_MAP if present
            if "TASK" in config_dict and "MEASUREMENTS" in config_dict["TASK"]:
                measurements = config_dict["TASK"]["MEASUREMENTS"]
                if isinstance(measurements, list):
                    original_count = len(measurements)
                    config_dict["TASK"]["MEASUREMENTS"] = [
                        m for m in measurements if m.upper() != "TOP_DOWN_MAP"
                    ]
                    filtered_count = len(config_dict["TASK"]["MEASUREMENTS"])
                    
                    if original_count > filtered_count:
                        print("Note: TOP_DOWN_MAP automatically disabled during training for faster preload")
                        print(f"  (filtered from {original_count} to {filtered_count} measurements)")
            
            return OmegaConf.create(config_dict)
        else:
            # Dict format - make a deep copy
            config_dict = copy.deepcopy(config)
            
            # Filter out TOP_DOWN_MAP if present
            if "TASK" in config_dict and "MEASUREMENTS" in config_dict["TASK"]:
                measurements = config_dict["TASK"]["MEASUREMENTS"]
                if isinstance(measurements, list):
                    original_count = len(measurements)
                    config_dict["TASK"]["MEASUREMENTS"] = [
                        m for m in measurements if m.upper() != "TOP_DOWN_MAP"
                    ]
                    filtered_count = len(config_dict["TASK"]["MEASUREMENTS"])
                    
                    if original_count > filtered_count:
                        print("Note: TOP_DOWN_MAP automatically disabled during training for faster preload")
                        print(f"  (filtered from {original_count} to {filtered_count} measurements)")
            
            return config_dict
    
    def _load_vocabulary(self) -> VocabDict:
        """Load or build vocabulary from dataset.
        
        Returns:
            VocabDict instance
        """
        # Check if vocabulary file is specified in config
        vocab_file = getattr(self.config.DATASET, 'vocab_file', None)
        
        if vocab_file:
            # Load from file
            print(f"Loading vocabulary from: {vocab_file}")
            vocab = VocabDict.load(vocab_file)
        else:
            # Build from dataset
            dataset_path = self.config.DATASET.DATA_PATH.format(
                split=self.config.DATASET.SPLIT
            )
            print(f"Building vocabulary from dataset: {dataset_path}")
            vocab = build_vocab_from_dataset(dataset_path)
        
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def _resize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Resize RGB image to target size.
        
        This method resizes RGB images to reduce memory usage. The default target
        size is 224x224, which matches ResNet50's optimal input size and reduces
        GPU memory usage by ~4-5x compared to 512x512 images.
        
        Args:
            rgb: RGB tensor of shape (H, W, 3) or (H, W, C)
            
        Returns:
            Resized RGB tensor of shape (target_rgb_size, target_rgb_size, 3)
        """
        if rgb.shape[0] == self.target_rgb_size and rgb.shape[1] == self.target_rgb_size:
            return rgb
        
        # HWC -> CHW -> NCHW
        rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
        
        # Resize (convert to float, resize, convert back to uint8)
        rgb_resized = F.interpolate(
            rgb_chw.float(),
            size=(self.target_rgb_size, self.target_rgb_size),
            mode='bilinear',
            align_corners=False
        )
        
        # NCHW -> CHW -> HWC
        rgb_resized = rgb_resized.squeeze(0).permute(1, 2, 0)
        
        # Convert back to uint8
        rgb_resized = rgb_resized.clamp(0, 255).to(torch.uint8)
        
        return rgb_resized
    
    def _extract_trajectories(self) -> Dict[str, List[Tuple[int, int]]]:
        """Extract GT action sequences from all episodes.
        
        Uses ReferencePathFollower to convert reference_path to discrete actions.
        
        Returns:
            Dict mapping episode_id to list of (prev_action, action) tuples
        """
        trajectories = {}
        
        for episode_idx in self._episode_indices:
            episode = self.env._dataset.episodes[episode_idx]
            # Reset environment to this episode's start
            # Note: Since env cycles through episodes, we need to reset until we get this one
            # For now, we'll directly use the simulator
            
            # Get reference path
            reference_path = episode.reference_path
            
            if not reference_path or len(reference_path) < 2:
                print(f"Warning: Episode {episode.episode_id} has invalid reference_path, skipping")
                continue
            
            # Reset simulator to episode start
            try:
                self.env._sim.reset(episode.scene_id)
                self.env._sim.set_agent_state(
                    position=episode.start_position,
                    rotation=episode.start_rotation
                )
            except ValueError as e:
                error_msg = str(e)
                if "Camera view bounds exceed" in error_msg:
                    print(f"Warning: Episode {episode.episode_id} start position too close to map edge, skipping")
                else:
                    print(f"Warning: Episode {episode.episode_id} reset failed: {e}, skipping")
                continue
            
            # Use ReferencePathFollower to generate action sequence
            try:
                actions = self.path_follower.follow_path(
                    reference_path=reference_path,
                    simulator=self.env._sim,
                    execute=True  # Execute actions in simulator
                )

                # Convert action strings to indices and create (prev_action, action) pairs
                traj = []
                prev_action = 0  # STOP
                for action_str in actions:
                    if action_str == Action.STOP:
                        # Don't include STOP in training trajectory
                        break
                    
                    action_idx = Action.get_action_index(action_str)
                    traj.append((prev_action, action_idx))
                    prev_action = action_idx
                
                # Check trajectory length limit
                if isinstance(self.config, dict):
                    il_config = self.config.get('IL', {})
                    recollect_config = il_config.get('RECOLLECT_TRAINER', {})
                    max_len = recollect_config.get('max_traj_len', 500)
                else:
                    il_config = getattr(self.config, 'IL', {})
                    recollect_config = getattr(il_config, 'RECOLLECT_TRAINER', {})
                    max_len = getattr(recollect_config, 'max_traj_len', 500)
                if len(traj) > max_len:
                    print(f"Warning: Episode {episode.episode_id} trajectory too long "
                          f"({len(traj)} > {max_len}), truncating")
                    traj = traj[:max_len]
                
                if len(traj) > 0:
                    trajectories[episode.episode_id] = traj
                else:
                    print(f"Warning: Episode {episode.episode_id} has empty trajectory")
                    
            except Exception as e:
                error_msg = str(e)
                if "Camera view bounds exceed" in error_msg:
                    print(f"Warning: Episode {episode.episode_id} start position too close to map edge, skipping")
                else:
                    print(f"Error extracting trajectory for episode {episode.episode_id}: {e}")
                continue
        
        return trajectories
    
    def _tokenize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize instruction in observation.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            Modified observation with tokenized instruction (padded tensor)
        """
        from satnav.utils.build_vocab import tokenize_instruction_in_observation
        
        # Use unified tokenization function (tensor format with padding for training)
        return tokenize_instruction_in_observation(
            obs, self.vocab, max_length=self.max_instruction_len, output_format="tensor"
        )
    
    def _collect_episode(self, episode_idx: int) -> List[Tuple[Dict[str, Any], int, int]]:
        """Collect data for one episode by executing GT actions.
        
        Args:
            episode_idx: Index of episode in dataset
            
        Returns:
            List of (observation, prev_action, teacher_action) tuples
        """
        episode = self.env._dataset.episodes[episode_idx]
        
        # Check if we have trajectory for this episode
        if episode.episode_id not in self.trajectories:
            return []
        
        # Reset environment directly to the target episode for deterministic control
        try:
            obs = self.env.reset_to_episode(episode)
        except Exception as e:
            print(f"Warning: Could not reset to episode {episode.episode_id}: {e}")
            return []
        
        # Tokenize initial observation
        obs = self._tokenize_observation(obs)
        
        # Get trajectory
        trajectory = self.trajectories[episode.episode_id]
        
        # Collect data by executing GT actions
        collected_data = []
        for prev_action, teacher_action in trajectory:
            # Convert observation to proper format
            obs_dict = {}
            
            # RGB observation with resizing for memory efficiency
            if 'rgb' in obs:
                # Convert to tensor if needed
                rgb = obs['rgb']
                if not isinstance(rgb, torch.Tensor):
                    rgb = torch.from_numpy(rgb)
                # Ensure correct dtype
                if rgb.dtype != torch.uint8:
                    rgb = rgb.to(torch.uint8)
                
                # Resize to target size (default: 224x224) to reduce memory usage
                rgb = self._resize_rgb(rgb)
                obs_dict['rgb'] = rgb
            
            # Instruction observation (already tokenized)
            if 'instruction' in obs:
                obs_dict['instruction'] = obs['instruction']
            
            # Store (obs, prev_action, teacher_action)
            collected_data.append((
                obs_dict,
                prev_action,
                teacher_action
            ))
            
            # Execute teacher action
            obs, done, info = self.env.step(teacher_action)
            
            # Tokenize new observation
            obs = self._tokenize_observation(obs)
            
            if done:
                break
        
        return collected_data
    
    def _fill_preload_buffer(self):
        """Fill the preload buffer with episode data."""
        if isinstance(self.config, dict):
            il_config = self.config.get('IL', {})
            recollect_config = il_config.get('RECOLLECT_TRAINER', {})
            preload_size = recollect_config.get('preload_size', 10)
        else:
            il_config = getattr(self.config, 'IL', {})
            recollect_config = getattr(il_config, 'RECOLLECT_TRAINER', {})
            preload_size = getattr(recollect_config, 'preload_size', 10)
        
        while len(self._preload) < preload_size:
            # Get next episode
            if self._current_episode_idx >= len(self._episode_indices):
                # Shuffle and restart
                import random
                random.shuffle(self._episode_indices)
                self._current_episode_idx = 0
            
            episode_idx = self._episode_indices[self._current_episode_idx]
            self._current_episode_idx += 1
            
            # Collect episode data
            episode_data = self._collect_episode(episode_idx)
            
            if len(episode_data) > 0:
                self._preload.append(episode_data)
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Get next training sample.
        
        Returns:
            Tuple of:
            - observations: Dict of tensors (each with shape (T, ...))
            - prev_actions: Tensor of shape (T,)
            - teacher_actions: Tensor of shape (T,)
        """
        # Fill preload buffer if needed
        if len(self._preload) == 0:
            self._fill_preload_buffer()
        
        if len(self._preload) == 0:
            raise StopIteration
        
        # Get episode data
        episode_data = self._preload.popleft()
        
        # Unpack into separate lists
        # Handle both old format (3 items) and new format (3 items, weights computed later)
        obs_list, prev_actions, teacher_actions = zip(*episode_data)
        
        # Stack observations by sensor
        stacked_obs = defaultdict(list)
        for obs in obs_list:
            for sensor, value in obs.items():
                stacked_obs[sensor].append(value)
        
        # Stack each sensor's observations
        for sensor in stacked_obs:
            stacked_obs[sensor] = torch.stack(stacked_obs[sensor], dim=0)
        
        # Convert actions to tensors
        prev_actions = torch.tensor(prev_actions, dtype=torch.long)
        teacher_actions = torch.tensor(teacher_actions, dtype=torch.long)
        
        # Compute inflection weights (weight timesteps where action changes)
        if self.use_inflection_weighting:
            # First timestep is always an inflection point (action change from STOP)
            inflections = torch.cat([
                torch.tensor([1], dtype=torch.long),  # First timestep
                (teacher_actions[1:] != teacher_actions[:-1]).long(),  # Action changes
            ])
            weights = self.inflec_weights[inflections]
        else:
            # No weighting: all timesteps have equal weight
            weights = torch.ones(len(teacher_actions), dtype=torch.float32)
        
        return dict(stacked_obs), prev_actions, teacher_actions, weights

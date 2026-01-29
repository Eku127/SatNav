#!/usr/bin/env python3
"""Environment class for SatNav VLN tasks."""

from typing import Any, Dict, Iterator, Optional, Tuple, Union

from omegaconf import DictConfig

from satnav.core.episode import VLNEpisode
from satnav.core.simulator import Simulator
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.sims import create_simulator
from satnav.task.vln_task import VLNTask


class Env:
    """Environment class for SatNav VLN tasks in continuous space.
    
    This class connects the dataset, simulator, and task components to provide
    a unified interface for VLN navigation tasks. It follows a similar design to
    habitat-lab's Env class but simplified for continuous space navigation.
    
    The environment provides:
    - reset(): Reset environment and return initial observations
    - step(action): Execute action and return (obs, done, info)
    - get_metrics(): Get current evaluation metrics
    """
    
    def __init__(
        self,
        config: Union[DictConfig, dict],
        dataset: Optional[SatNavDataset] = None,
        cycle: bool = False
    ):
        """Initialize the environment.
        
        Args:
            config: Configuration object containing:
                - ENVIRONMENT.MAX_EPISODE_STEPS: Maximum steps per episode
                - SIMULATOR: Simulator configuration
                - TASK: Task configuration
                - DATASET: Dataset configuration (if dataset is None)
            dataset: Optional dataset object. If None, will be created from config.
            cycle: Whether to cycle through episodes when exhausted. 
                - False (default): Raise error when all episodes are exhausted (suitable for evaluation)
                - True: Automatically restart from the first episode (suitable for training)
        """
        self._config = config
        self._cycle = cycle
        
        # Extract configuration values
        if isinstance(config, DictConfig):
            env_config = getattr(config, "ENVIRONMENT", {})
            self.max_episode_steps = getattr(env_config, "MAX_EPISODE_STEPS", 500)
            sim_config = getattr(config, "SIMULATOR", {})
            task_config = getattr(config, "TASK", {})
            dataset_config = getattr(config, "DATASET", None)
        else:
            env_config = config.get("ENVIRONMENT", {})
            self.max_episode_steps = env_config.get("MAX_EPISODE_STEPS", 500)
            sim_config = config.get("SIMULATOR", {})
            task_config = config.get("TASK", {})
            dataset_config = config.get("DATASET", None)
        
        # Initialize dataset
        if dataset is None and dataset_config is not None:
            self._dataset = SatNavDataset(dataset_config)
        else:
            self._dataset = dataset
        
        # Initialize simulator using factory function
        # Currently supports SatSim (2D satellite imagery based simulator)
        scenes_dir = None
        if dataset_config is not None:
            if isinstance(dataset_config, DictConfig):
                scenes_dir = getattr(dataset_config, "SCENES_DIR", None)
            else:
                scenes_dir = dataset_config.get("SCENES_DIR")
        
        self._sim = create_simulator(config, scenes_dir=scenes_dir)
        
        # Initialize task
        # here we use VLNTask as default task
        self._task = VLNTask(task_config, self._sim)
        
        # Episode management
        self._current_episode: Optional[VLNEpisode] = None
        self._episode_iterator: Optional[Iterator[VLNEpisode]] = None
        self._elapsed_steps = 0
        self._episode_over = False
        
        # Setup episode iterator if dataset is available
        if self._dataset is not None:
            if len(self._dataset.episodes) == 0:
                raise ValueError("Dataset must have at least one episode")
            self._episode_iterator = self._dataset.get_episode_iterator()
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return initial observations.
        
        This method:
        1. Gets the next episode from the dataset
        2. Resets the task and simulator
        3. Returns initial observations
        
        Returns:
            Dictionary containing initial observations from all sensors:
                - "rgb": RGB image
                - "instruction": Instruction text
        
        Raises:
            RuntimeError: If dataset is not available or has no more episodes.
        """
        if self._dataset is None:
            raise RuntimeError(
                "Cannot reset: dataset is not available. "
                "Provide dataset in __init__ or in config."
            )
        
        if self._episode_iterator is None:
            self._episode_iterator = self._dataset.get_episode_iterator()
        
        # Get next episode
        try:
            self._current_episode = next(self._episode_iterator)
        except StopIteration:
            # Handle iterator exhaustion based on cycle mode
            if self._cycle:
                # Training mode: restart iterator and continue
                self._episode_iterator = self._dataset.get_episode_iterator()
                self._current_episode = next(self._episode_iterator)
            else:
                # Evaluation mode: raise error when all episodes are exhausted
                raise RuntimeError(
                    "All episodes exhausted. Cannot reset. "
                    "Set cycle=True in Env.__init__() to enable cycling for training."
                )
        
        # Reset task (which resets simulator and measures)
        observations = self._task.reset(self._current_episode)
        
        # Reset episode state
        self._elapsed_steps = 0
        self._episode_over = False
        
        return observations
    
    def reset_to_episode(self, episode: VLNEpisode) -> Dict[str, Any]:
        """Reset environment directly to a specified episode.
        
        This bypasses the dataset iterator and is useful for training-time
        data collection where we want deterministic control of episode order.
        """
        if episode is None:
            raise RuntimeError("Cannot reset_to_episode: episode is None")

        # Reset episode state FIRST (before task reset, to ensure clean state
        # even if task.reset() fails)
        self._elapsed_steps = 0
        self._episode_over = False
        self._current_episode = episode

        # Reset task (which resets simulator and measures)
        # If this fails, the environment is still in a clean state for the next episode
        observations = self._task.reset(self._current_episode)

        return observations
    
    def step(
        self,
        action: Union[str, Dict[str, Any], int]
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Execute an action and return new observations.
        
        Args:
            action: Action to execute. Can be:
                - Action string: "MOVE_FORWARD", "TURN_LEFT", etc.
                - Action dictionary: {"action": "MOVE_FORWARD"}
                - Action index: 0, 1, 2, 3
        
        Returns:
            Tuple containing:
                - obs: Dictionary of observations from all sensors
                - done: Boolean indicating if episode is done
                - info: Dictionary containing metrics and episode information
        
        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._current_episode is None:
            raise RuntimeError(
                "Cannot step: episode not set. Call reset() first."
            )
        
        if self._episode_over:
            raise RuntimeError(
                "Cannot step: episode is over. Call reset() to start a new episode."
            )
        
        # Execute action in task
        observations = self._task.step(action)
        
        # Update step count
        self._elapsed_steps += 1
        
        # Get metrics
        metrics = self._task.get_metrics()
        
        # Check if episode is done
        # Episode ends if:
        # 1. STOP action was called (consistent with VLN-CE behavior)
        # 2. Maximum steps reached
        # Note: Success measure is only used for metrics, not for termination
        stop_called = self._task.is_stop_called
        max_steps_reached = self._elapsed_steps >= self.max_episode_steps
        
        self._episode_over = stop_called or max_steps_reached
        
        # Prepare info dictionary
        info = {
            "metrics": metrics,
            "episode_id": self._current_episode.episode_id,
            "elapsed_steps": self._elapsed_steps,
            "episode_over": self._episode_over,
        }
        
        return observations, self._episode_over, info
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current evaluation metrics.
        
        Returns:
            Dictionary containing current metric values:
                - "distance_to_goal": Distance to goal in meters
                - "success": Success value (1.0 or 0.0)
                - "path_length": Path length in meters
                - "spl": SPL value (0.0 to 1.0)
        """
        if self._current_episode is None:
            return {}
        
        return self._task.get_metrics()
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get observation space description.
        
        This is a simplified description for documentation purposes.
        SatNav does not use gym.Space objects to keep the implementation minimal.
        
        Returns:
            Dictionary describing observation space structure.
        """
        return {
            "rgb": {
                "shape": (224, 224, 3),
                "dtype": "uint8",
                "description": "RGB image from satellite map"
            },
            "instruction": {
                "type": "dict",
                "keys": ["text"],
                "description": "Natural language navigation instruction"
            }
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Get action space description.
        
        This is a simplified description for documentation purposes.
        SatNav does not use gym.Space objects to keep the implementation minimal.
        
        Returns:
            Dictionary describing action space.
        """
        return {
            "actions": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
            "description": "Discrete action space for VLN navigation"
        }
    
    @property
    def current_episode(self) -> Optional[VLNEpisode]:
        """Get the current episode.
        
        Returns:
            Current episode, or None if not set.
        """
        return self._current_episode
    
    @property
    def episode_over(self) -> bool:
        """Check if current episode is over.
        
        Returns:
            True if episode is over, False otherwise.
        """
        return self._episode_over


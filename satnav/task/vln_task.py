#!/usr/bin/env python3
"""VLN task implementation for SatNav continuous space navigation."""

import os
import sys
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from satnav.core.episode import VLNEpisode
from satnav.core.simulator import Simulator
from satnav.task.actions import Action
from satnav.task.measures import (
    DistanceToGoal,
    Measure,
    OracleSuccess,
    PathLength,
    SPL,
    Success,
    TopDownMapSatNav,
)
from satnav.task.sensors import InstructionSensor, RGBSensor, Sensor

# Debug logging for specific rank
# Set SATNAV_DEBUG_RANK environment variable to enable debug logging for a specific rank
# Default is -999 (disabled), set to -1 to debug single-process runs
_DEBUG_RANK = int(os.environ.get('SATNAV_DEBUG_RANK', '-999'))
_DEBUG_LOG_FILE = os.environ.get('SATNAV_DEBUG_LOG', None)
_debug_file_handle = None

def _debug_log(msg: str, force: bool = False):
    """Log debug message if debugging is enabled for this rank."""
    global _debug_file_handle
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '-1')))
    if rank == _DEBUG_RANK or force:
        log_msg = f"[Rank {rank}][VLNTask] {msg}"
        if _DEBUG_LOG_FILE:
            if _debug_file_handle is None:
                _debug_file_handle = open(_DEBUG_LOG_FILE, 'a')
            _debug_file_handle.write(log_msg + '\n')
            _debug_file_handle.flush()
        else:
            print(log_msg, file=sys.stderr, flush=True)


class VLNTask:
    """VLN task class for continuous space navigation.
    
    This class manages sensors, measures, and task execution for VLN tasks
    in continuous space using geographic coordinates.
    """
    
    def __init__(
        self,
        config: Union[DictConfig, dict],
        simulator: Simulator
    ):
        """Initialize VLN task.
        
        Args:
            config: Task configuration containing:
                - SUCCESS_DISTANCE: Distance threshold for success (meters)
                - POSSIBLE_ACTIONS: List of allowed actions
                - MEASUREMENTS: List of measurement names to enable
            simulator: Simulator instance.
        """
        self._config = config
        self._sim = simulator
        self._current_episode: Optional[VLNEpisode] = None
        self.is_stop_called: bool = False  # Track if STOP action was called
        
        # Extract configuration values
        if isinstance(config, DictConfig):
            success_distance_config = getattr(config, "SUCCESS_DISTANCE", 3.0)
            possible_actions = getattr(config, "POSSIBLE_ACTIONS", Action.ALL_ACTIONS)
            measurements = getattr(config, "MEASUREMENTS", [])
        else:
            success_distance_config = config.get("SUCCESS_DISTANCE", 3.0)
            possible_actions = config.get("POSSIBLE_ACTIONS", Action.ALL_ACTIONS)
            measurements = config.get("MEASUREMENTS", [])
        
        # Handle SUCCESS_DISTANCE config (supports both old format and new dict format)
        # Old format: SUCCESS_DISTANCE: 10.0
        # New format: SUCCESS_DISTANCE: {DEFAULT: 10.0, Boundary: 10.0, LandmarkSet: 30.0}
        self._success_distance_config = success_distance_config
        if isinstance(success_distance_config, (int, float)):
            self.success_distance = float(success_distance_config)
        elif hasattr(success_distance_config, "DEFAULT"):
            self.success_distance = float(success_distance_config.DEFAULT)
        elif isinstance(success_distance_config, dict) and "DEFAULT" in success_distance_config:
            self.success_distance = float(success_distance_config["DEFAULT"])
        else:
            self.success_distance = 10.0  # Fallback default
        
        # Initialize sensors
        self.sensors: List[Sensor] = []
        self._init_sensors()
        
        # Initialize measures
        self.measures: List[Measure] = []
        self._init_measures(measurements)
        
        # Set up measure dependencies
        self._setup_measure_dependencies()
    
    def _init_sensors(self):
        """Initialize sensors for the task."""
        # RGB sensor - always included
        rgb_sensor = RGBSensor()
        self.sensors.append(rgb_sensor)
        
        # Instruction sensor - always included
        instruction_sensor = InstructionSensor()
        self.sensors.append(instruction_sensor)
    
    def _init_measures(self, measurements: List[str]):
        """Initialize measures based on configuration.
        
        Args:
            measurements: List of measurement names to enable.
        """
        # Extract TopDownMap configuration if available
        if isinstance(self._config, DictConfig):
            topdown_config = getattr(self._config, "TOP_DOWN_MAP", {})
            map_resolution = getattr(topdown_config, "MAP_RESOLUTION", 1024)
            padding_meters = getattr(topdown_config, "PADDING_METERS", 50.0)
            draw_reference_path = getattr(topdown_config, "DRAW_REFERENCE_PATH", True)
            draw_source_and_target = getattr(topdown_config, "DRAW_SOURCE_AND_TARGET", True)
            max_episode_steps = getattr(self._config, "MAX_EPISODE_STEPS", 500)
        else:
            topdown_config = self._config.get("TOP_DOWN_MAP", {})
            map_resolution = topdown_config.get("MAP_RESOLUTION", 1024)
            padding_meters = topdown_config.get("PADDING_METERS", 50.0)
            draw_reference_path = topdown_config.get("DRAW_REFERENCE_PATH", True)
            draw_source_and_target = topdown_config.get("DRAW_SOURCE_AND_TARGET", True)
            max_episode_steps = self._config.get("MAX_EPISODE_STEPS", 500)
        
        # Create measure instances
        distance_to_goal = DistanceToGoal(simulator=self._sim)
        success = Success(success_distance=self.success_distance, simulator=self._sim)
        oracle_success = OracleSuccess(success_distance=self.success_distance, simulator=self._sim)
        path_length = PathLength(simulator=self._sim)
        spl = SPL(simulator=self._sim)
        top_down_map = TopDownMapSatNav(
            simulator=self._sim,
            map_resolution=map_resolution,
            padding_meters=padding_meters,
            draw_reference_path=draw_reference_path,
            draw_source_and_target=draw_source_and_target,
            max_episode_steps=max_episode_steps,
            success_distance=self.success_distance,
        )
        
        # Store measures by name for easy access
        self._measures_dict = {
            "DISTANCE_TO_GOAL": distance_to_goal,
            "SUCCESS": success,
            "ORACLE_SUCCESS": oracle_success,
            "PATH_LENGTH": path_length,
            "SPL": spl,
            "TOP_DOWN_MAP": top_down_map,
        }
        
        # Add enabled measures to list
        if not measurements:
            # If no measurements specified, enable all except TOP_DOWN_MAP
            # (TOP_DOWN_MAP is optional for visualization)
            measurements = ["DISTANCE_TO_GOAL", "SUCCESS", "ORACLE_SUCCESS", "PATH_LENGTH", "SPL"]
        
        for measure_name in measurements:
            measure_name_upper = measure_name.upper()
            if measure_name_upper in self._measures_dict:
                self.measures.append(self._measures_dict[measure_name_upper])
    
    def _setup_measure_dependencies(self):
        """Set up dependencies between measures."""
        # Success depends on DistanceToGoal
        if "SUCCESS" in self._measures_dict and "DISTANCE_TO_GOAL" in self._measures_dict:
            self._measures_dict["SUCCESS"].set_distance_to_goal_measure(
                self._measures_dict["DISTANCE_TO_GOAL"]
            )
        
        # OracleSuccess depends on DistanceToGoal
        if "ORACLE_SUCCESS" in self._measures_dict and "DISTANCE_TO_GOAL" in self._measures_dict:
            self._measures_dict["ORACLE_SUCCESS"].set_distance_to_goal_measure(
                self._measures_dict["DISTANCE_TO_GOAL"]
            )
        
        # SPL depends on Success and PathLength
        if "SPL" in self._measures_dict:
            if "SUCCESS" in self._measures_dict and "PATH_LENGTH" in self._measures_dict:
                self._measures_dict["SPL"].set_measures(
                    self._measures_dict["SUCCESS"],
                    self._measures_dict["PATH_LENGTH"]
                )
    
    def reset(self, episode: VLNEpisode) -> Dict[str, Any]:
        """Reset the task for a new episode.
        
        Args:
            episode: The VLN episode to reset for.
            
        Returns:
            Initial observations dictionary.
        """
        _debug_log(f"=" * 60)
        _debug_log(f"VLNTask.reset() called for episode {episode.episode_id}")
        _debug_log(f"  trajectory_type: {getattr(episode, 'trajectory_type', 'N/A')}")
        _debug_log(f"  start_position: {episode.start_position}")
        _debug_log(f"  start_rotation: {episode.start_rotation}")
        _debug_log(f"  scene_id: {episode.scene_id}")
        
        self._current_episode = episode
        self.is_stop_called = False  # Reset stop flag for new episode
        
        # Update success_distance based on trajectory_type (for evaluation)
        trajectory_type = getattr(episode, 'trajectory_type', None)
        if trajectory_type and not isinstance(self._success_distance_config, (int, float)):
            # New dict format: get type-specific value
            if hasattr(self._success_distance_config, trajectory_type):
                eval_success_distance = float(getattr(self._success_distance_config, trajectory_type))
            elif isinstance(self._success_distance_config, dict) and trajectory_type in self._success_distance_config:
                eval_success_distance = float(self._success_distance_config[trajectory_type])
            else:
                eval_success_distance = self.success_distance  # Use DEFAULT
            
            # Update Success measure's threshold
            if "SUCCESS" in self._measures_dict:
                self._measures_dict["SUCCESS"].set_success_distance(eval_success_distance)
                _debug_log(f"  Updated SUCCESS measure: success_distance={eval_success_distance}m for {trajectory_type}")
        
        # Reset simulator (load scene and set initial state)
        _debug_log(f"  Step 1: Calling _sim.reset()")
        sim_obs = self._sim.reset(episode.scene_id)
        _debug_log(f"  Step 1 complete: sim_obs keys = {list(sim_obs.keys())}")
        
        _debug_log(f"  Step 2: Calling _sim.set_agent_state()")
        self._sim.set_agent_state(episode.start_position, episode.start_rotation)
        _debug_log(f"  Step 2 complete")
        
        # Reset all measures
        _debug_log(f"  Step 3: Resetting {len(self.measures)} measures")
        for i, measure in enumerate(self.measures):
            measure_name = measure.__class__.__name__
            _debug_log(f"    Resetting measure {i}: {measure_name}")
            try:
                measure.reset(episode, self._sim)
                _debug_log(f"    {measure_name} reset complete")
            except Exception as e:
                _debug_log(f"    {measure_name} reset FAILED: {e}")
                raise
        _debug_log(f"  Step 3 complete")
        
        # Get initial observations
        _debug_log(f"  Step 4: Calling get_observations()")
        observations = self.get_observations()
        _debug_log(f"  Step 4 complete: obs keys = {list(observations.keys())}")
        _debug_log(f"VLNTask.reset() completed successfully")
        
        return observations
    
    def step(
        self,
        action: Union[str, Dict[str, Any], int]
    ) -> Dict[str, Any]:
        """Execute an action and update the task state.
        
        Args:
            action: Action to execute. Can be:
                - Action string: "MOVE_FORWARD", "TURN_LEFT", etc.
                - Action dictionary: {"action": "MOVE_FORWARD"}
                - Action index: 0, 1, 2, 3
                
        Returns:
            Observations dictionary after executing the action.
            
        Raises:
            ValueError: If action is invalid.
        """
        # Parse action
        action_str = self._parse_action(action)
        
        # Validate action
        if not Action.is_valid_action(action_str):
            raise ValueError(
                f"Invalid action: {action_str}. "
                f"Valid actions are: {Action.ALL_ACTIONS}"
            )
        
        # Track STOP action (for episode termination, consistent with VLN-CE)
        if action_str == Action.STOP:
            self.is_stop_called = True
        
        # Execute action in simulator
        sim_obs = self._sim.step(action_str)
        
        # Update all measures
        for measure in self.measures:
            measure.update(self._sim, action_str, self._current_episode)
        
        # Get observations
        observations = self.get_observations()
        
        return observations
    
    def _parse_action(
        self,
        action: Union[str, Dict[str, Any], int]
    ) -> str:
        """Parse action from various input formats.
        
        Args:
            action: Action in various formats.
            
        Returns:
            Action string.
        """
        if isinstance(action, str):
            return action
        elif isinstance(action, dict):
            if "action" in action:
                return action["action"]
            else:
                raise ValueError(f"Action dictionary must contain 'action' key: {action}")
        elif isinstance(action, int):
            return Action.get_action_from_index(action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
    
    def get_observations(self) -> Dict[str, Any]:
        """Get observations from all sensors.
        
        Returns:
            Dictionary containing observations from all sensors:
                - "rgb": RGB image from RGBSensor
                - "instruction": Instruction text from InstructionSensor
                
        Raises:
            RuntimeError: If episode is not set (reset() not called).
        """
        # Check if episode is set first (before calling simulator)
        if self._current_episode is None:
            raise RuntimeError("Cannot get observations: episode not set. Call reset() first.")
        
        observations = {}
        
        # Get simulator observations (for RGB sensor)
        sim_obs = self._sim.get_observations()
        
        # Get observations from each sensor
        for sensor in self.sensors:
            if isinstance(sensor, RGBSensor):
                # RGB sensor needs simulator observations
                obs = sensor.get_observation(sim_obs=sim_obs)
                observations[sensor.uuid] = obs
            elif isinstance(sensor, InstructionSensor):
                # Instruction sensor needs episode
                obs = sensor.get_observation(episode=self._current_episode)
                observations[sensor.uuid] = obs
            else:
                # Generic sensor - try to get observation with available data
                try:
                    obs = sensor.get_observation(
                        sim_obs=sim_obs,
                        episode=self._current_episode,
                        simulator=self._sim
                    )
                    observations[sensor.uuid] = obs
                except Exception as e:
                    # Skip sensors that can't provide observation
                    pass
        
        return observations
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics from all measures.
        
        Returns:
            Dictionary containing metric values:
                - "distance_to_goal": Distance to goal in meters
                - "success": Success value (1.0 or 0.0)
                - "oracle_success": Oracle success value (1.0 or 0.0)
                - "path_length": Path length in meters
                - "spl": SPL value (0.0 to 1.0)
                - "top_down_map": Top-down map info dict (if enabled)
        """
        metrics = {}
        
        for measure in self.measures:
            if isinstance(measure, DistanceToGoal):
                metrics["distance_to_goal"] = measure.get_metric()
            elif isinstance(measure, Success):
                metrics["success"] = measure.get_metric()
            elif isinstance(measure, OracleSuccess):
                metrics["oracle_success"] = measure.get_metric()
            elif isinstance(measure, PathLength):
                metrics["path_length"] = measure.get_metric()
            elif isinstance(measure, SPL):
                metrics["spl"] = measure.get_metric()
            elif isinstance(measure, TopDownMapSatNav):
                metrics["top_down_map"] = measure.get_metric()
        
        return metrics
    
    def get_info(self) -> Dict[str, Any]:
        """Get info dictionary including visualization data.
        
        This method returns a dictionary suitable for visualization,
        including the top-down map if enabled.
        
        Returns:
            Dictionary containing:
                - All scalar metrics (distance_to_goal, success, etc.)
                - "top_down_map": Top-down map visualization dict (if enabled)
        """
        info = self.get_metrics()
        return info
    
    @property
    def current_episode(self) -> Optional[VLNEpisode]:
        """Get the current episode.
        
        Returns:
            Current episode, or None if not set.
        """
        return self._current_episode


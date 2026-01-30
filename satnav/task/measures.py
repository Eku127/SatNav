#!/usr/bin/env python3
"""Measure implementations for SatNav VLN tasks in continuous space.

All distance calculations use geodesic distance (Haversine formula) to account
for Earth's curvature, which is essential for continuous space navigation.
"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from satnav.core.episode import VLNEpisode
from satnav.core.simulator import Simulator


class Measure(abc.ABC):
    """Abstract base class for evaluation measures in SatNav.
    
    Measures track various metrics during episode execution, such as
    distance to goal, success, path length, and SPL.
    """
    
    def __init__(self):
        """Initialize the measure."""
        self._metric: Any = None
    
    @abc.abstractmethod
    def reset(self, episode: VLNEpisode, simulator: Optional[Simulator] = None) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode to reset for.
            simulator: Simulator instance (optional, may be provided in update).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update the measure after an action is taken.
        
        Args:
            simulator: Simulator instance to get agent state from.
            action: Action that was taken (optional).
            episode: Current episode (optional, may be needed for some measures).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_metric(self) -> Any:
        """Get the current metric value.
        
        Returns:
            Current metric value (type depends on measure implementation).
        """
        raise NotImplementedError


class DistanceToGoal(Measure):
    """Measure that calculates geodesic distance to the goal.
    
    This measure computes the great-circle distance (geodesic distance) from
    the agent's current position to the goal position, accounting for Earth's
    curvature. This is essential for continuous space navigation.
    """
    
    def __init__(self, simulator: Optional[Simulator] = None):
        """Initialize distance to goal measure.
        
        Args:
            simulator: Simulator instance (optional, can be provided in reset/update).
        """
        super().__init__()
        self._sim = simulator
        self._goal_position: Optional[List[float]] = None
        self._previous_position: Optional[List[float]] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode containing goal information.
            simulator: Simulator instance (if not provided in __init__).
        """
        if len(episode.goals) == 0:
            raise ValueError("episode must have at least one goal")
        
        # Store goal position (use first goal)
        self._goal_position = episode.goals[0].position
        
        # Reset previous position
        self._previous_position = None
        
        # Update metric with initial distance
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        self.update(sim, episode=episode)
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update distance to goal after an action.
        
        Args:
            simulator: Simulator instance to get agent state from.
            action: Action that was taken (unused).
            episode: Current episode (unused, goal stored in reset).
        """
        if self._goal_position is None:
            raise RuntimeError("reset() must be called before update()")
        
        # Get current agent position
        agent_state = simulator.get_agent_state()
        current_position = agent_state.position.tolist()
        
        # Always update distance - geodesic distance calculation is fast
        # Previous implementation used np.allclose with atol=1e-4 which was too
        # coarse for geographic coordinates (0.0001 degrees ≈ 11 meters at equator)
        self._metric = simulator.geodesic_distance(
            current_position,
            self._goal_position
        )
        self._previous_position = current_position.copy()
    
    def get_metric(self) -> float:
        """Get current distance to goal.
        
        Returns:
            Distance to goal in meters (geodesic distance).
        """
        if self._metric is None:
            return float('inf')
        return float(self._metric)


class Success(Measure):
    """Measure that determines if the agent has reached the goal.
    
    Success is defined as the agent being within success_distance of the goal
    and having called the STOP action.
    
    For boundary tasks (where start == goal), uses "leave and return" logic:
    - Agent must first leave the start area (distance > departure_threshold)
    - Then return to the goal area and call STOP (distance < success_distance)
    - This prevents trivial success by staying at the starting point
    """
    
    def __init__(
        self,
        success_distance: float = 3.0,
        simulator: Optional[Simulator] = None,
        departure_threshold_multiplier: float = 2.0
    ):
        """Initialize success measure.
        
        Args:
            success_distance: Distance threshold for success (in meters).
            simulator: Simulator instance (optional).
            departure_threshold_multiplier: For boundary tasks, agent must leave
                start area by this multiplier × success_distance before success
                can be triggered. Default 2.0 means 2× success_distance.
        """
        super().__init__()
        self._success_distance = success_distance
        self._departure_threshold = success_distance * departure_threshold_multiplier
        self._sim = simulator
        self._distance_to_goal: Optional[DistanceToGoal] = None
        self._is_stop_called = False
        
        # Boundary task detection and tracking
        self._is_boundary_task: bool = False
        self._has_left_start: bool = False
        self._start_position: Optional[List[float]] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode.
            simulator: Simulator instance (if not provided in __init__).
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        self._is_stop_called = False
        self._metric = 0.0
        
        # Store start position
        self._start_position = list(episode.start_position)
        
        # Check if this is a boundary task (start position ≈ goal position)
        if len(episode.goals) > 0:
            goal_position = episode.goals[0].position
            distance_start_to_goal = sim.geodesic_distance(
                self._start_position,
                goal_position
            )
            # If start and goal are within success_distance, it's a boundary task
            self._is_boundary_task = (distance_start_to_goal < self._success_distance)
        else:
            self._is_boundary_task = False
        
        # Reset boundary task tracking
        self._has_left_start = False
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update success measure after an action.
        
        Args:
            simulator: Simulator instance.
            action: Action that was taken. If "STOP", mark stop as called.
            episode: Current episode (unused).
        """
        # Check if STOP action was called
        if action is not None:
            if isinstance(action, str):
                action_str = action
            elif isinstance(action, dict) and "action" in action:
                action_str = action["action"]
            else:
                action_str = None
            
            if action_str == "STOP":
                self._is_stop_called = True
        
        # Get distance to goal (requires DistanceToGoal measure)
        if self._distance_to_goal is None:
            raise RuntimeError(
                "DistanceToGoal measure must be set before updating Success measure. "
                "Use set_distance_to_goal_measure() method."
            )
        
        distance_to_target = self._distance_to_goal.get_metric()
        
        # Handle boundary tasks differently
        if self._is_boundary_task:
            # For boundary tasks, use "leave and return" logic
            
            # Check if agent has left the start area
            if not self._has_left_start and self._start_position is not None:
                agent_state = simulator.get_agent_state()
                current_position = agent_state.position.tolist()
                distance_from_start = simulator.geodesic_distance(
                    current_position,
                    self._start_position
                )
                
                if distance_from_start > self._departure_threshold:
                    self._has_left_start = True
            
            # Success only if agent has left start area, called STOP, and within success distance
            if self._has_left_start and self._is_stop_called and distance_to_target < self._success_distance:
                self._metric = 1.0
            else:
                self._metric = 0.0
        else:
            # For normal tasks, standard logic: stop was called and within success distance
            if self._is_stop_called and distance_to_target < self._success_distance:
                self._metric = 1.0
            else:
                self._metric = 0.0
    
    def set_distance_to_goal_measure(self, distance_to_goal: DistanceToGoal) -> None:
        """Set the DistanceToGoal measure dependency.
        
        Args:
            distance_to_goal: DistanceToGoal measure instance.
        """
        self._distance_to_goal = distance_to_goal
    
    def get_metric(self) -> float:
        """Get current success value.
        
        Returns:
            1.0 if successful, 0.0 otherwise.
        """
        if self._metric is None:
            return 0.0
        return float(self._metric)


class PathLength(Measure):
    """Measure that accumulates the path length traveled by the agent.
    
    Path length is calculated by summing geodesic distances between consecutive
    positions. This accounts for Earth's curvature in continuous space.
    """
    
    def __init__(self, simulator: Optional[Simulator] = None):
        """Initialize path length measure.
        
        Args:
            simulator: Simulator instance (optional).
        """
        super().__init__()
        self._sim = simulator
        self._previous_position: Optional[List[float]] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode.
            simulator: Simulator instance (if not provided in __init__).
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        agent_state = sim.get_agent_state()
        self._previous_position = agent_state.position.tolist()
        self._metric = 0.0
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update path length after an action.
        
        Args:
            simulator: Simulator instance to get agent state from.
            action: Action that was taken (unused).
            episode: Current episode (unused).
        """
        if self._previous_position is None:
            raise RuntimeError("reset() must be called before update()")
        
        # Get current position
        agent_state = simulator.get_agent_state()
        current_position = agent_state.position.tolist()
        
        # Calculate geodesic distance from previous to current position
        # This is the key difference from habitat-lab: we use geodesic distance
        # instead of Euclidean distance for continuous space navigation
        step_distance = simulator.geodesic_distance(
            self._previous_position,
            current_position
        )
        
        # Accumulate path length
        self._metric += step_distance
        
        # Update previous position
        self._previous_position = current_position.copy()
    
    def get_metric(self) -> float:
        """Get accumulated path length.
        
        Returns:
            Total path length in meters (sum of geodesic distances).
        """
        if self._metric is None:
            return 0.0
        return float(self._metric)


class SPL(Measure):
    """Success weighted by Path Length measure.
    
    SPL (Success weighted by Path Length) is a combined metric that considers
    both success and path efficiency. It is defined as:
    
        SPL = Success * (reference_path_length / max(reference_path_length, actual_path_length))
    
    Reference: "On Evaluation of Embodied Agents" - Anderson et al.
    https://arxiv.org/pdf/1807.06757.pdf
    
    This measure depends on Success and PathLength measures, and requires
    the reference path length from the episode.
    """
    
    def __init__(
        self,
        simulator: Optional[Simulator] = None
    ):
        """Initialize SPL measure.
        
        Args:
            simulator: Simulator instance (optional).
        """
        super().__init__()
        self._sim = simulator
        self._success_measure: Optional[Success] = None
        self._path_length_measure: Optional[PathLength] = None
        self._reference_path_length: Optional[float] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode containing reference path.
            simulator: Simulator instance (if not provided in __init__).
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        
        # Calculate reference path length from episode
        # Reference path is a list of continuous coordinate points
        if episode.reference_path is None or len(episode.reference_path) < 2:
            # If no reference path, use straight-line geodesic distance
            if len(episode.goals) > 0:
                self._reference_path_length = sim.geodesic_distance(
                    episode.start_position,
                    episode.goals[0].position
                )
            else:
                self._reference_path_length = 0.0
        else:
            # Calculate path length by summing geodesic distances between consecutive points
            self._reference_path_length = 0.0
            for i in range(len(episode.reference_path) - 1):
                self._reference_path_length += sim.geodesic_distance(
                    episode.reference_path[i],
                    episode.reference_path[i + 1]
                )
        
        self._metric = 0.0
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update SPL measure after an action.
        
        Args:
            simulator: Simulator instance.
            action: Action that was taken (unused).
            episode: Current episode (unused).
        """
        if self._success_measure is None or self._path_length_measure is None:
            raise RuntimeError(
                "Success and PathLength measures must be set before updating SPL. "
                "Use set_measures() method."
            )
        
        # Get success value
        ep_success = self._success_measure.get_metric()
        
        # Get actual path length
        actual_path_length = self._path_length_measure.get_metric()
        
        # Calculate SPL
        if self._reference_path_length is None or self._reference_path_length <= 0:
            # No valid reference path
            self._metric = 0.0
        else:
            # SPL formula: success * (ref_len / max(ref_len, actual_len))
            self._metric = ep_success * (
                self._reference_path_length / max(
                    self._reference_path_length,
                    actual_path_length
                )
            )
    
    def set_measures(
        self,
        success_measure: Success,
        path_length_measure: PathLength
    ) -> None:
        """Set the Success and PathLength measure dependencies.
        
        Args:
            success_measure: Success measure instance.
            path_length_measure: PathLength measure instance.
        """
        self._success_measure = success_measure
        self._path_length_measure = path_length_measure
    
    def get_metric(self) -> float:
        """Get current SPL value.
        
        Returns:
            SPL value in [0, 1] range.
        """
        if self._metric is None:
            return 0.0
        return float(self._metric)


class OracleSuccess(Measure):
    """Oracle Success measure with boundary task support.
    
    Oracle Success measures whether the agent ever reached the goal during
    the entire trajectory (regardless of where it stopped). This metric is
    useful for evaluating navigation capability independent of the stopping
    decision.
    
    For boundary tasks (where start == goal), uses "leave and return" logic:
    - Agent must first leave the start area (distance > departure_threshold)
    - Then return to the goal area (distance < success_distance)
    - This prevents trivial success by staying at the starting point
    
    Unlike Success (which requires STOP to be called at the goal), OracleSuccess
    is 1.0 if the agent was ever within success_distance of the goal at any
    point during the episode (after leaving start area for boundary tasks).
    """
    
    def __init__(
        self,
        success_distance: float = 3.0,
        simulator: Optional[Simulator] = None,
        departure_threshold_multiplier: float = 2.0
    ):
        """Initialize oracle success measure.
        
        Args:
            success_distance: Distance threshold for success (in meters).
            simulator: Simulator instance (optional).
            departure_threshold_multiplier: For boundary tasks, agent must leave
                start area by this multiplier × success_distance before oracle
                success can be triggered. Default 2.0 means 2× success_distance.
        """
        super().__init__()
        self._success_distance = success_distance
        self._departure_threshold = success_distance * departure_threshold_multiplier
        self._sim = simulator
        self._distance_to_goal: Optional[DistanceToGoal] = None
        
        # Boundary task detection and tracking
        self._is_boundary_task: bool = False
        self._has_left_start: bool = False
        self._start_position: Optional[List[float]] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode.
            simulator: Simulator instance (if not provided in __init__).
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        self._metric = 0.0
        
        # Store start position
        self._start_position = list(episode.start_position)
        
        # Check if this is a boundary task (start position ≈ goal position)
        if len(episode.goals) > 0:
            goal_position = episode.goals[0].position
            distance_start_to_goal = sim.geodesic_distance(
                self._start_position,
                goal_position
            )
            # If start and goal are within success_distance, it's a boundary task
            self._is_boundary_task = (distance_start_to_goal < self._success_distance)
        else:
            self._is_boundary_task = False
        
        # Reset boundary task tracking
        self._has_left_start = False
        
        # Check initial distance (in case agent starts at goal for non-boundary tasks)
        if not self._is_boundary_task:
            self.update(sim, episode=episode)
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update oracle success measure after an action.
        
        Args:
            simulator: Simulator instance.
            action: Action that was taken (unused).
            episode: Current episode (unused).
        """
        # Once oracle success is achieved, it stays at 1.0
        if self._metric == 1.0:
            return
        
        # Get distance to goal (requires DistanceToGoal measure)
        if self._distance_to_goal is None:
            raise RuntimeError(
                "DistanceToGoal measure must be set before updating OracleSuccess measure. "
                "Use set_distance_to_goal_measure() method."
            )
        
        distance_to_target = self._distance_to_goal.get_metric()
        
        # Handle boundary tasks differently
        if self._is_boundary_task:
            # For boundary tasks, use "leave and return" logic
            
            # Check if agent has left the start area
            if not self._has_left_start and self._start_position is not None:
                agent_state = simulator.get_agent_state()
                current_position = agent_state.position.tolist()
                distance_from_start = simulator.geodesic_distance(
                    current_position,
                    self._start_position
                )
                
                if distance_from_start > self._departure_threshold:
                    self._has_left_start = True
            
            # Only check for oracle success if agent has left start area
            if self._has_left_start and distance_to_target < self._success_distance:
                self._metric = 1.0
        else:
            # For normal tasks, standard logic: within success distance = oracle success
            if distance_to_target < self._success_distance:
                self._metric = 1.0
    
    def set_distance_to_goal_measure(self, distance_to_goal: DistanceToGoal) -> None:
        """Set the DistanceToGoal measure dependency.
        
        Args:
            distance_to_goal: DistanceToGoal measure instance.
        """
        self._distance_to_goal = distance_to_goal
    
    def get_metric(self) -> float:
        """Get current oracle success value.
        
        Returns:
            1.0 if agent ever reached goal, 0.0 otherwise.
        """
        if self._metric is None:
            return 0.0
        return float(self._metric)


class TopDownMapSatNav(Measure):
    """Top-down map visualization measure for SatNav.
    
    This measure generates a top-down visualization of the navigation task,
    including:
    - Cropped satellite map as background (containing reference path and goal)
    - Reference path (green line)
    - Start position (blue dot)
    - Goal position (red dot)
    - Agent trajectory (gradient colored path)
    - Current agent position with heading arrow
    
    The map is cropped to include all reference path points and goal with padding,
    avoiding rendering the entire (potentially large) satellite map.
    
    Adapted from VLN-CE's TopDownMapVLNCE for geographic coordinates.
    """
    
    def __init__(
        self,
        simulator: Optional[Simulator] = None,
        map_resolution: int = 1024,
        padding_meters: float = 50.0,
        draw_reference_path: bool = True,
        draw_source_and_target: bool = True,
        agent_sprite_size: int = 30,
        path_thickness: int = 4,  # Increased from 2 to 4 for better visibility
        max_episode_steps: int = 500,
        success_distance: float = 3.0
    ):
        """Initialize the TopDownMapSatNav measure.
        
        Args:
            simulator: Simulator instance (optional, can be provided in reset).
            map_resolution: Maximum resolution of the map image (longest side).
            padding_meters: Padding to add around the bounding box in meters.
            draw_reference_path: Whether to draw the reference path.
            draw_source_and_target: Whether to draw start/goal markers.
            agent_sprite_size: Size of the agent arrow sprite in pixels.
            path_thickness: Thickness of path lines.
            max_episode_steps: Maximum steps for gradient calculation.
            success_distance: Success distance threshold in meters.
        """
        super().__init__()
        self._sim = simulator
        self._map_resolution = map_resolution
        self._padding_meters = padding_meters
        self._draw_reference_path = draw_reference_path
        self._draw_source_and_target = draw_source_and_target
        self._agent_sprite_size = agent_sprite_size
        self._path_thickness = path_thickness
        self._max_episode_steps = max_episode_steps
        self._success_distance = success_distance
        
        # State
        self._top_down_map: Optional[np.ndarray] = None
        self._bounds: Optional[Dict[str, float]] = None
        self._step_count: int = 0
        self._agent_path: List[List[float]] = []  # List of [lon, lat, alt]
        self._previous_position: Optional[List[float]] = None
        
        # Episode data
        self._start_position: Optional[List[float]] = None
        self._goal_position: Optional[List[float]] = None
        self._reference_path: Optional[List[List[float]]] = None
    
    def reset(
        self,
        episode: VLNEpisode,
        simulator: Optional[Simulator] = None
    ) -> None:
        """Reset the measure for a new episode.
        
        Args:
            episode: The VLN episode containing reference path and goals.
            simulator: Simulator instance (if not provided in __init__).
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or reset")
        
        self._sim = sim
        self._step_count = 0
        self._agent_path = []
        self._previous_position = None
        
        # Store episode data
        self._start_position = list(episode.start_position)
        self._goal_position = list(episode.goals[0].position) if episode.goals else None
        self._reference_path = episode.reference_path
        
        # Initialize map
        self._initialize_map(episode)
        
        # Add starting position to path
        agent_state = sim.get_agent_state()
        self._agent_path.append(agent_state.position.tolist())
        self._previous_position = agent_state.position.tolist()
        
        # Initial metric update
        self._update_metric_internal()
    
    def _initialize_map(self, episode: VLNEpisode) -> None:
        """Initialize the top-down map by cropping satellite imagery.
        
        Args:
            episode: The VLN episode.
        """
        # Import here to avoid circular imports
        from satnav.utils.maps import (
            crop_satellite_map,
            draw_reference_path,
            draw_source_and_target,
        )
        
        # Get the satellite TIF from simulator
        # Access the internal SatSim/AerialSim to get the scene
        # SatSimWrapper stores scene in _satsim._current_scene
        # AerialSimWrapper stores scene in _current_scene directly
        if hasattr(self._sim, '_satsim') and self._sim._satsim._current_scene is not None:
            sat_tif = self._sim._satsim._current_scene
        elif hasattr(self._sim, '_current_scene') and self._sim._current_scene is not None:
            sat_tif = self._sim._current_scene
        else:
            raise RuntimeError(
                "Cannot access satellite map from simulator. "
                "For AerialSim, ensure SCENES_DIR is set in config and TIF files exist."
            )
        
        if sat_tif is None:
            raise RuntimeError("No satellite scene loaded in simulator")
        
        # Prepare reference path (use empty list if None)
        ref_path = self._reference_path if self._reference_path else []
        
        # Crop satellite map
        self._top_down_map, self._bounds = crop_satellite_map(
            sat_tif=sat_tif,
            reference_path=ref_path,
            goal_position=self._goal_position or self._start_position,
            start_position=self._start_position,
            padding_meters=self._padding_meters,
            max_resolution=self._map_resolution
        )
        
        # Draw reference path
        if self._draw_reference_path and ref_path:
            draw_reference_path(
                self._top_down_map,
                ref_path,
                self._bounds,
                thickness=self._path_thickness
            )
        
        # Draw source and target
        if self._draw_source_and_target and self._goal_position:
            draw_source_and_target(
                self._top_down_map,
                self._start_position,
                self._goal_position,
                self._bounds
                )
    
    def update(
        self,
        simulator: Simulator,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        episode: Optional[VLNEpisode] = None
    ) -> None:
        """Update the top-down map after an action.
        
        Args:
            simulator: Simulator instance.
            action: Action that was taken (unused).
            episode: Current episode (unused).
        """
        self._step_count += 1
        
        # Get current agent position
        agent_state = simulator.get_agent_state()
        current_position = agent_state.position.tolist()
        
        # Add to path if position changed
        # Use rtol=0 to disable relative tolerance, only use absolute tolerance
        # This is important for geographic coordinates where values are large (e.g., 114 degrees)
        # and we want to detect small absolute changes (e.g., 0.00003 degrees ≈ 3 meters)
        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-6, rtol=0
        ):
            self._agent_path.append(current_position)
            self._previous_position = current_position
        
        # Update metric
        self._update_metric_internal()
    
    def _update_metric_internal(self) -> None:
        """Update the metric (rendered map image)."""
        from satnav.utils.maps import draw_agent, draw_path, geo_to_pixel
        
        if self._top_down_map is None:
            return
        
        # Get current agent state (always use latest state, not from path)
        agent_state = self._sim.get_agent_state()
        current_position = agent_state.position.tolist()
        current_rotation = agent_state.rotation
        
        # Create a copy of the base map
        rendered_map = self._top_down_map.copy()
        map_shape = rendered_map.shape[:2]
        
        # Draw agent path (yellow)
        # Include current position in path even if it hasn't changed (for real-time visualization)
        path_to_draw = self._agent_path.copy()
        
        # Always add current position to show complete path up to current step
        # This ensures the path is drawn even when agent is turning (position unchanged)
        if not path_to_draw:
            path_to_draw.append(current_position)
        
        if len(path_to_draw) >= 2:
            path_pixels = []
            for point in path_to_draw:
                row, col = geo_to_pixel(point[0], point[1], self._bounds, map_shape)
                path_pixels.append((row, col))
            
            draw_path(
                rendered_map,
                path_pixels,
                color=(0, 255, 255),  # Yellow (BGR format)
                thickness=self._path_thickness,
                max_steps=self._max_episode_steps
            )
        
        # Draw current agent position with heading (always use latest position and rotation)
        rendered_map = draw_agent(
            rendered_map,
            current_position,
            current_rotation,
            self._bounds,
            agent_radius_px=self._agent_sprite_size // 2  # Convert sprite_size to radius
        )
        
        # Calculate agent position in pixel coordinates for output
        row, col = geo_to_pixel(current_position[0], current_position[1], self._bounds, map_shape)
        agent_map_coord = (row, col)
        agent_angle = current_rotation
        
        # Store metric
        self._metric = {
            "map": rendered_map,
            "agent_map_coord": agent_map_coord,
            "agent_angle": agent_angle,
            "bounds": self._bounds,
            "step_count": self._step_count,
        }
    
    def get_metric(self) -> Dict[str, Any]:
        """Get the current top-down map metric.
        
        Returns:
            Dictionary containing:
                - "map": RGB image of the top-down map (H, W, 3) uint8.
                - "agent_map_coord": (row, col) agent position in pixel coordinates.
                - "agent_angle": Agent heading in degrees.
                - "bounds": Geographic bounds dictionary.
                - "step_count": Current step count.
        """
        if self._metric is None:
            return {
                "map": np.zeros((100, 100, 3), dtype=np.uint8),
                "agent_map_coord": (0, 0),
                "agent_angle": 0.0,
                "bounds": {},
                "step_count": 0,
            }
        return self._metric
    
    def get_agent_path(self) -> List[List[float]]:
        """Get the recorded agent path.
        
        Returns:
            List of agent positions as [[lon, lat, alt], ...].
        """
        return self._agent_path.copy()


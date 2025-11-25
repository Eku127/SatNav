#!/usr/bin/env python3
"""Measure implementations for SatNav VLN tasks in continuous space.

All distance calculations use geodesic distance (Haversine formula) to account
for Earth's curvature, which is essential for continuous space navigation.
"""

import abc
from typing import Any, Dict, List, Optional, Union

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
        
        # Only update if position has changed significantly
        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            # Calculate geodesic distance (not Euclidean!)
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
    """
    
    def __init__(
        self,
        success_distance: float = 3.0,
        simulator: Optional[Simulator] = None
    ):
        """Initialize success measure.
        
        Args:
            success_distance: Distance threshold for success (in meters).
            simulator: Simulator instance (optional).
        """
        super().__init__()
        self._success_distance = success_distance
        self._sim = simulator
        self._distance_to_goal: Optional[DistanceToGoal] = None
        self._is_stop_called = False
    
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
        
        # Success if stop was called and within success distance
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


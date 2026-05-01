#!/usr/bin/env python3
from __future__ import annotations

"""Path follower for SatNav continuous navigation.

This module provides path following utilities for navigating in continuous
geographic space (WGS84 coordinates). It is adapted from habitat-lab's
ShortestPathFollower and VLN-CE's path following utilities.

Key differences from habitat-lab:
- Uses geographic coordinates (longitude, latitude, altitude) instead of 3D Cartesian
- Uses heading in degrees (0 = North) instead of quaternion rotation
- Uses geodesic distance (Haversine) instead of Euclidean distance
- Uses geographic bearing calculation for direction

Reference:
- habitat-lab: habitat/tasks/nav/shortest_path_follower.py
- VLN-CE: habitat_extensions/shortest_path_follower.py
"""

import math
from typing import List, Optional, Union


from satnav.core.simulator import Simulator
from satnav.core.utils import geodesic_distance
from satnav.task.actions import Action


def calculate_bearing(
    lon1: float, lat1: float,
    lon2: float, lat2: float
) -> float:
    """Calculate initial bearing from point 1 to point 2.
    
    Uses the forward azimuth formula to calculate the initial bearing
    (direction) from one geographic point to another.
    
    Args:
        lon1: Longitude of starting point in degrees.
        lat1: Latitude of starting point in degrees.
        lon2: Longitude of destination point in degrees.
        lat2: Latitude of destination point in degrees.
        
    Returns:
        Bearing in degrees (0-360, 0 = North, 90 = East, 180 = South, 270 = West).
        
    Example:
        >>> # Bearing from Beijing to Shanghai
        >>> bearing = calculate_bearing(116.4, 39.9, 121.5, 31.2)
        >>> print(f"Bearing: {bearing:.1f}°")  # ~136° (Southeast)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_diff = math.radians(lon2 - lon1)
    
    x = math.sin(lon_diff) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff))
    
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def normalize_angle_diff(angle: float) -> float:
    """Normalize angle difference to [-180, 180] range.
    
    Args:
        angle: Angle in degrees.
        
    Returns:
        Normalized angle in [-180, 180] range.
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


class SatNavPathFollower:
    """Path follower that provides next action to move towards a goal.
    
    Unlike DiscretePathPlanner which plans an entire action sequence,
    SatNavPathFollower is used iteratively - it returns one action at a time
    based on the current agent state and goal position.
    
    This is useful for:
    - Teacher forcing during training (oracle action generation)
    - Shortest path sensor (provides ground truth next action)
    - Online navigation with replanning
    
    The follower uses a simple greedy strategy:
    1. Calculate bearing to goal from current position
    2. Calculate angle difference from current heading to required bearing
    3. If facing roughly towards goal, move forward
    4. Otherwise, turn towards the goal
    
    Attributes:
        goal_radius: Distance threshold for considering goal reached (meters)
        turn_angle: Angle rotated per TURN_LEFT/TURN_RIGHT action (degrees)
        
    Example:
        >>> follower = SatNavPathFollower(goal_radius=3.0, turn_angle=15.0)
        >>> goal = [114.07, 22.54, 100.0]  # [lon, lat, alt]
        >>> while True:
        ...     action = follower.get_next_action(goal, simulator)
        ...     if action == Action.STOP:
        ...         break
        ...     simulator.step(action)
    """
    
    def __init__(
        self,
        goal_radius: float = 3.0,
        turn_angle: float = 15.0,
        return_action_string: bool = True
    ):
        """Initialize the path follower.
        
        Args:
            goal_radius: Distance threshold for considering goal reached (meters).
                When agent is within this distance of the goal, STOP is returned.
            turn_angle: Angle rotated per TURN_LEFT/TURN_RIGHT action (degrees).
                Used to determine when agent is "close enough" to facing the goal.
            return_action_string: If True, return action as string (e.g., "MOVE_FORWARD").
                If False, return action index (e.g., 1).
        """
        self.goal_radius = goal_radius
        self.turn_angle = turn_angle
        self.return_action_string = return_action_string
    
        # State for oscillation prevention
        self._last_action: Optional[str] = None
    
    def get_next_action(
        self,
        goal_position: List[float],
        simulator: Simulator
    ) -> Union[str, int]:
        """Get the next action to move towards the goal.
        
        Given the current agent state (from simulator) and a goal position,
        returns the best next action to take.
        
        Args:
            goal_position: Target position as [longitude, latitude, altitude].
            simulator: SatNav simulator instance to get current agent state.
            
        Returns:
            If return_action_string is True: Action string (e.g., "MOVE_FORWARD").
            If return_action_string is False: Action index (e.g., 1).
            
        Note:
            Returns STOP if:
            - Agent is within goal_radius of the goal
            - Goal position is invalid (None or same as current position)
        """
        # Get current agent state
        agent_state = simulator.get_agent_state()
        current_position = agent_state.position  # [lon, lat, alt]
        current_heading = agent_state.rotation    # degrees, 0 = North
        
        # Calculate distance to goal
        distance = geodesic_distance(current_position, goal_position)
        
        # Check if already at goal
        if distance <= self.goal_radius:
            return self._return_action(Action.STOP)
        
        # Calculate bearing to goal (absolute bearing from North)
        target_bearing = calculate_bearing(
            current_position[0], current_position[1],  # lon, lat
            goal_position[0], goal_position[1]
        )
        
        # Calculate angle difference (how much we need to turn)
        angle_diff = normalize_angle_diff(target_bearing - current_heading)
        
        # Adaptive threshold: larger when close to goal to prevent oscillation
        # Base threshold: half turn angle for normal navigation
        base_threshold = self.turn_angle / 2
        
        # When close to goal (within 2x goal_radius), use larger threshold
        # This prevents oscillation when small movements cause large bearing changes
        if distance < self.goal_radius * 2:
            # Within 2x goal radius, use full turn_angle as threshold
            threshold = self.turn_angle
        else:
            threshold = base_threshold
        
        # Hysteresis: use larger threshold if we were moving forward
        # This prevents rapid switching between forward and turn actions
        if self._last_action == Action.MOVE_FORWARD:
            # If we were moving forward, use larger threshold to continue
            # This creates a "dead zone" that prevents oscillation
            threshold = min(threshold * 1.5, self.turn_angle * 1.5)
        
        # Decide action based on angle difference with adaptive threshold
        if abs(angle_diff) <= threshold:
            # Close enough to facing the goal, move forward
            action = Action.MOVE_FORWARD
        elif angle_diff > 0:
            # Need to turn right (clockwise, positive direction)
            action = Action.TURN_RIGHT
        else:
            # Need to turn left (counter-clockwise, negative direction)
            action = Action.TURN_LEFT
        
        # Store last action for hysteresis
        self._last_action = action
        
        return self._return_action(action)
    
    def _return_action(self, action: str) -> Union[str, int]:
        """Return action in the requested format.
        
        Args:
            action: Action string.
            
        Returns:
            Action string or index depending on return_action_string setting.
        """
        if self.return_action_string:
            return action
        else:
            return Action.get_action_index(action)
    
    def get_action_sequence_to_goal(
        self,
        goal_position: List[float],
        simulator: Simulator,
        max_steps: int = 1000,
        execute_actions: bool = False
    ) -> List[str]:
        """Get complete action sequence to reach a goal.
        
        Iteratively calls get_next_action and optionally executes each action
        until the goal is reached or max_steps is exceeded.
        
        Args:
            goal_position: Target position as [longitude, latitude, altitude].
            simulator: SatNav simulator instance.
            max_steps: Maximum number of actions to generate.
            execute_actions: If True, execute each action in the simulator.
                If False, only calculate actions without modifying simulator state.
                
        Returns:
            List of action strings to reach the goal.
            
        Warning:
            If execute_actions is False, the returned sequence may not be accurate
            because it doesn't account for actual movement. Use execute_actions=True
            for accurate sequences, or use DiscretePathPlanner for theoretical paths.
        """
        # Reset last action state for fresh start
        self._last_action = None
        
        actions: List[str] = []
        
        for _ in range(max_steps):
            action = self.get_next_action(goal_position, simulator)
            
            # Convert to string if needed
            if isinstance(action, int):
                action = Action.get_action_from_index(action)
            
            if action == Action.STOP:
                break
            
            actions.append(action)
            
            if execute_actions:
                simulator.step(action)
        
        return actions


class ReferencePathFollower:
    """Follower that navigates along a reference path of waypoints.
    
    Given a reference path (list of geographic waypoints), this follower
    generates actions to visit each waypoint in sequence. This is useful for:
    - Converting reference_path from dataset to action sequences (for training)
    - Generating teacher forcing data for offline/recollection training
    - Evaluating path following accuracy
    
    The follower maintains internal state tracking which waypoint it's currently
    navigating to, and advances to the next waypoint when within reach.
    
    Attributes:
        goal_radius: Distance threshold for considering a waypoint reached.
        turn_angle: Angle per turn action.
        
    Example:
        >>> reference_path = [
        ...     [114.06, 22.54, 100.0],  # start
        ...     [114.07, 22.54, 100.0],  # waypoint 1
        ...     [114.08, 22.55, 100.0],  # waypoint 2 (goal)
        ... ]
        >>> follower = ReferencePathFollower()
        >>> actions = follower.follow_path(reference_path, simulator, execute=True)
    """
    
    def __init__(
        self,
        goal_radius: float = 3.0,
        turn_angle: float = 15.0
    ):
        """Initialize the reference path follower.
        
        Args:
            goal_radius: Distance threshold for all waypoints including final goal (meters).
            turn_angle: Angle per turn action (degrees).
        """
        self.goal_radius = goal_radius
        self.turn_angle = turn_angle
        
        # Internal path follower using unified goal_radius for all waypoints
        self._follower = SatNavPathFollower(
            goal_radius=goal_radius,
            turn_angle=turn_angle
        )
        self._reference_path: Optional[List[List[float]]] = None
        self._current_waypoint_idx: int = 0
    
    def reset(self, reference_path: List[List[float]]) -> None:
        """Reset the follower with a new reference path.
        
        Args:
            reference_path: List of waypoints as [[lon, lat, alt], ...].
                First point is start position, last point is goal.
        """
        self._reference_path = reference_path
        self._current_waypoint_idx = 1  # Start navigating to first waypoint (skip start)
    
    def get_next_action(
        self,
        simulator: Simulator
    ) -> str:
        """Get the next action to follow the reference path.
        
        Args:
            simulator: SatNav simulator instance.
            
        Returns:
            Next action string. Returns STOP when path is completed.
            
        Raises:
            RuntimeError: If reset() was not called with a path.
        """
        if self._reference_path is None:
            raise RuntimeError("reset() must be called with a reference path first")
        
        # Check if we've completed the path
        if self._current_waypoint_idx >= len(self._reference_path):
            return Action.STOP
        
        # Get current waypoint target
        current_waypoint = self._reference_path[self._current_waypoint_idx]
        
        # Check if we've reached the current waypoint
        agent_state = simulator.get_agent_state()
        current_position = agent_state.position
        distance = geodesic_distance(current_position, current_waypoint)
        
        # Use goal_radius for all waypoints
        if distance <= self.goal_radius:
            # Reached current waypoint, advance to next
            self._current_waypoint_idx += 1
            if self._current_waypoint_idx >= len(self._reference_path):
                return Action.STOP
            current_waypoint = self._reference_path[self._current_waypoint_idx]
        
        # Use unified follower with goal_radius for all waypoints
        return self._follower.get_next_action(current_waypoint, simulator)
    
    def follow_path(
        self,
        reference_path: List[List[float]],
        simulator: Simulator,
        execute: bool = True,
        max_steps: int = 2000
    ) -> List[str]:
        """Follow an entire reference path and return action sequence.
        
        Args:
            reference_path: List of waypoints as [[lon, lat, alt], ...].
            simulator: SatNav simulator instance.
            execute: If True, execute actions in simulator.
            max_steps: Maximum number of steps.
            
        Returns:
            List of action strings to follow the path.
        """
        self.reset(reference_path)
        
        actions: List[str] = []
        
        for _ in range(max_steps):
            action = self.get_next_action(simulator)
            
            if action == Action.STOP:
                actions.append(action)
                break
            
            actions.append(action)
            
            if execute:
                simulator.step(action)
        
        return actions
    
    def get_current_waypoint_index(self) -> int:
        """Get the index of the current target waypoint.
        
        Returns:
            Index into the reference path of the current target waypoint.
        """
        return self._current_waypoint_idx
    
    def get_progress(self) -> float:
        """Get progress along the reference path as a fraction.
        
        Returns:
            Progress in [0, 1] range. 0 = at start, 1 = completed path.
        """
        if self._reference_path is None or len(self._reference_path) <= 1:
            return 0.0
        
        return self._current_waypoint_idx / (len(self._reference_path) - 1)


class ShortestPathSensor:
    """Sensor-style wrapper for path follower (compatible with SatNav sensor interface).
    
    This class provides a sensor-like interface for getting oracle actions,
    similar to VLN-CE's ShortestPathSensor. It can be used to provide
    ground truth actions for supervised learning.
    
    Example:
        >>> sensor = ShortestPathSensor(goal_radius=3.0)
        >>> # Get oracle action for current state
        >>> action = sensor.get_observation(
        ...     episode=current_episode,
        ...     simulator=simulator
        ... )
    """
    
    def __init__(
        self,
        goal_radius: float = 3.0,
        turn_angle: float = 15.0,
        return_index: bool = True
    ):
        """Initialize the shortest path sensor.
        
        Args:
            goal_radius: Distance threshold for goal reached.
            turn_angle: Angle per turn action.
            return_index: If True, return action as index. If False, return string.
        """
        self._follower = SatNavPathFollower(
            goal_radius=goal_radius,
            turn_angle=turn_angle,
            return_action_string=not return_index
        )
        self._return_index = return_index
    
    def get_observation(
        self,
        simulator: Simulator,
        episode=None,
        goal_position: Optional[List[float]] = None
    ) -> Union[int, str, List[float]]:
        """Get the oracle action observation.
        
        Args:
            simulator: SatNav simulator instance.
            episode: Episode object (uses episode.goals[0].position if goal_position not provided).
            goal_position: Goal position override. If None, uses episode goal.
            
        Returns:
            Oracle action (as index, string, or one-hot encoding).
        """
        # Get goal position
        if goal_position is None:
            if episode is None:
                raise ValueError("Either episode or goal_position must be provided")
            goal_position = episode.goals[0].position
        
        # Get next action
        action = self._follower.get_next_action(goal_position, simulator)
        
        return action

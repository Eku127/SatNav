#!/usr/bin/env python3
"""Discrete path planner for SatNav continuous navigation.

This module provides a greedy path planner that converts relative waypoints
(distance, bearing) to discrete action sequences. It is adapted from VLN-CE's
DiscretePathPlanner but uses geographic coordinate conventions:
- Degrees instead of radians for bearing
- North-referenced bearing (0 = North, 90 = East)

Reference: VLN-CE habitat_extensions/discrete_planner.py
"""

import math
from typing import List, Tuple

import numpy as np

from satnav.task.actions import Action


class DiscretePathPlanner:
    """Greedy path planner that converts relative waypoints to discrete actions.
    
    Given a relative target position (distance in meters, bearing in degrees),
    generates a sequence of discrete actions (TURN_LEFT, TURN_RIGHT, MOVE_FORWARD)
    to reach the target.
    
    The planner uses a greedy algorithm that:
    1. Calculates the bearing to the goal
    2. Turns to face the goal (minimizing turn count)
    3. Moves forward
    4. Repeats until within goal_radius of the target
    
    This is designed for use with waypoint-based navigation models where the model
    predicts high-level waypoints and this planner converts them to low-level actions.
    
    Attributes:
        forward_distance: Distance moved per MOVE_FORWARD action (meters)
        turn_angle: Angle rotated per TURN_LEFT/TURN_RIGHT action (degrees)
        goal_radius: Distance threshold for considering goal reached (meters)
        step_limit: Maximum number of actions to generate
        
    Example:
        >>> planner = DiscretePathPlanner(
        ...     forward_distance=0.25,  # 25cm per step
        ...     turn_angle=15.0,        # 15 degrees per turn
        ... )
        >>> # Plan to reach a point 10m away at 45 degrees to the right
        >>> actions = planner.plan(distance_meters=10.0, bearing_degrees=45.0)
        >>> print(actions)  # ['TURN_RIGHT', 'TURN_RIGHT', 'TURN_RIGHT', 'MOVE_FORWARD', ...]
    """
    
    def __init__(
        self,
        forward_distance: float = 0.25,
        turn_angle: float = 15.0,
        goal_radius: float = 0.5,
        step_limit: int = 500
    ):
        """Initialize the discrete path planner.
        
        Args:
            forward_distance: Distance moved per MOVE_FORWARD action (meters).
                Default is 0.25m (25cm) which is standard for VLN tasks.
            turn_angle: Angle rotated per TURN_LEFT/TURN_RIGHT action (degrees).
                Default is 15 degrees. Must evenly divide 360.
            goal_radius: Distance threshold for considering goal reached (meters).
                Default is 0.5m. Planner stops when within this distance of goal.
            step_limit: Maximum number of actions to generate. Default is 500.
                Prevents infinite loops in edge cases.
                
        Raises:
            AssertionError: If turn_angle does not evenly divide 360.
        """
        assert 360 % turn_angle == 0, \
            f"turn_angle ({turn_angle}) must evenly divide 360"
        
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.goal_radius = goal_radius
        self.step_limit = step_limit
        self.num_turns_in_circle = int(360 / turn_angle)
    
    def plan(
        self,
        distance_meters: float,
        bearing_degrees: float
    ) -> List[str]:
        """Plan discrete actions to reach a relative target.
        
        Given a target specified as relative distance and bearing from the
        current position and heading, plans a sequence of discrete actions
        to reach the target.
        
        The bearing is relative to the agent's current facing direction:
        - 0 degrees: directly ahead (forward)
        - 90 degrees: to the right
        - -90 degrees (or 270): to the left
        - 180 degrees: behind
        
        Args:
            distance_meters: Distance to target in meters.
            bearing_degrees: Bearing to target in degrees relative to current heading.
                0 = forward, positive = clockwise (right), negative = counter-clockwise (left).
                
        Returns:
            List of action strings (e.g., ['TURN_RIGHT', 'MOVE_FORWARD', ...])
            to reach the target. Does NOT include final STOP action.
            
        Example:
            >>> planner = DiscretePathPlanner()
            >>> # Target is 5m ahead and 30 degrees to the right
            >>> actions = planner.plan(5.0, 30.0)
            >>> print(actions[:5])
            ['TURN_RIGHT', 'TURN_RIGHT', 'MOVE_FORWARD', 'MOVE_FORWARD', ...]
        """
        # Handle edge case: already at goal
        if distance_meters <= self.goal_radius:
            return []
        
        # Convert to local Cartesian coordinates
        # x = East (positive right), y = North (positive forward)
        bearing_rad = math.radians(bearing_degrees)
        goal_x = distance_meters * math.sin(bearing_rad)
        goal_y = distance_meters * math.cos(bearing_rad)
        goal = np.array([goal_x, goal_y])
        
        # Current state: at origin, facing North (positive y direction)
        position = np.array([0.0, 0.0])
        heading = 0.0  # degrees, 0 = facing forward (North/positive y)
        
        actions: List[str] = []
        
        while np.linalg.norm(position - goal) > self.goal_radius:
            if len(actions) >= self.step_limit:
                break
            
            # Calculate angle to goal from current position
            delta = goal - position
            if np.linalg.norm(delta) < 1e-6:
                break
            
            # Angle to goal in degrees (0 = North/forward direction)
            goal_angle_deg = math.degrees(math.atan2(delta[0], delta[1]))
            
            # Angle difference (how much we need to turn)
            angle_diff = goal_angle_deg - heading
            
            # Normalize to [-180, 180]
            angle_diff = self._normalize_angle(angle_diff)
            
            # Decide action based on angle difference
            if abs(angle_diff) <= self.turn_angle / 2:
                # Close enough to facing goal, move forward
                heading_rad = math.radians(heading)
                position[0] += self.forward_distance * math.sin(heading_rad)
                position[1] += self.forward_distance * math.cos(heading_rad)
                actions.append(Action.MOVE_FORWARD)
            elif angle_diff > 0:
                # Need to turn right (clockwise)
                heading = (heading + self.turn_angle) % 360
                actions.append(Action.TURN_RIGHT)
            else:
                # Need to turn left (counter-clockwise)
                heading = (heading - self.turn_angle) % 360
                actions.append(Action.TURN_LEFT)
        
        # Optionally adjust final heading to face away from start
        # (useful for continuing navigation along a path)
        actions.extend(self._adjust_final_heading(position, goal, heading))
        
        return actions
    
    def plan_polar(self, r: float, theta: float) -> List[str]:
        """Plan using polar coordinates (VLN-CE compatible interface).
        
        This method provides compatibility with VLN-CE's waypoint action format
        where waypoints are specified as (r, theta) in polar coordinates.
        
        Args:
            r: Distance to target in meters (radius in polar coords).
            theta: Angle to target in radians relative to current heading.
                0 = forward, positive = counter-clockwise (left).
                Note: This follows VLN-CE convention which is opposite to
                standard geographic bearing!
                
        Returns:
            List of action strings to reach the target.
        """
        # Convert VLN-CE convention (radians, CCW positive) to 
        # SatNav convention (degrees, CW positive for right)
        bearing_degrees = -math.degrees(theta)
        return self.plan(r, bearing_degrees)
    
    def plan_to_waypoints(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> List[str]:
        """Plan actions to visit a sequence of relative waypoints.
        
        Each waypoint is relative to the agent's position and heading
        at the START of the entire plan (not relative to previous waypoint).
        
        Args:
            waypoints: List of (distance_meters, bearing_degrees) tuples.
                Each tuple specifies a waypoint relative to the initial
                agent position and heading.
                
        Returns:
            Concatenated list of actions to visit all waypoints in order.
            
        Note:
            This method assumes waypoints are given in global relative coordinates
            (all relative to start). For waypoints relative to previous waypoint,
            you need to transform them first or use plan() sequentially.
        """
        if not waypoints:
            return []
        
        all_actions: List[str] = []
        
        # For simplicity, plan to each waypoint sequentially
        # This is a greedy approach; optimal path planning would be more complex
        for distance, bearing in waypoints:
            # Plan from current (simulated) position to this waypoint
            wp_actions = self.plan(distance, bearing)
            all_actions.extend(wp_actions)
        
        return all_actions
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] range.
        
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
    
    def _adjust_final_heading(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        current_heading: float
    ) -> List[str]:
        """Optionally adjust heading at goal to face away from start.
        
        This ensures the agent is oriented to continue navigation along a path,
        rather than facing an arbitrary direction after reaching the waypoint.
        
        Args:
            position: Current (final) position as [x, y].
            goal: Goal position as [x, y].
            current_heading: Current heading in degrees.
            
        Returns:
            List of turn actions to adjust heading, or empty list.
        """
        # Calculate ideal heading (facing away from origin towards goal direction)
        if np.linalg.norm(goal) < 1e-6:
            return []
        
        ideal_heading = math.degrees(math.atan2(goal[0], goal[1]))
        angle_diff = self._normalize_angle(ideal_heading - current_heading)
        
        actions: List[str] = []
        while abs(angle_diff) > self.turn_angle / 2:
            if len(actions) >= self.num_turns_in_circle:
                break
            if angle_diff > 0:
                actions.append(Action.TURN_RIGHT)
                angle_diff -= self.turn_angle
            else:
                actions.append(Action.TURN_LEFT)
                angle_diff += self.turn_angle
        
        return actions


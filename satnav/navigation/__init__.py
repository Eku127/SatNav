#!/usr/bin/env python3
"""Navigation utilities for SatNav continuous space navigation.

This module provides path following and discrete action planning utilities
for converting continuous navigation paths to discrete action sequences.

Main components:
- DiscretePathPlanner: Converts relative waypoints (distance, bearing) to
  discrete action sequences.
- SatNavPathFollower: Provides next action given a goal position in geographic
  coordinates.
- ReferencePathFollower: Follows a reference path by sequentially navigating
  to each waypoint.
"""

from satnav.navigation.discrete_planner import DiscretePathPlanner
from satnav.navigation.path_follower import (
    ReferencePathFollower,
    SatNavPathFollower,
)

__all__ = [
    "DiscretePathPlanner",
    "SatNavPathFollower",
    "ReferencePathFollower",
]


#!/usr/bin/env python3
"""Episode data structures for VLN tasks in continuous space."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InstructionData:
    """Instruction data for VLN episodes.
    
    Attributes:
        instruction_text: Natural language instruction text.
    """
    instruction_text: str


@dataclass
class NavigationGoal:
    """Navigation goal specification for continuous space.
    
    Attributes:
        position: Goal position as [longitude, latitude, altitude].
    """
    position: List[float]  # [longitude, latitude, altitude]


@dataclass
class VLNEpisode:
    """VLN Episode data structure for continuous space navigation.
    
    This class represents a single navigation episode in continuous space,
    using geographic coordinates (longitude, latitude, altitude) and roll angle
    for rotation, rather than discrete navigation graph nodes.
    
    Attributes:
        episode_id: Unique identifier for the episode.
        scene_id: Identifier for the scene/map.
        start_position: Starting position as [longitude, latitude, altitude].
        start_rotation: Starting rotation as roll angle (0-360 degrees, 0 = North).
        goals: List of navigation goals (continuous coordinates, not graph nodes).
        reference_path: Reference path as list of continuous coordinate points.
            Format: [[lon1, lat1, alt1], [lon2, lat2, alt2], ...]
        instruction: Instruction data containing the natural language instruction.
        trajectory_id: Identifier for the ground truth trajectory.
        trajectory_type: Type of trajectory ('Boundary' or 'LandmarkSet'), optional.
    """
    episode_id: str
    scene_id: str
    start_position: List[float]  # [longitude, latitude, altitude]
    start_rotation: float  # roll angle (0-360 degrees, 0 = North)
    goals: List[NavigationGoal]
    reference_path: List[List[float]]  # [[lon, lat, alt], ...]
    instruction: InstructionData
    trajectory_id: str
    trajectory_type: Optional[str] = None  # 'Boundary' or 'LandmarkSet'


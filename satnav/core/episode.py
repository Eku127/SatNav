#!/usr/bin/env python3
"""Episode data structures for VLN tasks in continuous space.

The public episode objects intentionally keep benchmark metadata separate from
machine-local runtime state.  In particular, ``scene_id`` is the stable logical
identifier stored in results while ``scene_path`` is an optional resolved path
used only to locate local simulator assets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InstructionData:
    """Instruction data for VLN episodes.
    
    Attributes:
        instruction_text: Natural language instruction text.
    """
    instruction_text: str
    instruction_type: Optional[str] = None
    difficulty_level: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return all source instruction fields without dropping extensions."""
        data = dict(self.extras)
        data["instruction_text"] = self.instruction_text
        if self.instruction_type is not None:
            data["instruction_type"] = self.instruction_type
        if self.difficulty_level is not None:
            data["difficulty_level"] = self.difficulty_level
        return data


@dataclass
class NavigationGoal:
    """Navigation goal specification for continuous space.
    
    Attributes:
        position: Goal position as [longitude, latitude, altitude].
    """
    position: List[float]  # [longitude, latitude, altitude]
    extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return all source goal fields without dropping extensions."""
        data = dict(self.extras)
        data["position"] = self.position
        return data


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
        trajectory_type: Type of trajectory ('Boundary', 'LandmarkSet', or 'Road'), optional.
        trajectory_subtype: More specific trajectory category, optional.
        waypoints: Original high-level waypoints from the release data.
        aux_info: Additional benchmark metadata.
        split: Dataset split used to build the stable episode key.
        scene_path: Optional local path resolved from ``scene_id``.  This is
            runtime-only and is excluded from :meth:`to_dict` by default.
    """
    episode_id: str
    scene_id: str
    start_position: List[float]  # [longitude, latitude, altitude]
    start_rotation: float  # roll angle (0-360 degrees, 0 = North)
    goals: List[NavigationGoal]
    reference_path: List[List[float]]  # [[lon, lat, alt], ...]
    instruction: InstructionData
    trajectory_id: str
    trajectory_type: Optional[str] = None  # 'Boundary', 'LandmarkSet', or 'Road'
    trajectory_subtype: Optional[str] = None
    waypoints: List[List[float]] = field(default_factory=list)
    aux_info: Dict[str, Any] = field(default_factory=dict)
    split: Optional[str] = None
    scene_path: Optional[str] = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"serialize": False, "runtime_only": True},
    )
    extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def episode_key(self) -> str:
        """Return the stable ``split::scene_id::episode_id`` identity."""
        return "::".join(
            (str(self.split or ""), str(self.scene_id), str(self.episode_id))
        )

    @property
    def key(self) -> str:
        """Compatibility-friendly short alias for :attr:`episode_key`."""
        return self.episode_key

    def to_dict(self, include_runtime: bool = False) -> Dict[str, Any]:
        """Serialize the lossless benchmark fields.

        Machine-local ``scene_path`` and derived identity values are omitted by
        default so writing an episode to a result cannot leak local paths.
        """
        data = dict(self.extras)
        data.update(
            {
                "episode_id": self.episode_id,
                "trajectory_id": self.trajectory_id,
                "trajectory_type": self.trajectory_type,
                "scene_id": self.scene_id,
                "start_position": self.start_position,
                "start_rotation": self.start_rotation,
                "goals": [
                    goal.to_dict() if isinstance(goal, NavigationGoal) else goal
                    for goal in self.goals
                ],
                "instruction": (
                    self.instruction.to_dict()
                    if isinstance(self.instruction, InstructionData)
                    else self.instruction
                ),
                "waypoints": self.waypoints,
                "reference_path": self.reference_path,
                "aux_info": self.aux_info,
                "trajectory_subtype": self.trajectory_subtype,
            }
        )
        if include_runtime:
            data.update(
                {
                    "split": self.split,
                    "scene_path": self.scene_path,
                    "episode_key": self.episode_key,
                }
            )
        return data

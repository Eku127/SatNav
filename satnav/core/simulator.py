#!/usr/bin/env python3
from __future__ import annotations

"""Simulator interfaces and data structures for SatNav."""

import abc
from typing import Any, Dict, List, Optional, Sequence, Union



class Position3D(list):
    """List-like position container with numpy-like tolist() compatibility."""

    def tolist(self) -> List[float]:
        return list(self)


class AgentState:
    """Represents the state of an agent in the environment.
    
    Attributes:
        position: Agent position as [longitude, latitude, altitude].
        rotation: Agent rotation as roll angle (0-360 degrees, 0 = North).
    """
    def __init__(self, position: Sequence[float], rotation: float):
        """Initialize agent state.
        
        Args:
            position: Agent position as [longitude, latitude, altitude].
            rotation: Agent rotation as roll angle in degrees.
        """
        self.position = Position3D(float(value) for value in position)
        self.rotation = float(rotation)
    
    def __repr__(self) -> str:
        return f"AgentState(position={self.position}, rotation={self.rotation})"


class Simulator(abc.ABC):
    """Abstract base class for simulators in SatNav.
    
    This class defines the interface that all simulators must implement.
    Simulators handle the interaction between agents and the environment.
    """
    
    @abc.abstractmethod
    def reset(self, scene_id: str) -> Dict[str, Any]:
        """Reset the simulator and load a new scene.
        
        Args:
            scene_id: Identifier for the scene to load.
            
        Returns:
            Initial observations from the simulator.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, action: Union[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute an action in the simulator.
        
        Args:
            action: Action to execute. Can be an action name, action index,
                   or action dictionary.
                   
        Returns:
            Observations after executing the action.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_agent_state(self) -> AgentState:
        """Get the current state of the agent.
        
        Returns:
            Current agent state containing position and rotation.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_agent_state(self, position: List[float], rotation: float) -> None:
        """Set the state of the agent.
        
        Args:
            position: New position as [longitude, latitude, altitude].
            rotation: New rotation as roll angle in degrees.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from all sensors.
        
        Returns:
            Dictionary containing observations from all sensors.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def geodesic_distance(
        self,
        position_a: Sequence[float],
        position_b: Sequence[float]
    ) -> float:
        """Calculate geodesic distance between two positions.
        
        Uses the Haversine formula to calculate great-circle distance
        between two points on Earth's surface.
        
        Args:
            position_a: First position as [longitude, latitude, altitude].
            position_b: Second position as [longitude, latitude, altitude].
            
        Returns:
            Distance in meters between the two positions.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_navigable(self, position: Sequence[float]) -> bool:
        """Check if a position is navigable.
        
        Args:
            position: Position to check as [longitude, latitude, altitude].
            
        Returns:
            True if the position is navigable, False otherwise.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def sample_navigable_point(self) -> List[float]:
        """Sample a random navigable point in the current scene.
        
        Returns:
            A navigable position as [longitude, latitude, altitude].
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release simulator resources.

        The default is a no-op so existing external simulator implementations
        remain source-compatible while resource-owning backends can override it.
        """
        return None
    
    @property
    @abc.abstractmethod
    def sensor_suite(self):
        """Get the sensor suite.
        
        Returns:
            The sensor suite containing all sensors.
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def action_space(self):
        """Get the action space.
        
        Returns:
            The action space defining valid actions.
        """
        raise NotImplementedError


# Type alias for observations
Observations = Dict[str, Any]

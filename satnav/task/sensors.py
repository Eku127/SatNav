#!/usr/bin/env python3
"""Sensor implementations for SatNav VLN tasks."""

import abc
from typing import Any, Dict, Optional

import numpy as np

from satnav.core.episode import VLNEpisode
from satnav.core.simulator import Simulator


class Sensor(abc.ABC):
    """Abstract base class for sensors in SatNav.
    
    Sensors provide observations from the environment to the agent.
    Each sensor must implement the get_observation method.
    """
    
    @abc.abstractmethod
    def get_observation(self, **kwargs) -> Any:
        """Get the current observation from this sensor.
        
        Args:
            **kwargs: Additional arguments that may be needed:
                - sim_obs: Dictionary of observations from simulator
                - episode: Current VLN episode
                - simulator: Simulator instance
                
        Returns:
            Observation data (type depends on sensor implementation).
        """
        raise NotImplementedError


class RGBSensor(Sensor):
    """RGB sensor that extracts RGB images from simulator observations.
    
    This sensor retrieves RGB images that are cropped from satellite maps
    based on the agent's current position, altitude, and camera HFOV.
    """
    
    def __init__(self, uuid: str = "rgb"):
        """Initialize RGB sensor.
        
        Args:
            uuid: Unique identifier for this sensor (default: "rgb").
        """
        self.uuid = uuid
    
    def get_observation(self, sim_obs: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Get RGB image observation.
        
        Args:
            sim_obs: Dictionary containing simulator observations.
                Should contain "rgb" key with RGB image as numpy array.
            **kwargs: Additional arguments (unused).
                
        Returns:
            RGB image as numpy array with shape (H, W, 3) and dtype uint8.
            
        Raises:
            KeyError: If "rgb" key is not found in sim_obs.
            ValueError: If sim_obs is None.
        """
        if sim_obs is None:
            raise ValueError("sim_obs cannot be None for RGBSensor")
        
        if "rgb" not in sim_obs:
            raise KeyError("sim_obs must contain 'rgb' key")
        
        rgb_image = sim_obs["rgb"]
        
        # Ensure it's a numpy array
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)
        
        return rgb_image


class InstructionSensor(Sensor):
    """Instruction sensor that provides natural language navigation instructions.
    
    This sensor extracts the instruction text from the current episode.
    """
    
    def __init__(self, uuid: str = "instruction"):
        """Initialize instruction sensor.
        
        Args:
            uuid: Unique identifier for this sensor (default: "instruction").
        """
        self.uuid = uuid
    
    def get_observation(
        self,
        episode: Optional[VLNEpisode] = None,
        **kwargs
    ) -> Dict[str, str]:
        """Get instruction text observation.
        
        Args:
            episode: Current VLN episode containing instruction data.
            **kwargs: Additional arguments (unused).
                
        Returns:
            Dictionary containing instruction text:
                {"text": instruction_text}
            
        Raises:
            ValueError: If episode is None or instruction is missing.
        """
        if episode is None:
            raise ValueError("episode cannot be None for InstructionSensor")
        
        if episode.instruction is None:
            raise ValueError("episode.instruction cannot be None")
        
        return {
            "text": episode.instruction.instruction_text
        }


class GlobalGPSSensor(Sensor):
    """Global GPS sensor that provides the agent's current position in geographic coordinates.
    
    This sensor returns the agent's current longitude and latitude (and optionally altitude)
    in the global coordinate frame. Useful for continuous space navigation.
    """
    
    def __init__(
        self,
        simulator: Optional[Simulator] = None,
        dimensionality: int = 2,
        uuid: str = "globalgps"
    ):
        """Initialize global GPS sensor.
        
        Args:
            simulator: Simulator instance to get agent state from.
            dimensionality: Number of dimensions to return:
                - 2: [longitude, latitude]
                - 3: [longitude, latitude, altitude]
            uuid: Unique identifier for this sensor (default: "globalgps").
        """
        self._sim = simulator
        self._dimensionality = dimensionality
        self.uuid = uuid
    
    def get_observation(
        self,
        simulator: Optional[Simulator] = None,
        **kwargs
    ) -> np.ndarray:
        """Get current GPS position observation.
        
        Args:
            simulator: Simulator instance (if not provided in __init__).
            **kwargs: Additional arguments (unused).
                
        Returns:
            GPS position as numpy array:
                - If dimensionality=2: [longitude, latitude]
                - If dimensionality=3: [longitude, latitude, altitude]
            
        Raises:
            ValueError: If simulator is not available.
        """
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or get_observation")
        
        agent_state = sim.get_agent_state()
        position = agent_state.position  # [longitude, latitude, altitude]
        
        if self._dimensionality == 2:
            # Return only longitude and latitude
            return np.array([position[0], position[1]], dtype=np.float32)
        else:
            # Return longitude, latitude, and altitude
            return np.array(position, dtype=np.float32)


class VLNOracleProgressSensor(Sensor):
    """Oracle progress sensor that provides relative progress towards the goal.
    
    This sensor calculates the relative progress from start to goal based on
    geodesic distances. Progress is normalized to [0, 1] range.
    """
    
    def __init__(
        self,
        simulator: Optional[Simulator] = None,
        uuid: str = "progress"
    ):
        """Initialize oracle progress sensor.
        
        Args:
            simulator: Simulator instance to get agent state and calculate distances.
            uuid: Unique identifier for this sensor (default: "progress").
        """
        self._sim = simulator
        self.uuid = uuid
        self._initial_distance: Optional[float] = None
    
    def get_observation(
        self,
        episode: Optional[VLNEpisode] = None,
        simulator: Optional[Simulator] = None,
        **kwargs
    ) -> np.ndarray:
        """Get relative progress towards goal.
        
        Args:
            episode: Current VLN episode containing goals and start position.
            simulator: Simulator instance (if not provided in __init__).
            **kwargs: Additional arguments (unused).
                
        Returns:
            Progress value as numpy array with shape (1,) and dtype float32.
            Value is in [0, 1] range:
                - 0: At start position
                - 1: At goal position
                - Values > 1: Beyond goal (should not happen normally)
            
        Raises:
            ValueError: If episode or simulator is not available.
        """
        if episode is None:
            raise ValueError("episode cannot be None for VLNOracleProgressSensor")
        
        if len(episode.goals) == 0:
            raise ValueError("episode must have at least one goal")
        
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError("simulator must be provided either in __init__ or get_observation")
        
        # Get current agent position
        agent_state = sim.get_agent_state()
        current_position = agent_state.position.tolist()
        
        # Get goal position
        goal_position = episode.goals[0].position
        
        # Calculate distance to target
        distance_to_target = sim.geodesic_distance(current_position, goal_position)
        
        # Handle invalid distances
        if not np.isfinite(distance_to_target):
            return np.array([0.0], dtype=np.float32)
        
        # Calculate initial distance from start to goal (cache it)
        if self._initial_distance is None:
            start_position = episode.start_position
            self._initial_distance = sim.geodesic_distance(start_position, goal_position)
        
        # Calculate progress: (initial_distance - current_distance) / initial_distance
        if self._initial_distance <= 0:
            # Already at goal or invalid distance
            return np.array([1.0], dtype=np.float32)
        
        progress = (self._initial_distance - distance_to_target) / self._initial_distance
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        
        return np.array([progress], dtype=np.float32)
    
    def reset(self):
        """Reset the cached initial distance.
        
        Call this when starting a new episode.
        """
        self._initial_distance = None


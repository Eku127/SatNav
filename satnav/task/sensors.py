#!/usr/bin/env python3
from __future__ import annotations

"""Sensor implementations for SatNav VLN tasks."""

import abc
import math
from typing import Any, Dict, List, Optional

from satnav.core.episode import VLNEpisode
from satnav.core.simulator import Simulator
from satnav.core.utils import lonlat_to_ego_displacement, wrap_heading_deg


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
    
    def get_observation(self, sim_obs: Optional[Dict[str, Any]] = None, **kwargs):
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
        
        import numpy as np

        rgb_image = sim_obs["rgb"]
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


class AgentPoseSensor(Sensor):
    """Agent pose sensor that provides ego-frame relative pose from episode start.
    
    The coordinate system is defined by the agent's initial heading at episode start
    (no dependency on cardinal directions N/S/E/W):
    
    - Forward axis (+Y): the direction the agent was facing at t=0
      (= "up" in the first BEV image)
    - Right axis (+X): 90 degrees clockwise from forward
    - Heading zero: the initial heading at episode start
    
    Output: np.array([delta_forward_m, delta_right_m,
                      sin(delta_heading), cos(delta_heading)],
                     dtype=float32)
    
    Where:
    - delta_forward_m: displacement along the initial forward direction (meters)
    - delta_right_m: displacement perpendicular to forward (meters, right=positive)
    - sin/cos(delta_heading): heading change from initial heading, encoded as
      sin/cos to avoid discontinuity at +/-180 degrees
    
    At episode start (t=0), the output is always [0, 0, 0, 1] (sin(0)=0, cos(0)=1).
    """
    
    def __init__(
        self,
        simulator: Optional[Simulator] = None,
        uuid: str = "agent_pose"
    ):
        """Initialize agent pose sensor.
        
        Args:
            simulator: Simulator instance to get agent state from.
            uuid: Unique identifier for this sensor (default: "agent_pose").
        """
        self._sim = simulator
        self.uuid = uuid
        self._start_position: Optional[List[float]] = None
        self._start_heading: Optional[float] = None
    
    def reset(self, episode: VLNEpisode) -> None:
        """Reset sensor state for a new episode.
        
        Records the simulator's current position and heading as the reference
        for all subsequent relative pose calculations. This is called after the
        simulator is reset to the episode start, so using the simulator state
        avoids small projection round-trip offsets in SatSim.
        
        Args:
            episode: The VLN episode to reset for.
        """
        if self._sim is not None:
            try:
                agent_state = self._sim.get_agent_state()
                self._start_position = agent_state.position.tolist()
                self._start_heading = float(agent_state.rotation)
                return
            except (AttributeError, RuntimeError, ValueError):
                pass

        self._start_position = list(episode.start_position)
        self._start_heading = float(episode.start_rotation)
    
    def get_observation(
        self,
        simulator: Optional[Simulator] = None,
        **kwargs
    ):
        """Get relative pose observation in the ego-frame.
        
        Args:
            simulator: Simulator instance (if not provided in __init__).
            **kwargs: Additional arguments (unused).
                
        Returns:
            Pose as numpy array with shape (4,) and dtype float32:
                [delta_forward_m, delta_right_m, sin(delta_heading), cos(delta_heading)]
            
        Raises:
            RuntimeError: If reset() has not been called.
            ValueError: If simulator is not available.
        """
        if self._start_position is None or self._start_heading is None:
            raise RuntimeError(
                "AgentPoseSensor.reset() must be called before get_observation(). "
                "This is typically called in VLNTask.reset()."
            )
        
        sim = simulator if simulator is not None else self._sim
        if sim is None:
            raise ValueError(
                "simulator must be provided either in __init__ or get_observation"
            )
        
        # Get current agent state
        agent_state = sim.get_agent_state()
        current_position = agent_state.position.tolist()
        current_heading = float(agent_state.rotation)
        
        # Calculate ego-frame displacement (forward, right) in meters
        delta_forward, delta_right = lonlat_to_ego_displacement(
            self._start_position,
            self._start_heading,
            current_position
        )
        
        # Calculate relative heading change
        delta_heading_deg = wrap_heading_deg(current_heading - self._start_heading)
        delta_heading_rad = math.radians(delta_heading_deg)
        
        sin_dh = math.sin(delta_heading_rad)
        cos_dh = math.cos(delta_heading_rad)
        
        import numpy as np
        return np.array(
            [delta_forward, delta_right, sin_dh, cos_dh],
            dtype=np.float32
        )

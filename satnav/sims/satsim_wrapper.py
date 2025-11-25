#!/usr/bin/env python3
"""SatSim wrapper implementation for SatNav."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig

from satnav.core.simulator import AgentState, Observations, Simulator
from satnav.core.utils import geodesic_distance, geodesic_distance_with_altitude


class SatSimWrapper(Simulator):
    """Wrapper for the satsim satellite map simulator.
    
    This class wraps the satsim simulator and implements the Simulator interface.
    Currently, all methods raise NotImplementedError as the actual satsim engine
    is not yet implemented. This provides a complete interface definition for
    future implementation.
    
    The simulator uses geographic coordinates (longitude, latitude, altitude)
    and roll angle (heading) for agent state. RGB observations are generated
    by cropping satellite map imagery based on the agent's position, altitude,
    and camera HFOV.
    """
    
    def __init__(self, config: Union[DictConfig, dict]) -> None:
        """Initialize the simulator wrapper.
        
        Args:
            config: Simulator configuration containing:
                - FORWARD_STEP_SIZE: Step size for forward movement (meters)
                - TURN_ANGLE: Angle for turning (degrees)
                - RGB_SENSOR: RGB sensor configuration
                    - WIDTH: Image width in pixels
                    - HEIGHT: Image height in pixels
                    - HFOV: Horizontal field of view in degrees
        """
        # Store config
        if isinstance(config, DictConfig):
            self.config = config
            self.forward_step_size = getattr(config, "FORWARD_STEP_SIZE", 0.25)
            self.turn_angle = getattr(config, "TURN_ANGLE", 15.0)
            rgb_config = getattr(config, "RGB_SENSOR", {})
            self.rgb_width = getattr(rgb_config, "WIDTH", 224)
            self.rgb_height = getattr(rgb_config, "HEIGHT", 224)
            self.rgb_hfov = getattr(rgb_config, "HFOV", 90.0)
        else:
            self.config = config
            self.forward_step_size = config.get("FORWARD_STEP_SIZE", 0.25)
            self.turn_angle = config.get("TURN_ANGLE", 15.0)
            rgb_config = config.get("RGB_SENSOR", {})
            self.rgb_width = rgb_config.get("WIDTH", 224)
            self.rgb_height = rgb_config.get("HEIGHT", 224)
            self.rgb_hfov = rgb_config.get("HFOV", 90.0)
        
        # Current agent state (will be set by reset/set_agent_state)
        self._agent_state: Optional[AgentState] = None
        
        # Current scene ID
        self._scene_id: Optional[str] = None
        
        # TODO: Initialize satsim engine when available
        # self._satsim = SatSim(config)
    
    def reset(self, scene_id: str) -> Observations:
        """Reset the simulator and load a new scene.
        
        Args:
            scene_id: Identifier for the scene to load.
            
        Returns:
            Initial observations from the simulator (currently empty dict).
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
        """
        self._scene_id = scene_id
        
        # TODO: Call satsim.reset(scene_id) to load satellite map scene
        # TODO: Set initial agent state from satsim
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "satsim.reset() not yet implemented. "
            "This method should load the satellite map scene and initialize the agent state."
        )
    
    def step(self, action: Union[int, str, Dict[str, Any]]) -> Observations:
        """Execute an action in the simulator.
        
        Args:
            action: Action to execute. Can be:
                - Action name: "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"
                - Action index: 0, 1, 2, 3
                - Action dictionary: {"action": "MOVE_FORWARD"}
                
        Returns:
            Observations after executing the action.
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
        """
        # TODO: Parse action and call satsim.step(action)
        # TODO: Update agent state
        # TODO: Return observations
        raise NotImplementedError(
            "satsim.step() not yet implemented. "
            "This method should execute the action and update the agent state."
        )
    
    def get_agent_state(self) -> AgentState:
        """Get the current state of the agent.
        
        Returns:
            Current agent state containing position and rotation.
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
            RuntimeError: If agent state has not been initialized.
        """
        if self._agent_state is None:
            raise RuntimeError(
                "Agent state not initialized. Call reset() or set_agent_state() first."
            )
        
        # TODO: Get state from satsim engine
        # For now, return stored state (if available)
        return self._agent_state
    
    def set_agent_state(
        self,
        position: Union[List[float], np.ndarray],
        rotation: float
    ) -> None:
        """Set the state of the agent.
        
        Args:
            position: New position as [longitude, latitude, altitude].
            rotation: New rotation as roll angle in degrees (0-360, 0 = North).
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
        """
        # Convert position to numpy array
        position = np.array(position, dtype=np.float32)
        
        if len(position) != 3:
            raise ValueError(
                f"Position must have 3 elements [longitude, latitude, altitude], "
                f"got {len(position)} elements"
            )
        
        # Normalize rotation to [0, 360)
        rotation = rotation % 360.0
        
        # Update stored state
        self._agent_state = AgentState(position=position, rotation=rotation)
        
        # TODO: Call satsim.set_agent_state(position, rotation)
        # For now, just store the state
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from all sensors.
        
        This method simulates a drone's overhead camera. It calculates the ground
        coverage based on the agent's altitude and camera HFOV, then crops the
        corresponding region from the satellite map.
        
        Camera model:
        - Ground_Width = 2 * Altitude * tan(HFOV / 2)
        - GSD (Ground Sample Distance) = Ground_Width / WIDTH
        
        Returns:
            Dictionary containing observations from all sensors:
                - "rgb": RGB image as numpy array (H, W, 3) uint8
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
            RuntimeError: If agent state has not been initialized.
        """
        if self._agent_state is None:
            raise RuntimeError(
                "Agent state not initialized. Call reset() or set_agent_state() first."
            )
        
        # TODO: Get RGB observation from satsim
        # The satsim should:
        # 1. Calculate ground coverage based on altitude and HFOV
        # 2. Crop satellite map image centered at (longitude, latitude)
        # 3. Rotate image according to roll angle
        # 4. Return RGB image as (H, W, 3) uint8 array
        
        raise NotImplementedError(
            "satsim.get_observations() not yet implemented. "
            "This method should crop and return RGB image from satellite map."
        )
    
    def geodesic_distance(
        self,
        position_a: Union[List[float], np.ndarray],
        position_b: Union[List[float], np.ndarray]
    ) -> float:
        """Calculate geodesic distance between two positions.
        
        Uses the Haversine formula to calculate great-circle distance
        between two points on Earth's surface, considering altitude difference.
        
        Args:
            position_a: First position as [longitude, latitude, altitude].
            position_b: Second position as [longitude, latitude, altitude].
            
        Returns:
            Distance in meters between the two positions (3D distance including altitude).
        """
        position_a = np.array(position_a, dtype=np.float32)
        position_b = np.array(position_b, dtype=np.float32)
        
        # Use utility function that considers altitude
        return geodesic_distance_with_altitude(position_a, position_b)
    
    def is_navigable(self, position: Union[List[float], np.ndarray]) -> bool:
        """Check if a position is navigable.
        
        Args:
            position: Position to check as [longitude, latitude, altitude].
            
        Returns:
            True if the position is navigable, False otherwise.
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
        """
        # TODO: Call satsim.is_navigable(position)
        # The satsim should check if the position is:
        # - Within valid geographic bounds
        # - Not in restricted areas (water, buildings, etc.)
        # - At a valid altitude
        
        raise NotImplementedError(
            "satsim.is_navigable() not yet implemented. "
            "This method should check if a position is navigable."
        )
    
    def sample_navigable_point(self) -> List[float]:
        """Sample a random navigable point in the current scene.
        
        Returns:
            A navigable position as [longitude, latitude, altitude].
            
        Raises:
            NotImplementedError: satsim engine not yet implemented.
            RuntimeError: If scene has not been loaded.
        """
        if self._scene_id is None:
            raise RuntimeError(
                "Scene not loaded. Call reset(scene_id) first."
            )
        
        # TODO: Call satsim.sample_navigable_point()
        # The satsim should:
        # 1. Get valid navigable regions in the current scene
        # 2. Sample a random point from these regions
        # 3. Return [longitude, latitude, altitude]
        
        raise NotImplementedError(
            "satsim.sample_navigable_point() not yet implemented. "
            "This method should sample a random navigable point in the scene."
        )
    
    @property
    def sensor_suite(self):
        """Get the sensor suite.
        
        Returns:
            Dictionary containing sensor configurations.
        """
        return {
            "rgb": {
                "width": self.rgb_width,
                "height": self.rgb_height,
                "hfov": self.rgb_hfov,
            }
        }
    
    @property
    def action_space(self):
        """Get the action space.
        
        Returns:
            List of available action names.
        """
        return ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    
    @property
    def scene_id(self) -> Optional[str]:
        """Get the current scene ID.
        
        Returns:
            Current scene ID, or None if no scene is loaded.
        """
        return self._scene_id


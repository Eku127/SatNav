#!/usr/bin/env python3
"""SatSim wrapper implementation for SatNav."""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig

from satnav.core.simulator import AgentState, Observations, Simulator
from satnav.core.utils import geodesic_distance_with_altitude
from satnav.sims.satsim import SatSim


class SatSimWrapper(Simulator):
    """Wrapper for the satsim satellite map simulator.
    
    This class wraps the satsim simulator and implements the Simulator interface.
    It handles scene path resolution and delegates all operations to the SatSim engine.
    
    The simulator uses geographic coordinates (longitude, latitude, altitude)
    and roll angle (heading) for agent state. RGB observations are generated
    by cropping satellite map imagery based on the agent's position, altitude,
    and camera HFOV.
    """
    
    def __init__(self, config: Union[DictConfig, dict], scenes_dir: Optional[str] = None) -> None:
        """Initialize the simulator wrapper.
        
        Args:
            config: Simulator configuration containing:
                - FORWARD_STEP_SIZE: Step size for forward movement (meters)
                - TURN_ANGLE: Angle for turning (degrees)
                - RGB_SENSOR: RGB sensor configuration
                    - WIDTH: Image width in pixels
                    - HEIGHT: Image height in pixels
                    - HFOV: Horizontal field of view in degrees
            scenes_dir: Optional directory containing scene datasets. If None,
                scene_id is assumed to be a full path.
        """
        # Store config
        self.config = config
        
        # Extract scenes_dir from config if available
        if scenes_dir is None:
            if isinstance(config, DictConfig):
                # Try to get scenes_dir from parent config if available
                # This handles the case where full config is passed
                if hasattr(config, "DATASET"):
                    scenes_dir = getattr(config.DATASET, "SCENES_DIR", None)
            else:
                # Check if config contains DATASET
                if "DATASET" in config and isinstance(config["DATASET"], dict):
                    scenes_dir = config["DATASET"].get("SCENES_DIR")
        
        self._scenes_dir = scenes_dir
        
        # Initialize SatSim engine with SIMULATOR config
        if isinstance(config, DictConfig):
            sim_config = getattr(config, "SIMULATOR", config)
        else:
            sim_config = config.get("SIMULATOR", config)
        
        self._satsim = SatSim(sim_config)
        
        # Current scene ID (for reference)
        self._scene_id: Optional[str] = None
    
    def reset(self, scene_id: str) -> Observations:
        """Reset the simulator and load a new scene.
        
        Args:
            scene_id: Identifier for the scene to load. If scenes_dir is set,
                this will be combined with scenes_dir to form the full path.
                Otherwise, scene_id is treated as a full path.
            
        Returns:
            Initial observations from the simulator. Returns empty dict if
            agent state has not been set yet (should be set via set_agent_state).
        """
        # Check if we need to load a new scene
        # If scene_id is the same, skip loading (scene is already loaded)
        # This optimization avoids path processing and load_scene() overhead
        # Note: SatSim.load_scene() already has caching, but checking scene_id
        # first avoids unnecessary path processing and function calls
        scene_changed = (self._scene_id != scene_id)
        if scene_changed:
            # Combine scene path
            scene_path = self._combine_scene_path(scene_id)
            # Load scene in SatSim (will use cache if already loaded)
            self._satsim.load_scene(scene_path)
            self._scene_id = scene_id
        # else: scene_id is the same, scene is already loaded, skip loading
        
        # Return initial observations if agent state is set AND scene hasn't changed
        # If scene changed, agent state needs to be updated via set_agent_state() first
        # (old agent position from previous scene would be invalid in new scene)
        if scene_changed:
            # Scene changed, don't try to get observations yet
            # Agent state must be set via set_agent_state() before getting observations
            return {}
        
        try:
            return self._satsim.get_observations()
        except RuntimeError:
            # Agent state not set yet, return empty observations
            return {}
    
    def step(self, action: Union[int, str, Dict[str, Any]]) -> Observations:
        """Execute an action in the simulator.
        
        Args:
            action: Action to execute. Can be:
                - Action name: "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"
                - Action index: 0, 1, 2, 3
                - Action dictionary: {"action": "MOVE_FORWARD"}
                
        Returns:
            Observations after executing the action.
        """
        return self._satsim.step(action)
    
    def get_agent_state(self) -> AgentState:
        """Get the current state of the agent.
        
        Returns:
            Current agent state containing position and rotation.
            
        Raises:
            RuntimeError: If agent state has not been initialized.
        """
        position, rotation = self._satsim.get_agent_state()
        return AgentState(position=position, rotation=rotation)
    
    def set_agent_state(
        self,
        position: Union[List[float], np.ndarray],
        rotation: float
    ) -> None:
        """Set the state of the agent.
        
        Args:
            position: New position as [longitude, latitude, altitude].
            rotation: New rotation as roll angle in degrees (0-360, 0 = North).
        """
        self._satsim.set_agent_state(position, rotation)
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from all sensors.
        
        This method simulates a drone's overhead camera. It calculates the ground
        coverage based on the agent's altitude and camera HFOV, then crops the
        corresponding region from the satellite map.
        
        Camera model:
        - Ground_Width = 2 * Altitude * tan(HFOV / 2)  (HFOV controls horizontal direction)
        - Ground_Height = Ground_Width / aspect_ratio
        - GSD (Ground Sample Distance) = Ground_Width / WIDTH
        
        Returns:
            Dictionary containing observations from all sensors:
                - "rgb": RGB image as numpy array (H, W, 3) uint8
        """
        return self._satsim.get_observations()
    
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
            True if the position is navigable (within scene bounds), False otherwise.
        """
        return self._satsim.is_navigable(position)
    
    def sample_navigable_point(self) -> List[float]:
        """Sample a random navigable point in the current scene.
        
        Returns:
            A navigable position as [longitude, latitude, altitude].
            
        Raises:
            RuntimeError: If scene has not been loaded.
            NotImplementedError: This method is not yet implemented in SatSim.
        """
        if self._scene_id is None:
            raise RuntimeError(
                "Scene not loaded. Call reset(scene_id) first."
            )
        
        # TODO: Implement sampling in SatSim
        raise NotImplementedError(
            "sample_navigable_point() not yet implemented. "
            "This method should sample a random navigable point in the scene."
        )
    
    def _combine_scene_path(self, scene_id: str) -> str:
        """Combine scenes_dir with scene_id to form full path.
        
        Args:
            scene_id: Scene identifier (may be relative or absolute path).
            
        Returns:
            Full path to scene file.
        """
        if self._scenes_dir is None:
            # If no scenes_dir, assume scene_id is already a full path
            return scene_id
        
        # Check if scene_id is already an absolute path or contains path separators
        # (meaning it's already a full path from dataset)
        if os.path.isabs(scene_id) or os.sep in scene_id or '/' in scene_id:
            # scene_id is already a full path, use it directly
            return scene_id
        
        # Combine scenes_dir with scene_id
        scene_path = os.path.join(self._scenes_dir, scene_id)
        return scene_path
    
    @property
    def sensor_suite(self):
        """Get the sensor suite.
        
        Returns:
            Dictionary containing sensor configurations.
        """
        return {
            "rgb": {
                "width": self._satsim._camera.width,
                "height": self._satsim._camera.height,
                "hfov": self._satsim._camera.hfov,
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
    
    @property
    def forward_step_size(self) -> float:
        """Get the forward step size in meters.
        
        Returns:
            Forward step size in meters.
        """
        return self._satsim.forward_step_size
    
    @property
    def turn_angle(self) -> float:
        """Get the turn angle in degrees.
        
        Returns:
            Turn angle in degrees.
        """
        return self._satsim.turn_angle
    
    @property
    def rgb_width(self) -> int:
        """Get the RGB image width in pixels.
        
        Returns:
            RGB image width in pixels.
        """
        return self._satsim._camera.width
    
    @property
    def rgb_height(self) -> int:
        """Get the RGB image height in pixels.
        
        Returns:
            RGB image height in pixels.
        """
        return self._satsim._camera.height
    
    @property
    def rgb_hfov(self) -> float:
        """Get the RGB camera horizontal field of view in degrees.
        
        Returns:
            RGB camera HFOV in degrees.
        """
        return self._satsim._camera.hfov


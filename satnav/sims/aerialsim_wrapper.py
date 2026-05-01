#!/usr/bin/env python3
from __future__ import annotations

"""AerialSim wrapper implementation for SatNav.

This wrapper provides the Simulator interface for AerialSim, allowing seamless
switching between SatSim (2D satellite imagery) and AerialSim (3D aerial view).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from satnav.core.simulator import AgentState, Observations, Simulator
from satnav.core.utils import geodesic_distance_with_altitude
from satnav.sims.aerialsim import AerialSim


class AerialSimWrapper(Simulator):
    """Wrapper for the AerialSim 3D aerial view simulator.
    
    This class wraps the AerialSim engine and implements the Simulator interface.
    It provides 3D aerial views using Google's Photorealistic 3D Tiles.
    
    The simulator uses geographic coordinates (longitude, latitude, altitude)
    and rotation angle (heading) for agent state. RGB observations are generated
    by rendering 3D tiles from the agent's viewpoint.
    
    For TOP_DOWN_MAP compatibility, this wrapper also loads the satellite TIF
    file (same as SatSim) to provide map visualization.
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
                - AERIAL: AerialSim-specific settings
                    - API_KEY: Google 3D Tiles API key
                    - BROWSER: Browser to use (chrome/edge/firefox)
                    - HEADLESS: Whether to run in headless mode (default: True)
            scenes_dir: Optional directory containing scene datasets. Used for
                loading satellite TIF files for TOP_DOWN_MAP visualization.
        """
        self.config = config
        
        # Extract scenes_dir from config if not provided
        if scenes_dir is None:
            if isinstance(config, DictConfig):
                if hasattr(config, "DATASET"):
                    scenes_dir = getattr(config.DATASET, "SCENES_DIR", None)
            else:
                if "DATASET" in config and isinstance(config["DATASET"], dict):
                    scenes_dir = config["DATASET"].get("SCENES_DIR")
        
        self._scenes_dir = scenes_dir
        
        # Initialize AerialSim engine with SIMULATOR config
        if isinstance(config, DictConfig):
            sim_config = getattr(config, "SIMULATOR", config)
        else:
            sim_config = config.get("SIMULATOR", config)
        
        self._aerialsim = AerialSim(sim_config)
        
        # Current scene ID (for reference)
        self._scene_id: Optional[str] = None
        
        # Satellite map TIF for TOP_DOWN_MAP visualization (loaded on reset)
        self._current_scene = None
        self._scene_cache: Dict[str, Any] = {}
    
    def reset(self, scene_id: str) -> Observations:
        """Reset the simulator and load a new scene.
        
        For AerialSim, this stores the scene_id for reference and loads the
        corresponding satellite TIF file for TOP_DOWN_MAP visualization.
        
        Args:
            scene_id: Identifier for the scene. If scenes_dir is set,
                this will be combined with scenes_dir to form the full path
                for loading the satellite TIF file.
            
        Returns:
            Empty observations. Agent state must be set via set_agent_state()
            before getting observations.
        """
        if self._scene_id != scene_id:
            self._aerialsim.load_scene(scene_id)
            self._scene_id = scene_id
            
            # Load satellite TIF for TOP_DOWN_MAP visualization
            self._load_satellite_map(scene_id)
        
        # Return empty observations (agent state must be set first)
        return {}
    
    def _load_satellite_map(self, scene_id: str) -> None:
        """Load satellite TIF file for TOP_DOWN_MAP visualization.
        
        This allows AerialSim to use the same TOP_DOWN_MAP visualization
        as SatSim, showing the satellite map with agent trajectory overlay.
        
        Args:
            scene_id: Scene identifier or path.
        """
        # Determine full path to TIF file
        if self._scenes_dir:
            scene_path = Path(self._scenes_dir) / scene_id
        else:
            scene_path = Path(scene_id)
        
        # Add .tif extension if not present
        if not str(scene_path).lower().endswith('.tif'):
            scene_path = Path(str(scene_path) + '.tif')
        
        scene_path_str = str(scene_path)
        
        # Check cache first
        if scene_path_str in self._scene_cache:
            self._current_scene = self._scene_cache[scene_path_str]
            return
        
        # Check if file exists
        if not scene_path.exists():
            # TIF file not found - TOP_DOWN_MAP will not work, but RGB rendering is fine
            print(f"Warning: Satellite TIF not found: {scene_path}")
            print("  TOP_DOWN_MAP visualization will not be available.")
            self._current_scene = None
            return
        
        try:
            # Import and load the TIF (function defined in satsim.py)
            from satnav.sims.satsim.satsim import open_reprojected_to_epsg3857
            self._current_scene = open_reprojected_to_epsg3857(scene_path_str)
            self._scene_cache[scene_path_str] = self._current_scene
        except Exception as e:
            print(f"Warning: Failed to load satellite TIF: {e}")
            print("  TOP_DOWN_MAP visualization will not be available.")
            self._current_scene = None
    
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
        return self._aerialsim.step(action)
    
    def get_agent_state(self) -> AgentState:
        """Get the current state of the agent.
        
        Returns:
            Current agent state containing position and rotation.
            
        Raises:
            RuntimeError: If agent state has not been initialized.
        """
        position, rotation = self._aerialsim.get_agent_state()
        return AgentState(position=position, rotation=rotation)

    def get_aerial_diagnostics(self) -> Dict[str, Any]:
        """Return public AerialSim diagnostics for application code."""
        return self._aerialsim.get_diagnostics()

    def set_ground_height_cache_threshold(self, threshold_meters: float) -> None:
        """Configure AerialSim ground-height cache reuse distance."""
        self._aerialsim.set_ground_height_cache_threshold(threshold_meters)
    
    def set_agent_state(
        self,
        position: List[float],
        rotation: float
    ) -> None:
        """Set the state of the agent.
        
        Args:
            position: New position as [longitude, latitude, altitude].
            rotation: New rotation as heading in degrees (0=North, clockwise).
        """
        self._aerialsim.set_agent_state(position, rotation)
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from all sensors.
        
        This renders the 3D aerial view at the agent's current position.
        
        Returns:
            Dictionary containing observations from all sensors:
                - "rgb": RGB image as numpy array (H, W, 3) uint8
        """
        return self._aerialsim.get_observations()
    
    def geodesic_distance(
        self,
        position_a: List[float],
        position_b: List[float]
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
        return geodesic_distance_with_altitude(position_a, position_b)
    
    def is_navigable(self, position: List[float]) -> bool:
        """Check if a position is navigable.
        
        For AerialSim with global 3D tiles, all positions are considered
        navigable since we don't have explicit scene bounds.
        
        Args:
            position: Position to check as [longitude, latitude, altitude].
            
        Returns:
            True (always navigable for global 3D tiles).
        """
        return self._aerialsim.is_navigable(position)
    
    def sample_navigable_point(self) -> List[float]:
        """Sample a random navigable point in the current scene.
        
        Returns:
            A navigable position as [longitude, latitude, altitude].
            
        Raises:
            NotImplementedError: This method is not implemented for AerialSim.
        """
        raise NotImplementedError(
            "sample_navigable_point() not implemented for AerialSim. "
            "AerialSim uses global 3D tiles without explicit scene bounds."
        )
    
    @property
    def sensor_suite(self):
        """Get the sensor suite.
        
        Returns:
            Dictionary containing sensor configurations.
        """
        return {
            "rgb": {
                "width": self._aerialsim._camera.width,
                "height": self._aerialsim._camera.height,
                "hfov": self._aerialsim._camera.hfov,
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
        """Get the forward step size in meters."""
        return self._aerialsim.forward_step_size
    
    @property
    def turn_angle(self) -> float:
        """Get the turn angle in degrees."""
        return self._aerialsim.turn_angle
    
    @property
    def rgb_width(self) -> int:
        """Get the RGB image width in pixels."""
        return self._aerialsim._camera.width
    
    @property
    def rgb_height(self) -> int:
        """Get the RGB image height in pixels."""
        return self._aerialsim._camera.height
    
    @property
    def rgb_hfov(self) -> float:
        """Get the RGB camera horizontal field of view in degrees."""
        return self._aerialsim._camera.hfov
    
    def close(self) -> None:
        """Close the simulator and cleanup resources."""
        self._aerialsim.close()

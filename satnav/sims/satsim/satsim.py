#!/usr/bin/env python3
"""Core satellite map simulator engine."""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.vrt import WarpedVRT

from satnav.sims.satsim.camera import SatelliteCamera
from satnav.sims.satsim.geoutils import GeoUtils


def open_reprojected_to_epsg3857(src_path: Union[str, Path]) -> rasterio.DatasetReader:
    """Open raster file, reprojecting to EPSG:3857 if needed.
    
    Args:
        src_path: Path to raster file.
        
    Returns:
        Rasterio dataset reader (reprojected to EPSG:3857 if needed).
    """
    dst_crs = "EPSG:3857"
    src = rasterio.open(src_path)
    
    if src.crs is not None and src.crs.to_epsg() == 3857:
        return src
    else:
        vrt = WarpedVRT(src, crs=dst_crs)
        warnings.warn(
            f"Reprojecting raster {src_path} to EPSG:3857, which may cause distortion",
            UserWarning
        )
        return vrt


class SatSim:
    """Core satellite map simulator engine.
    
    This class handles:
    - Loading and caching satellite TIF scenes
    - Coordinate transformation between WGS84 and EPSG:3857
    - Action execution (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
    - RGB observation generation by cropping satellite imagery
    
    The simulator uses EPSG:3857 (Web Mercator) internally for all operations,
    but exposes WGS84 coordinates through its public API.
    """
    
    def __init__(self, config: Union[DictConfig, dict]):
        """Initialize SatSim simulator.
        
        Args:
            config: Simulator configuration containing:
                - FORWARD_STEP_SIZE: Step size for forward movement (meters)
                - TURN_ANGLE: Angle for turning (degrees)
                - RGB_SENSOR: RGB sensor configuration
                    - WIDTH: Image width in pixels
                    - HEIGHT: Image height in pixels
                    - HFOV: Horizontal field of view in degrees
        """
        # Extract configuration values
        if isinstance(config, DictConfig):
            self.forward_step_size = getattr(config, "FORWARD_STEP_SIZE", 0.25)
            self.turn_angle = getattr(config, "TURN_ANGLE", 15.0)
            rgb_config = getattr(config, "RGB_SENSOR", {})
            rgb_width = getattr(rgb_config, "WIDTH", 224)
            rgb_height = getattr(rgb_config, "HEIGHT", 224)
            rgb_hfov = getattr(rgb_config, "HFOV", 90.0)
        else:
            self.forward_step_size = config.get("FORWARD_STEP_SIZE", 0.25)
            self.turn_angle = config.get("TURN_ANGLE", 15.0)
            rgb_config = config.get("RGB_SENSOR", {})
            rgb_width = rgb_config.get("WIDTH", 224)
            rgb_height = rgb_config.get("HEIGHT", 224)
            rgb_hfov = rgb_config.get("HFOV", 90.0)
        
        # Initialize components
        self._camera = SatelliteCamera(rgb_width, rgb_height, rgb_hfov)
        
        # Scene management
        self._current_scene: Optional[rasterio.DatasetReader] = None
        self._scene_cache: Dict[str, rasterio.DatasetReader] = {}
        
        # Agent state (stored in EPSG:3857 internally)
        # Position: (x_mercator, y_mercator, altitude)
        self._agent_position: Optional[Tuple[float, float, float]] = None
        self._agent_rotation: Optional[float] = None  # degrees, 0=North
    
    def load_scene(self, scene_path: Union[str, Path]) -> bool:
        """Load a satellite scene from file.
        
        Args:
            scene_path: Path to the scene TIF file. If it doesn't have .tif
                extension, it will be appended.
                
        Returns:
            True if scene loaded successfully.
            
        Raises:
            FileNotFoundError: If scene file doesn't exist.
            rasterio.errors.RasterioIOError: If file cannot be opened.
        """
        scene_path = Path(scene_path)
        
        # Add .tif extension if not present
        # Check if path ends with .tif (case-insensitive)
        if not str(scene_path).lower().endswith('.tif'):
            # Append .tif instead of using with_suffix to avoid issues with
            # filenames containing dots (e.g., map_center_40.712800_-74.006000.tif)
            scene_path = Path(str(scene_path) + '.tif')
        
        scene_path_str = str(scene_path)
        
        # Check cache first
        if scene_path_str in self._scene_cache:
            self._current_scene = self._scene_cache[scene_path_str]
            return True
        
        # Check if file exists
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        
        # Open and reproject if needed
        self._current_scene = open_reprojected_to_epsg3857(scene_path_str)
        
        # Cache the scene
        self._scene_cache[scene_path_str] = self._current_scene
        
        return True
    
    def set_agent_state(
        self,
        position_wgs84: Union[List[float], np.ndarray],
        rotation: float,
        validate_navigable: bool = True
    ) -> None:
        """Set the agent state.
        
        Args:
            position_wgs84: Position as [longitude, latitude, altitude].
            rotation: Rotation as roll angle in degrees (0-360, 0=North).
            validate_navigable: If True, validate that the position is within
                safe navigable bounds. Default: True.
        
        Raises:
            ValueError: If position does not have 3 elements.
            ValueError: If validate_navigable=True and position is not within
                safe navigable bounds (camera view would exceed map bounds).
        """
        position_wgs84 = np.array(position_wgs84, dtype=np.float32)
        
        if len(position_wgs84) != 3:
            raise ValueError(
                f"Position must have 3 elements [longitude, latitude, altitude], "
                f"got {len(position_wgs84)} elements"
            )
        
        # Validate position is navigable (within safe bounds)
        if validate_navigable and self._current_scene is not None:
            if not self.is_navigable(position_wgs84):
                # Get bounds for error message
                bounds = self._current_scene.bounds
                margin = self._camera.get_margin(position_wgs84[2])
                raise ValueError(
                    f"Initial position {position_wgs84.tolist()} is not within safe navigable bounds. "
                    f"Camera view would exceed map boundaries. "
                    f"Scene bounds: left={bounds.left:.1f}, right={bounds.right:.1f}, "
                    f"bottom={bounds.bottom:.1f}, top={bounds.top:.1f}. "
                    f"Required margin for altitude {position_wgs84[2]}m: {margin:.1f}m"
                )
        
        # Convert to Mercator for internal storage
        self._agent_position = GeoUtils.position_wgs84_to_mercator(position_wgs84)
        
        # Normalize rotation to [0, 360)
        self._agent_rotation = rotation % 360.0
    
    def get_agent_state(self) -> Tuple[np.ndarray, float]:
        """Get the current agent state.
        
        Returns:
            Tuple of (position_wgs84, rotation) where:
                - position_wgs84: [longitude, latitude, altitude]
                - rotation: Roll angle in degrees (0=North)
                
        Raises:
            RuntimeError: If agent state has not been initialized.
        """
        if self._agent_position is None or self._agent_rotation is None:
            raise RuntimeError(
                "Agent state not initialized. Call set_agent_state() first."
            )
        
        # Convert back to WGS84
        position_wgs84 = GeoUtils.position_mercator_to_wgs84(self._agent_position)
        
        return (position_wgs84, self._agent_rotation)
    
    def step(self, action: Union[str, int, Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Execute an action and return observations.
        
        Args:
            action: Action to execute. Can be:
                - Action name: "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"
                - Action index: 0, 1, 2, 3
                - Action dictionary: {"action": "MOVE_FORWARD"}
                
        Returns:
            Dictionary containing observations:
                - "rgb": RGB image as numpy array (H, W, 3) uint8
                
        Raises:
            RuntimeError: If agent state or scene not initialized.
            ValueError: If action is invalid.
        """
        if self._agent_position is None or self._agent_rotation is None:
            raise RuntimeError(
                "Agent state not initialized. Call set_agent_state() first."
            )
        
        if self._current_scene is None:
            raise RuntimeError(
                "Scene not loaded. Call load_scene() first."
            )
        
        # Parse action
        action_str = self._parse_action(action)
        
        # Execute action in EPSG:3857 space
        x, y, alt = self._agent_position
        rotation = self._agent_rotation
        
        if action_str == "MOVE_FORWARD":
            x_new, y_new = GeoUtils.move_in_mercator(
                x, y, self.forward_step_size, rotation
            )
            # Check if new position is navigable (within safe bounds)
            # This prevents agent from moving to positions where camera view
            # would exceed map bounds, regardless of rotation angle
            position_new_wgs84 = GeoUtils.position_mercator_to_wgs84(
                np.array([x_new, y_new, alt], dtype=np.float32)
            )
            if self.is_navigable(position_new_wgs84):
                self._agent_position = (x_new, y_new, alt)
            # If not navigable, agent stays at current position (no update)
        elif action_str == "TURN_LEFT":
            self._agent_rotation = (rotation - self.turn_angle) % 360.0
        elif action_str == "TURN_RIGHT":
            self._agent_rotation = (rotation + self.turn_angle) % 360.0
        elif action_str == "STOP":
            # No movement
            pass
        else:
            raise ValueError(f"Invalid action: {action_str}")
        
        # Return observations
        return self.get_observations()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations from all sensors.
        
        Returns:
            Dictionary containing observations:
                - "rgb": RGB image as numpy array (H, W, 3) uint8
                
        Raises:
            RuntimeError: If agent state or scene not initialized.
        """
        if self._agent_position is None or self._agent_rotation is None:
            raise RuntimeError(
                "Agent state not initialized. Call set_agent_state() first."
            )
        
        if self._current_scene is None:
            raise RuntimeError(
                "Scene not loaded. Call load_scene() first."
            )
        
        # Render RGB image
        x, y, alt = self._agent_position
        rgb_image = self._camera.render_image(
            self._current_scene,
            (x, y),
            alt,
            self._agent_rotation
        )
        
        return {"rgb": rgb_image}
    
    def is_navigable(self, position_wgs84: Union[List[float], np.ndarray]) -> bool:
        """Check if a position is navigable.
        
        This method checks if the agent can be placed at the given position without
        the camera view exceeding map bounds. It uses a safety margin based on camera
        parameters and altitude to ensure that regardless of rotation angle, the
        camera view will always stay within map bounds.
        
        Args:
            position_wgs84: Position to check as [longitude, latitude, altitude].
            
        Returns:
            True if the position is within safe navigable bounds, False otherwise.
            Safe bounds are computed by shrinking the map bounds by the safety margin
            (which depends on altitude and camera parameters).
            
        Raises:
            RuntimeError: If scene not loaded.
        """
        if self._current_scene is None:
            raise RuntimeError(
                "Scene not loaded. Call load_scene() first."
            )
        
        # Convert to Mercator coordinates
        position_mercator = GeoUtils.position_wgs84_to_mercator(
            np.array(position_wgs84, dtype=np.float32)
        )
        x, y, altitude = position_mercator
        
        # Get map bounds
        bounds = self._current_scene.bounds
        
        # Calculate safety margin based on altitude and camera parameters
        # The margin represents the maximum distance from camera center to any corner
        # of the view rectangle when rotated. By ensuring the agent stays at least
        # this distance away from map edges, we guarantee the camera view will never
        # exceed map bounds regardless of rotation angle.
        margin = self._camera.get_margin(altitude)
        
        # Add a small epsilon to account for floating-point precision errors
        # in coordinate transformations and margin calculations.
        # Without this, edge cases can slip through (e.g., 0.09m difference)
        epsilon = 1.0  # 1 meter safety buffer
        margin = margin + epsilon
        
        # Compute safe navigable bounds by shrinking map bounds by the margin
        # This creates a "safe zone" where the agent can move freely without
        # risking camera view exceeding map boundaries
        safe_left = bounds.left + margin
        safe_right = bounds.right - margin
        safe_bottom = bounds.bottom + margin
        safe_top = bounds.top - margin
        
        # Check if position is within safe bounds
        # Also check if safe bounds are valid (map is large enough)
        if safe_left >= safe_right or safe_bottom >= safe_top:
            # Map is too small for the given altitude, no position is safe
            return False
        
        return (
            safe_left <= x <= safe_right and
            safe_bottom <= y <= safe_top
        )
    
    def get_scene_bounds(self) -> Tuple[float, float, float, float]:
        """Get current scene bounds in WGS84 coordinates.
        
        Returns:
            Tuple of (left, right, bottom, top) in degrees (WGS84).
            
        Raises:
            RuntimeError: If scene not loaded.
        """
        if self._current_scene is None:
            raise RuntimeError(
                "Scene not loaded. Call load_scene() first."
            )
        
        # Get bounds in Mercator
        bounds = self._current_scene.bounds
        
        # Convert to WGS84
        left_lon, bottom_lat = GeoUtils.mercator_to_wgs84(bounds.left, bounds.bottom)
        right_lon, top_lat = GeoUtils.mercator_to_wgs84(bounds.right, bounds.top)
        
        return (left_lon, right_lon, bottom_lat, top_lat)
    
    def _parse_action(self, action: Union[str, int, Dict[str, str]]) -> str:
        """Parse action from various input formats.
        
        Args:
            action: Action in various formats.
            
        Returns:
            Action string.
        """
        if isinstance(action, str):
            return action
        elif isinstance(action, int):
            action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
            if 0 <= action < len(action_names):
                return action_names[action]
            else:
                raise ValueError(f"Action index {action} out of range [0, {len(action_names)})")
        elif isinstance(action, dict):
            if "action" in action:
                return action["action"]
            else:
                raise ValueError(f"Action dictionary must contain 'action' key: {action}")
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
    
    def close(self):
        """Close all open scenes and clear cache."""
        # Close cached scenes
        for scene in self._scene_cache.values():
            scene.close()
        
        self._scene_cache.clear()
        self._current_scene = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


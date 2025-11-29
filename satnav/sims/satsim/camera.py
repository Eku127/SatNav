#!/usr/bin/env python3
"""Camera/rendering logic for satellite imagery observations."""

import math
from typing import Tuple, Union

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import rotate


class SatelliteCamera:
    """Camera for rendering satellite imagery observations.
    
    This class handles the rendering of RGB images from satellite TIF files
    based on agent position, altitude, and camera parameters. It calculates
    ground coverage bounds and crops/rotates the satellite imagery accordingly.
    """
    
    def __init__(self, width: int, height: int, hfov: float):
        """Initialize satellite camera.
        
        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            hfov: Horizontal field of view in degrees.
        """
        self.width = width
        self.height = height
        self.hfov = hfov
        
        # Calculate aspect ratio
        self.aspect_ratio = width / height
        
        # Pre-compute margin factor for boundary checking optimization
        # The margin (safety distance from map edge) is linearly proportional to altitude:
        #   margin = altitude * margin_factor
        # where margin_factor = sqrt(tan^2(HFOV/2) * (1 + aspect_ratio^2))
        # 
        # This factor represents the maximum distance from camera center to any corner
        # of the view rectangle when rotated. By ensuring agent stays at least this
        # distance away from map edges, we guarantee that camera view will never exceed
        # map bounds regardless of rotation angle.
        tan_half_hfov = math.tan(math.radians(self.hfov / 2.0))
        self._margin_factor = math.sqrt(
            tan_half_hfov ** 2 * (1 + self.aspect_ratio ** 2)
        )
    
    def get_view_bounds(
        self,
        position_mercator: Tuple[float, float],
        altitude: float,
        rotation: float
    ) -> Tuple[float, float, float, float]:
        """Calculate ground coverage bounds in Mercator coordinates.
        
        This method calculates the bounding box of the ground area visible
        to the camera, accounting for altitude, field of view, and rotation.
        
        Args:
            position_mercator: Agent position as (x, y) in meters (Web Mercator).
            altitude: Agent altitude in meters.
            rotation: Camera rotation (roll angle) in degrees (0=North).
            
        Returns:
            Tuple of (left, right, bottom, top) in meters (Web Mercator).
        """
        cx, cy = position_mercator
        h = altitude
        
        # Ground half-height/width in meters
        # Vertical half-width (north-south direction)
        half_y_m = h * math.tan(math.radians(self.hfov / 2.0))
        # Horizontal half-width (east-west direction)
        half_x_m = half_y_m * self.aspect_ratio
        
        # Distance to four corners (diagonal half-distance)
        r = math.hypot(half_x_m, half_y_m)
        
        # Base azimuth angles for four corners (relative to North)
        base_az_deg = [
            math.degrees(math.atan2(+half_x_m, +half_y_m)),  # NE
            math.degrees(math.atan2(-half_x_m, +half_y_m)),  # NW
            math.degrees(math.atan2(-half_x_m, -half_y_m)),  # SW
            math.degrees(math.atan2(+half_x_m, -half_y_m)),  # SE
        ]
        
        # Apply rotation
        az_deg = [((az + rotation) % 360.0) for az in base_az_deg]
        
        # Calculate corner positions in Mercator
        corners = [
            (
                cx + r * math.sin(math.radians(az)),
                cy + r * math.cos(math.radians(az))
            )
            for az in az_deg
        ]
        
        # Calculate bounding box
        xs = [x for x, y in corners]
        ys = [y for x, y in corners]
        left = min(xs)
        right = max(xs)
        bottom = min(ys)
        top = max(ys)
        
        return (left, right, bottom, top)
    
    def get_margin(self, altitude: float) -> float:
        """Get the safety margin (minimum distance from map edge) for given altitude.
        
        The margin represents the maximum distance from camera center to any corner
        of the view rectangle when rotated. This ensures that if the agent stays at
        least this distance away from map edges, the camera view will never exceed
        map bounds regardless of rotation angle.
        
        Args:
            altitude: Agent altitude in meters.
            
        Returns:
            Safety margin in meters (Web Mercator).
            
        Note:
            Margin is linearly proportional to altitude:
            margin = altitude * margin_factor
            where margin_factor is pre-computed in __init__ based on camera parameters.
        """
        if altitude < 0:
            altitude = 0.0
        return altitude * self._margin_factor
    
    def render_image(
        self,
        sat_tif: rasterio.DatasetReader,
        position_mercator: Tuple[float, float],
        altitude: float,
        rotation: float
    ) -> np.ndarray:
        """Render RGB image from satellite TIF.
        
        This method:
        1. Calculates view bounds based on position, altitude, and rotation
        2. Crops the satellite image using the calculated bounds
        3. Rotates the image according to the rotation angle
        4. Resizes to the target dimensions
        
        Args:
            sat_tif: Open rasterio dataset (must be in EPSG:3857).
            position_mercator: Agent position as (x, y) in meters (Web Mercator).
            altitude: Agent altitude in meters.
            rotation: Camera rotation (roll angle) in degrees (0=North).
            
        Returns:
            RGB image as numpy array (H, W, 3) uint8.
            
        Raises:
            ValueError: If view bounds exceed image bounds.
        """
        # Get view bounds for rotated view (to check if within map bounds)
        left, right, bottom, top = self.get_view_bounds(
            position_mercator, altitude, rotation
        )
        
        # Get view bounds for unrotated view (for cropping)
        # We always crop an unrotated region, then rotate the image
        _left, _right, _bottom, _top = self.get_view_bounds(
            position_mercator, altitude, 0.0
        )
        
        # Check bounds (use rotated bounds to ensure we don't exceed map)
        bounds = sat_tif.bounds
        if left < bounds.left or right > bounds.right or bottom < bounds.bottom or top > bounds.top:
            raise ValueError("Camera view bounds exceed image bounds")
        
        # Expand unrotated bounds to include rotated view (for proper cropping)
        # We need a larger region to crop from, then rotate
        expanded_left = min(_left, left)
        expanded_right = max(_right, right)
        expanded_bottom = min(_bottom, bottom)
        expanded_top = max(_top, top)
        
        # Create window for cropping (expanded bounds)
        window = from_bounds(
            expanded_left, expanded_bottom, expanded_right, expanded_top,
            transform=sat_tif.transform
        )
        unrotated_window = from_bounds(
            _left, _bottom, _right, _top,
            transform=sat_tif.transform
        )
        
        # Read images
        img_expanded = sat_tif.read(window=window)  # Expanded region
        img_unrotated = sat_tif.read(window=unrotated_window)  # Unrotated view size
        
        _, h_expanded, w_expanded = img_expanded.shape
        _, h_unrotated, w_unrotated = img_unrotated.shape
        
        # Convert to (H, W, C) format and take RGB channels
        img_expanded = np.transpose(img_expanded[:3], (1, 2, 0))
        
        # Rotate image
        # scipy.ndimage.rotate: positive angle = counterclockwise, negative = clockwise
        # 
        # User expectation: When drone turns left (rotation decreases), 
        # image should rotate relatively right (clockwise from viewer's perspective).
        # 
        # Implementation: We use angle = rotation to achieve correct relative rotation:
        # - rotation decreases (left turn) → angle decreases → clockwise rotation (relative right) ✓
        # - rotation increases (right turn) → angle increases → counterclockwise rotation (relative left) ✓
        # 
        # Note: This prioritizes relative rotation correctness over absolute direction.
        # The absolute direction may differ from the rotation value (e.g., rotation=90 may not show East),
        # but the relative rotation direction matches the turn direction as expected by the user.
        img_rotated = rotate(
            img_expanded,
            angle=rotation,
            reshape=False,
            mode="constant",
            cval=0
        )
        
        # Center crop to unrotated view size
        start_y = (h_expanded - h_unrotated) // 2
        start_x = (w_expanded - w_unrotated) // 2
        cropped = img_rotated[start_y:start_y+h_unrotated, start_x:start_x+w_unrotated, :]
        
        # Resize to target dimensions
        cropped = cv2.resize(cropped, (self.width, self.height))
        
        # Ensure uint8 dtype
        cropped = np.clip(cropped, 0, 255).astype(np.uint8)
        
        return cropped


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
        # Get view bounds for rotated view
        left, right, bottom, top = self.get_view_bounds(
            position_mercator, altitude, rotation
        )
        
        # Get view bounds for unrotated view (for proper cropping)
        _left, _right, _bottom, _top = self.get_view_bounds(
            position_mercator, altitude, 0.0
        )
        
        # Check bounds
        bounds = sat_tif.bounds
        if left < bounds.left or right > bounds.right or bottom < bounds.bottom or top > bounds.top:
            raise ValueError("Camera view bounds exceed image bounds")
        
        # Expand bounds to include rotated view
        left = min(left, _left)
        right = max(right, _right)
        bottom = min(bottom, _bottom)
        top = max(top, _top)
        
        # Create windows for cropping
        window = from_bounds(left, bottom, right, top, transform=sat_tif.transform)
        _window = from_bounds(_left, _bottom, _right, _top, transform=sat_tif.transform)
        
        # Read images
        img1 = sat_tif.read(window=window)  # (bands, h1, w1)
        img2 = sat_tif.read(window=_window)  # (bands, h2, w2)
        
        _, h1, w1 = img1.shape
        _, h2, w2 = img2.shape
        
        # Convert to (H, W, C) format and take RGB channels
        img1 = np.transpose(img1[:3], (1, 2, 0))
        
        # Rotate image
        img1 = rotate(
            img1,
            angle=-rotation,
            reshape=False,
            mode="constant",
            cval=0
        )
        
        img2 = np.transpose(img2[:3], (1, 2, 0))
        
        # Center crop to unrotated view size
        start_y = (h1 - h2) // 2
        start_x = (w1 - w2) // 2
        cropped = img1[start_y:start_y+h2, start_x:start_x+w2, :]
        
        # Resize to target dimensions
        cropped = cv2.resize(cropped, (self.width, self.height))
        
        # Ensure uint8 dtype
        cropped = np.clip(cropped, 0, 255).astype(np.uint8)
        
        return cropped


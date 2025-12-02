#!/usr/bin/env python3
"""Map visualization utilities for SatNav top-down map rendering.

This module provides utilities for rendering top-down maps with satellite imagery,
including drawing agent paths, reference paths, waypoints, and agent sprites.

Adapted from VLN-CE habitat_extensions/maps.py for geographic coordinates.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import cv2
import numpy as np

# Optional imports for satellite map cropping
try:
    import rasterio
    from rasterio.windows import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

# Lazy import for GeoUtils to avoid circular imports
if TYPE_CHECKING:
    from satnav.sims.satsim.geoutils import GeoUtils


# Map element constants
MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_REFERENCE_PATH_COLOR = 8
MAP_AGENT_PATH_START = 10  # Start of gradient colors for agent path

# Color definitions (BGR format for OpenCV)
TOP_DOWN_MAP_COLORS = {
    "source": (200, 0, 0),       # Blue (BGR) - start point
    "target": (0, 0, 200),       # Red (BGR) - goal point
    "reference_path": (0, 200, 0),  # Green (BGR) - reference/shortest path
    "agent_path_start": (255, 200, 0),  # Cyan (BGR) - agent path start
    "agent_path_end": (0, 100, 255),    # Orange (BGR) - agent path end
    "waypoint": (0, 255, 255),   # Yellow (BGR)
}


def create_agent_sprite(size: int = 30) -> np.ndarray:
    """Create an arrow-shaped agent sprite programmatically.
    
    Creates a simple arrow pointing up (north) that can be rotated to indicate
    agent heading direction.
    
    Args:
        size: Size of the sprite in pixels (width and height).
        
    Returns:
        RGBA image of the agent sprite (size, size, 4) uint8.
    """
    sprite = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Arrow shape points (pointing up/north)
    center = size // 2
    top = size // 6
    bottom = size - size // 6
    wing_width = size // 3
    
    # Define arrow polygon points
    points = np.array([
        [center, top],           # Top point
        [center + wing_width, bottom],  # Bottom right
        [center, bottom - size // 4],   # Bottom center indent
        [center - wing_width, bottom],  # Bottom left
    ], dtype=np.int32)
    
    # Fill arrow with color (white with full alpha)
    cv2.fillPoly(sprite, [points], (255, 255, 255, 255))
    
    # Add border for visibility
    cv2.polylines(sprite, [points], True, (0, 0, 0, 255), thickness=2)
    
    return sprite


# Global agent sprite (cached)
_AGENT_SPRITE: Optional[np.ndarray] = None


def get_agent_sprite(size: int = 30) -> np.ndarray:
    """Get the agent sprite, creating it if necessary.
    
    Args:
        size: Size of the sprite in pixels.
        
    Returns:
        RGBA image of the agent sprite.
    """
    global _AGENT_SPRITE
    if _AGENT_SPRITE is None or _AGENT_SPRITE.shape[0] != size:
        _AGENT_SPRITE = create_agent_sprite(size)
    return _AGENT_SPRITE.copy()


def geo_to_pixel(
    lon: float,
    lat: float,
    bounds: Dict[str, float],
    map_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """Convert geographic coordinates to pixel coordinates.
    
    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        bounds: Dictionary with 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
        map_shape: (height, width) of the map image.
        
    Returns:
        (row, col) pixel coordinates. Note: row is y-axis (vertical).
    """
    height, width = map_shape[:2]
    
    # Calculate normalized position [0, 1]
    lon_norm = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"])
    lat_norm = (lat - bounds["lat_min"]) / (bounds["lat_max"] - bounds["lat_min"])
    
    # Convert to pixel coordinates
    # Note: latitude increases upward but pixel row increases downward
    col = int(lon_norm * (width - 1))
    row = int((1 - lat_norm) * (height - 1))  # Flip for image coordinates
    
    # Clamp to valid range
    row = max(0, min(height - 1, row))
    col = max(0, min(width - 1, col))
    
    return (row, col)


def crop_satellite_map(
    sat_tif,  # rasterio.DatasetReader
    reference_path: List[List[float]],
    goal_position: List[float],
    start_position: List[float],
    padding_meters: float = 50.0,
    max_resolution: int = 1024
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Crop satellite map to include all reference points with padding.
    
    This function calculates the bounding box that contains all reference path
    points and the goal, then adds padding and crops the satellite map.
    
    Args:
        sat_tif: Open rasterio dataset (must be in EPSG:3857).
        reference_path: List of waypoints as [[lon, lat, alt], ...].
        goal_position: Goal position as [lon, lat, alt].
        start_position: Start position as [lon, lat, alt].
        padding_meters: Padding to add around the bounding box in meters.
        max_resolution: Maximum resolution of the output image (longest side).
        
    Returns:
        Tuple of:
            - Cropped satellite image as numpy array (H, W, 3) uint8.
            - Bounds dictionary with 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
            
    Raises:
        ImportError: If rasterio is not installed.
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for crop_satellite_map. "
            "Install it with: pip install rasterio"
        )
    
    # Import GeoUtils here to avoid circular imports
    from satnav.sims.satsim.geoutils import GeoUtils
    from rasterio.windows import from_bounds
    
    # Collect all points to include
    all_points = []
    
    if reference_path:
        all_points.extend(reference_path)
    
    all_points.append(goal_position)
    all_points.append(start_position)
    
    # Calculate bounding box in WGS84
    lons = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    # Convert to Mercator for padding calculation
    x_min, y_min = GeoUtils.wgs84_to_mercator(lon_min, lat_min)
    x_max, y_max = GeoUtils.wgs84_to_mercator(lon_max, lat_max)
    
    # Add padding in Mercator space (meters)
    x_min -= padding_meters
    x_max += padding_meters
    y_min -= padding_meters
    y_max += padding_meters
    
    # Clamp to scene bounds
    scene_bounds = sat_tif.bounds
    x_min = max(x_min, scene_bounds.left)
    x_max = min(x_max, scene_bounds.right)
    y_min = max(y_min, scene_bounds.bottom)
    y_max = min(y_max, scene_bounds.top)
    
    # Create window for cropping
    window = from_bounds(x_min, y_min, x_max, y_max, transform=sat_tif.transform)
    
    # Read the cropped region
    img = sat_tif.read(window=window)
    
    # Convert to (H, W, C) format and take RGB channels
    if img.shape[0] >= 3:
        img = np.transpose(img[:3], (1, 2, 0))
    else:
        # Handle single-channel or 2-channel images
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
    
    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Ensure uint8 and C-contiguous for OpenCV compatibility
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    
    # Convert bounds back to WGS84
    lon_min_final, lat_min_final = GeoUtils.mercator_to_wgs84(x_min, y_min)
    lon_max_final, lat_max_final = GeoUtils.mercator_to_wgs84(x_max, y_max)
    
    bounds_wgs84 = {
        "lon_min": lon_min_final,
        "lon_max": lon_max_final,
        "lat_min": lat_min_final,
        "lat_max": lat_max_final,
    }
    
    return img, bounds_wgs84


def draw_point(
    img: np.ndarray,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 5,
    thickness: int = -1
) -> None:
    """Draw a point (filled circle) on the image.
    
    Args:
        img: Image to draw on (modified in place).
        position: (row, col) pixel coordinates.
        color: BGR color tuple.
        radius: Circle radius in pixels.
        thickness: Circle thickness (-1 for filled).
    """
    # OpenCV uses (x, y) = (col, row)
    center = (position[1], position[0])
    cv2.circle(img, center, radius, color, thickness)


def draw_circle_outline(
    img: np.ndarray,
    position: List[float],
    radius_meters: float,
    bounds: Dict[str, float],
    color: Tuple[int, int, int] = (0, 255, 0),  # Green by default
    thickness: int = 2
) -> None:
    """Draw a circle outline (hollow circle) around a geographic position.
    
    Args:
        img: Image to draw on (modified in place).
        position: Geographic position as [lon, lat, alt].
        radius_meters: Radius of the circle in meters.
        bounds: Geographic bounds dictionary.
        color: BGR color tuple for the circle outline.
        thickness: Line thickness for the circle outline.
    """
    map_shape = img.shape[:2]
    
    # Get pixel position
    row, col = geo_to_pixel(position[0], position[1], bounds, map_shape)
    
    # Calculate radius in pixels
    # Approximate: 1 degree latitude ≈ 111,000 meters
    # Calculate meters per pixel
    lon_range = bounds["lon_max"] - bounds["lon_min"]
    lat_range = bounds["lat_max"] - bounds["lat_min"]
    
    # Average meters per degree (approximate)
    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * np.cos(np.radians((bounds["lat_min"] + bounds["lat_max"]) / 2))
    
    # Calculate pixel size
    pixels_per_degree_lon = map_shape[1] / lon_range
    pixels_per_degree_lat = map_shape[0] / lat_range
    
    # Convert radius to pixels (use average)
    radius_px_lon = radius_meters / meters_per_degree_lon * pixels_per_degree_lon
    radius_px_lat = radius_meters / meters_per_degree_lat * pixels_per_degree_lat
    radius_px = int((radius_px_lon + radius_px_lat) / 2)
    
    # Draw circle outline (hollow circle)
    # OpenCV uses (x, y) = (col, row)
    center = (col, row)
    cv2.circle(img, center, radius_px, color, thickness)


def draw_path(
    img: np.ndarray,
    path_points: List[Tuple[int, int]],
    color: Union[Tuple[int, int, int], str] = "gradient",
    thickness: int = 2,
    max_steps: int = 500
) -> None:
    """Draw a path on the image.
    
    Args:
        img: Image to draw on (modified in place).
        path_points: List of (row, col) pixel coordinates.
        color: BGR color tuple or "gradient" for gradient coloring.
        thickness: Line thickness.
        max_steps: Maximum steps for gradient calculation.
    """
    if len(path_points) < 2:
        return
    
    for i in range(len(path_points) - 1):
        pt1 = (path_points[i][1], path_points[i][0])      # (col, row) -> (x, y)
        pt2 = (path_points[i + 1][1], path_points[i + 1][0])
        
        if color == "gradient":
            # Calculate gradient color based on progress
            progress = min(i / max_steps, 1.0)
            start_color = np.array(TOP_DOWN_MAP_COLORS["agent_path_start"])
            end_color = np.array(TOP_DOWN_MAP_COLORS["agent_path_end"])
            current_color = tuple(
                int(start_color[j] * (1 - progress) + end_color[j] * progress)
                for j in range(3)
            )
        else:
            current_color = color
        
        cv2.line(img, pt1, pt2, current_color, thickness)


def draw_reference_path(
    img: np.ndarray,
    reference_path: List[List[float]],
    bounds: Dict[str, float],
    color: Tuple[int, int, int] = None,
    thickness: int = 2,
    draw_points: bool = True,
    point_radius: int = 4
) -> None:
    """Draw reference path on the image.
    
    Args:
        img: Image to draw on (modified in place).
        reference_path: List of waypoints as [[lon, lat, alt], ...].
        bounds: Geographic bounds dictionary.
        color: BGR color tuple (default: green).
        thickness: Line thickness.
        draw_points: Whether to draw points at each waypoint.
        point_radius: Radius of waypoint markers.
    """
    if not reference_path or len(reference_path) < 2:
        return
    
    if color is None:
        color = TOP_DOWN_MAP_COLORS["reference_path"]
    
    map_shape = img.shape[:2]
    
    # Convert to pixel coordinates
    pixel_points = []
    for point in reference_path:
        row, col = geo_to_pixel(point[0], point[1], bounds, map_shape)
        pixel_points.append((row, col))
    
    # Draw lines
    for i in range(len(pixel_points) - 1):
        pt1 = (pixel_points[i][1], pixel_points[i][0])
        pt2 = (pixel_points[i + 1][1], pixel_points[i + 1][0])
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw points at each waypoint
    if draw_points:
        for row, col in pixel_points:
            draw_point(img, (row, col), color, radius=point_radius)


def draw_source_and_target(
    img: np.ndarray,
    start_position: List[float],
    goal_position: List[float],
    bounds: Dict[str, float],
    source_color: Tuple[int, int, int] = None,
    target_color: Tuple[int, int, int] = None,
    radius: int = 8
) -> None:
    """Draw source (start) and target (goal) markers on the image.
    
    Args:
        img: Image to draw on (modified in place).
        start_position: Start position as [lon, lat, alt].
        goal_position: Goal position as [lon, lat, alt].
        bounds: Geographic bounds dictionary.
        source_color: BGR color for source marker (default: blue).
        target_color: BGR color for target marker (default: red).
        radius: Marker radius in pixels.
    """
    if source_color is None:
        source_color = TOP_DOWN_MAP_COLORS["source"]
    if target_color is None:
        target_color = TOP_DOWN_MAP_COLORS["target"]
    
    map_shape = img.shape[:2]
    
    # Draw source
    row, col = geo_to_pixel(start_position[0], start_position[1], bounds, map_shape)
    draw_point(img, (row, col), source_color, radius=radius)
    
    # Draw target
    row, col = geo_to_pixel(goal_position[0], goal_position[1], bounds, map_shape)
    draw_point(img, (row, col), target_color, radius=radius)


def draw_agent(
    img: np.ndarray,
    position: List[float],
    rotation: float,
    bounds: Dict[str, float],
    sprite_size: int = 30
) -> np.ndarray:
    """Draw the agent with heading arrow on the image.
    
    Args:
        img: Image to draw on (will be copied).
        position: Agent position as [lon, lat, alt].
        rotation: Agent heading in degrees (0 = North).
        bounds: Geographic bounds dictionary.
        sprite_size: Size of the agent sprite in pixels.
        
    Returns:
        Modified image with agent drawn.
    """
    img = img.copy()
    map_shape = img.shape[:2]
    
    # Get pixel position
    row, col = geo_to_pixel(position[0], position[1], bounds, map_shape)
    
    # Get and rotate agent sprite
    sprite = get_agent_sprite(sprite_size)
    
    # Rotate sprite (OpenCV rotation is counter-clockwise, we need clockwise for heading)
    center = (sprite_size // 2, sprite_size // 2)
    M = cv2.getRotationMatrix2D(center, -rotation, 1.0)  # Negative for clockwise
    rotated_sprite = cv2.warpAffine(
        sprite, M, (sprite_size, sprite_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    # Calculate paste position (centered on agent position)
    half_size = sprite_size // 2
    y_start = row - half_size
    y_end = row + half_size + (sprite_size % 2)
    x_start = col - half_size
    x_end = col + half_size + (sprite_size % 2)
    
    # Handle boundary conditions
    sprite_y_start = max(0, -y_start)
    sprite_y_end = sprite_size - max(0, y_end - map_shape[0])
    sprite_x_start = max(0, -x_start)
    sprite_x_end = sprite_size - max(0, x_end - map_shape[1])
    
    y_start = max(0, y_start)
    y_end = min(map_shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(map_shape[1], x_end)
    
    # Get sprite region
    sprite_region = rotated_sprite[
        sprite_y_start:sprite_y_end,
        sprite_x_start:sprite_x_end
    ]
    
    if sprite_region.shape[0] == 0 or sprite_region.shape[1] == 0:
        return img
    
    # Alpha blending
    if sprite_region.shape[2] == 4:
        alpha = sprite_region[:, :, 3:4] / 255.0
        sprite_rgb = sprite_region[:, :, :3]
        
        img_region = img[y_start:y_end, x_start:x_end]
        
        # Ensure shapes match
        if img_region.shape[:2] == sprite_rgb.shape[:2]:
            img[y_start:y_end, x_start:x_end] = (
                alpha * sprite_rgb + (1 - alpha) * img_region
            ).astype(np.uint8)
    
    return img


def colorize_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5
) -> np.ndarray:
    """Apply fog of war effect to the map.
    
    Args:
        top_down_map: RGB map image.
        fog_of_war_mask: Binary mask where 1 = visible, 0 = fog.
        fog_of_war_desat_amount: Amount to desaturate fogged areas.
        
    Returns:
        Map with fog of war applied.
    """
    if fog_of_war_mask is None:
        return top_down_map.copy()
    
    _map = top_down_map.copy().astype(np.float32)
    
    # Desaturate areas not in fog of war mask
    fog_indices = fog_of_war_mask == 0
    _map[fog_indices] = _map[fog_indices] * fog_of_war_desat_amount
    
    return _map.astype(np.uint8)


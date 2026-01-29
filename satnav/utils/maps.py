#!/usr/bin/env python3
"""Map visualization utilities for SatNav top-down map rendering.

This module provides utilities for rendering top-down maps with satellite imagery,
including drawing agent paths, reference paths, waypoints, and agent sprites.

Adapted from VLN-CE habitat_extensions/maps.py for geographic coordinates.
"""

import math
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import cv2
import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    imageio = None

try:
    import scipy.ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None

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


def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
) -> None:
    """Composites the foreground onto the background dealing with edge boundaries.
    
    Adapted from habitat-lab/habitat/utils/visualizations/utils.py
    
    Args:
        background: The background image to paste on (modified in place).
        foreground: The image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: The image coordinates to paste the foreground (row, col).
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground


# Global agent sprite (loaded from file, cached)
_AGENT_SPRITE: Optional[np.ndarray] = None


def get_agent_sprite() -> np.ndarray:
    """Get the agent sprite from VLN-CE/habitat-lab asset file.
    
    Returns:
        RGBA image of the agent sprite (100x100, 4 channels).
    """
    global _AGENT_SPRITE
    if _AGENT_SPRITE is None:
        if not HAS_IMAGEIO:
            raise ImportError("imageio is required to load agent sprite")
        
        sprite_path = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "maps_topdown_agent_sprite",
            "100x100.png",
        )
        
        if not os.path.exists(sprite_path):
            raise FileNotFoundError(
                f"Agent sprite not found at {sprite_path}. "
                "Please ensure the sprite file is copied from habitat-lab."
            )
        
        _AGENT_SPRITE = imageio.imread(sprite_path)
        # Flip vertically (as done in habitat-lab)
        _AGENT_SPRITE = np.ascontiguousarray(np.flipud(_AGENT_SPRITE))
    
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
    
    # Add padding in Mercator space
    # padding_meters is in TRUE ground meters, convert to Mercator meters
    # Use center Y coordinate for scale factor calculation
    center_y = (y_min + y_max) / 2
    padding_mercator = GeoUtils.true_meters_to_mercator(padding_meters, center_y)
    x_min -= padding_mercator
    x_max += padding_mercator
    y_min -= padding_mercator
    y_max += padding_mercator
    
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
    thickness: int = 4,  # Increased from 2 to 4 for better visibility
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
    thickness: int = 4,  # Increased from 2 to 4 for better visibility
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


def draw_camera_view_bounds(
    img: np.ndarray,
    agent_position: List[float],
    agent_rotation: float,
    altitude: float,
    hfov: float,
    image_width: int,
    image_height: int,
    bounds: Dict[str, float],
    color: Tuple[int, int, int] = (0, 255, 255),  # Yellow (BGR)
    thickness: int = 2
) -> None:
    """Draw RGB camera view bounds on the top-down map.
    
    Calculates the ground coverage area visible to the RGB camera and draws
    it as a yellow-bordered rectangle on the map.
    
    Args:
        img: Image to draw on (modified in place).
        agent_position: Agent position as [lon, lat, alt].
        agent_rotation: Agent rotation in degrees (0=North).
        altitude: Camera altitude in meters.
        hfov: Horizontal field of view in degrees.
        image_width: RGB image width in pixels.
        image_height: RGB image height in pixels.
        bounds: Geographic bounds dictionary.
        color: BGR color for the rectangle border (default: yellow).
        thickness: Border thickness in pixels.
    """
    # Import here to avoid circular imports
    from satnav.sims.satsim.geoutils import GeoUtils
    from satnav.sims.satsim.camera import SatelliteCamera
    
    # Create camera instance to calculate view bounds
    camera = SatelliteCamera(image_width, image_height, hfov)
    
    # Convert agent position to Mercator
    position_mercator = GeoUtils.wgs84_to_mercator(
        agent_position[0], agent_position[1]
    )
    
    # Calculate view bounds corners directly (similar to camera.get_view_bounds logic)
    cx, cy = position_mercator
    h = altitude
    
    # Get Mercator scale factor at current latitude
    scale_factor = GeoUtils.get_mercator_scale_factor(cy)
    
    # Ground half-height/width in TRUE ground meters
    half_x_true = h * math.tan(math.radians(hfov / 2.0))
    half_y_true = half_x_true / (image_width / image_height)  # aspect ratio
    
    # Convert to Mercator meters (for correct visualization)
    half_x_m = half_x_true * scale_factor
    half_y_m = half_y_true * scale_factor
    
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
    az_deg = [((az + agent_rotation) % 360.0) for az in base_az_deg]
    
    # Calculate corner positions in Mercator
    corners_mercator = [
        (
            cx + r * math.sin(math.radians(az)),
            cy + r * math.cos(math.radians(az))
        )
        for az in az_deg
    ]
    
    # Convert corners from Mercator to WGS84
    corners_wgs84 = [
        GeoUtils.mercator_to_wgs84(x, y) for x, y in corners_mercator
    ]
    
    # Convert to pixel coordinates (allow out-of-bounds coordinates)
    map_shape = img.shape[:2]
    pixel_corners = []
    for lon, lat in corners_wgs84:
        # Calculate pixel coordinates without clamping (allow negative or > image size)
        height, width = map_shape[:2]
        lon_norm = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"])
        lat_norm = (lat - bounds["lat_min"]) / (bounds["lat_max"] - bounds["lat_min"])
        col = lon_norm * (width - 1)
        row = (1 - lat_norm) * (height - 1)  # Flip for image coordinates
        pixel_corners.append((int(col), int(row)))  # OpenCV uses (x, y) = (col, row)
    
    # Draw rectangle using cv2.polylines
    # Even if corners are outside the image, cv2.polylines will draw the visible parts
    if len(pixel_corners) >= 4:
        # Close the polygon by adding first point at the end
        closed_corners = np.array(pixel_corners + [pixel_corners[0]], dtype=np.int32)
        cv2.polylines(img, [closed_corners], isClosed=True, color=color, thickness=thickness)


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
    agent_radius_px: int = 15
) -> np.ndarray:
    """Draw the agent with heading arrow on the image.
    
    Uses the same agent sprite and drawing method as VLN-CE/habitat-lab.
    
    Args:
        img: Image to draw on (will be copied).
        position: Agent position as [lon, lat, alt].
        rotation: Agent heading in degrees (0 = North).
        bounds: Geographic bounds dictionary.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
        
    Returns:
        Modified image with agent drawn.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for agent sprite rotation")
    
    img = img.copy()
    map_shape = img.shape[:2]
    
    # Get pixel position
    row, col = geo_to_pixel(position[0], position[1], bounds, map_shape)
    agent_center_coord = (row, col)
    
    # Get agent sprite (100x100 from file)
    agent_sprite = get_agent_sprite()
    
    # Coordinate system conversion:
    # - SatNav: rotation in degrees, 0 = North (up), 90 = East (right), 180 = South (down), 270 = West (left)
    # - Habitat-lab sprite: after flipud, initial direction needs to be determined
    # - scipy.ndimage.rotate: rotates counter-clockwise (positive angle = CCW)
    # 
    # Try different formulas based on sprite initial direction:
    # If sprite initially points East (right) when rotation=0:
    #   - To point North (up) when rotation=0, need -90° rotation
    #   - Formula: adjusted_rotation = rotation - 90.0
    # 
    # If sprite initially points West (left) when rotation=0:
    #   - To point North (up) when rotation=0, need +90° rotation  
    #   - Formula: adjusted_rotation = rotation + 90.0
    # 
    # If sprite initially points South (down) when rotation=0:
    #   - To point North (up) when rotation=0, need +180° rotation
    #   - Formula: adjusted_rotation = rotation + 180.0
    # 
    # Current test: try rotation - 90 (assuming sprite initially points East)
    adjusted_rotation_deg = -rotation + 180.0
    
    # Rotate before resize to keep good resolution (as in habitat-lab)
    rotated_agent = scipy.ndimage.rotate(
        agent_sprite, adjusted_rotation_deg, reshape=False
    )
    
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = agent_sprite.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    
    # Paste using the same method as habitat-lab
    paste_overlapping_image(img, resized_agent, agent_center_coord)
    
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


def add_instruction_text(
    img: np.ndarray,
    text: str,
    font_size: float = 1.1,
    thickness: int = 2,
    padding: int = 15,
    line_spacing: int = 25
) -> Tuple[np.ndarray, int]:
    """Add instruction text to an image with word wrapping.
    
    Adapted from VLN-CE habitat_extensions/utils.py add_instruction_on_img.
    
    Args:
        img: Image to add text to (BGR format).
        text: Instruction text to add.
        font_size: Font size for text.
        thickness: Text thickness.
        padding: Padding from edges.
        line_spacing: Spacing between lines in pixels.
        
    Returns:
        Tuple of (image with instruction text added, required height in pixels).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate character size for wrapping
    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(
        text, width=int((img.shape[1] - 2 * padding) / char_size[0])
    )
    
    # Calculate required height for all text lines
    total_height = padding  # Start with top padding
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        total_height += textsize[1] + line_spacing
    total_height += padding  # Add bottom padding
    
    # Resize image if needed to fit all text
    if total_height > img.shape[0]:
        # Create new image with required height
        new_img = np.ones((total_height, img.shape[1], 3), dtype=np.uint8) * 255
        img = new_img
    
    # Draw each line of text
    y = padding
    start_x = padding
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + line_spacing
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),  # Black text (BGR)
            thickness,
            lineType=cv2.LINE_AA,
        )
    
    return img, total_height


def create_video_frame(
    rgb_image: np.ndarray,
    topdown_image: np.ndarray,
    instruction_text: str,
    frame_width: int = 2048,
    min_frame_width: int = 2048,
    max_frame_width: int = 4096
) -> Tuple[np.ndarray, int]:
    """Create a single video frame combining RGB, top-down map, and instruction.
    
    Layout:
    - Top row: RGB image (left) | Top-down map (right)
    - Bottom row: Instruction text (full width)
    
    The frame width will be dynamically adjusted based on instruction text length
    to ensure all text is displayed without truncation.
    
    Args:
        rgb_image: RGB observation image (H, W, 3) in RGB format.
        topdown_image: Top-down map image (H, W, 3) in RGB format.
        instruction_text: Instruction text to display.
        frame_width: Initial target width for the frame.
        min_frame_width: Minimum frame width (default: 2048).
        max_frame_width: Maximum frame width (default: 4096).
        
    Returns:
        Tuple of (combined frame image (H, W, 3) in RGB format, actual frame width used).
    """
    # Convert RGB to BGR for OpenCV operations
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    topdown_bgr = cv2.cvtColor(topdown_image, cv2.COLOR_RGB2BGR)
    
    # Resize RGB to target width, maintaining aspect ratio
    rgb_h, rgb_w = rgb_bgr.shape[:2]
    rgb_new_h = int((frame_width / 2 / rgb_w) * rgb_h)
    rgb_resized = cv2.resize(
        rgb_bgr,
        (frame_width // 2, rgb_new_h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Resize top-down map maintaining aspect ratio, fit within target area
    topdown_h, topdown_w = topdown_bgr.shape[:2]
    target_topdown_width = frame_width // 2
    target_topdown_height = rgb_new_h
    
    # Calculate scaling factor to fit within target area while maintaining aspect ratio
    scale_w = target_topdown_width / topdown_w
    scale_h = target_topdown_height / topdown_h
    scale = min(scale_w, scale_h)  # Use smaller scale to ensure it fits
    
    # Calculate new dimensions maintaining aspect ratio
    topdown_new_w = int(topdown_w * scale)
    topdown_new_h = int(topdown_h * scale)
    
    # Resize top-down map with aspect ratio preserved
    topdown_resized = cv2.resize(
        topdown_bgr,
        (topdown_new_w, topdown_new_h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Calculate padding needed (centered)
    pad_width = target_topdown_width - topdown_new_w
    pad_height = target_topdown_height - topdown_new_h
    
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    
    # Add white padding around top-down map (centered)
    topdown_with_space = cv2.copyMakeBorder(
        topdown_resized,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)  # White padding (BGR)
    )
    
    # Ensure top_row width matches frame_width exactly
    top_row_width = rgb_resized.shape[1] + topdown_with_space.shape[1]
    if top_row_width != frame_width:
        # Adjust if needed (shouldn't happen, but safety check)
        if top_row_width > frame_width:
            # Crop if too wide
            top_row = np.concatenate([rgb_resized, topdown_with_space], axis=1)
            top_row = top_row[:, :frame_width, :]
        else:
            # Pad if too narrow
            padding_width = frame_width - top_row_width
            padding = np.ones((rgb_new_h, padding_width, 3), dtype=np.uint8) * 255
            top_row = np.concatenate([rgb_resized, topdown_with_space, padding], axis=1)
    else:
        # Combine RGB and top-down map horizontally
        top_row = np.concatenate([rgb_resized, topdown_with_space], axis=1)
    
    # Calculate required width for instruction text based on text length
    # Estimate text width: average character width * number of characters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.1
    thickness = 2
    padding = 15
    
    # Calculate average character width
    char_size = cv2.getTextSize("M", font, font_size, thickness)[0]
    avg_char_width = char_size[0]
    
    # Estimate text width needed (add padding and some margin)
    # Use textwrap to get actual wrapped lines to calculate width more accurately
    temp_panel = np.ones((200, frame_width, 3), dtype=np.uint8) * 255
    wrapped_text = textwrap.wrap(
        instruction_text, width=int((frame_width - 2 * padding) / avg_char_width)
    )
    
    # Find the longest line to determine required width
    max_line_width = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        max_line_width = max(max_line_width, textsize[0])
    
    # Calculate required width: longest line + padding + margin
    required_text_width = max_line_width + 2 * padding + 100
    
    # Determine actual frame width needed
    # Ensure it's at least min_frame_width, but expand if text needs more space
    actual_frame_width = max(frame_width, min(required_text_width, max_frame_width))
    
    # If we need to expand width, resize top_row accordingly
    if actual_frame_width > frame_width:
        # Resize top_row to match new width
        scale_factor = actual_frame_width / frame_width
        new_rgb_width = int(frame_width // 2 * scale_factor)
        new_topdown_width = actual_frame_width - new_rgb_width
        
        # Resize RGB and topdown to new widths, maintaining aspect ratio
        rgb_resized = cv2.resize(
            rgb_bgr,
            (new_rgb_width, int(rgb_new_h * scale_factor)),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Resize topdown maintaining aspect ratio
        topdown_h, topdown_w = topdown_bgr.shape[:2]
        topdown_scale_w = new_topdown_width / topdown_w
        topdown_scale_h = (rgb_resized.shape[0]) / topdown_h
        topdown_scale = min(topdown_scale_w, topdown_scale_h)
        
        topdown_new_w = int(topdown_w * topdown_scale)
        topdown_new_h = int(topdown_h * topdown_scale)
        
        topdown_resized = cv2.resize(
            topdown_bgr,
            (topdown_new_w, topdown_new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Recalculate padding for topdown
        pad_width = new_topdown_width - topdown_new_w
        pad_height = rgb_resized.shape[0] - topdown_new_h
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        topdown_with_space = cv2.copyMakeBorder(
            topdown_resized,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        
        top_row = np.concatenate([rgb_resized, topdown_with_space], axis=1)
        frame_width = actual_frame_width
    
    # Create instruction panel with dynamic height based on text length
    # Start with a reasonable initial height, will be adjusted based on text length
    initial_instruction_height = 200
    instruction_panel = np.ones((initial_instruction_height, frame_width, 3), dtype=np.uint8) * 255
    
    # Add instruction text (function will resize panel if needed to fit all text)
    instruction_panel, required_height = add_instruction_text(instruction_panel, instruction_text)
    
    # Combine top row and instruction panel vertically
    frame = np.concatenate([top_row, instruction_panel], axis=0)
    
    # Convert back to RGB for video output
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb, frame_width


def make_video(
    rgb_frames: List[np.ndarray],
    topdown_frames: List[np.ndarray],
    instruction_text: str,
    output_path: Union[str, Path],
    fps: int = 10,
    frame_width: int = 2048,
    quality: Optional[float] = 5
) -> None:
    """Generate a video from RGB and top-down map frames with instruction text.
    
    Creates a video where each frame shows:
    - Left side: RGB sensor observation
    - Right side: Top-down map visualization
    - Bottom: Instruction text
    
    Args:
        rgb_frames: List of RGB observation images (H, W, 3) in RGB format.
        topdown_frames: List of top-down map images (H, W, 3) in RGB format.
        instruction_text: Instruction text to display on all frames.
        output_path: Path to save the video file.
        fps: Frames per second for the video.
        frame_width: Target width for video frames.
        quality: Video quality (0-10, higher is better). Uses variable bitrate.
        
    Raises:
        ImportError: If imageio is not installed.
        ValueError: If rgb_frames and topdown_frames have different lengths.
    """
    if not HAS_IMAGEIO:
        raise ImportError(
            "imageio is required for video generation. "
            "Install it with: pip install imageio imageio-ffmpeg"
        )
    
    if len(rgb_frames) != len(topdown_frames):
        raise ValueError(
            f"rgb_frames and topdown_frames must have the same length. "
            f"Got {len(rgb_frames)} and {len(topdown_frames)}"
        )
    
    if len(rgb_frames) == 0:
        raise ValueError("No frames provided for video generation")
    
    if quality is not None:
        assert 0 <= quality <= 10, "quality must be between 0 and 10"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate video frames
    # First pass: determine the maximum width needed across all frames
    max_width = frame_width
    for rgb_frame, topdown_frame in zip(rgb_frames, topdown_frames):
        _, actual_width = create_video_frame(
            rgb_frame,
            topdown_frame,
            instruction_text,
            frame_width,
            min_frame_width=frame_width,
            max_frame_width=8192  # Allow up to 8K width for very long instructions
        )
        max_width = max(max_width, actual_width)
    
    # Second pass: create all frames with consistent width
    video_frames = []
    for rgb_frame, topdown_frame in zip(rgb_frames, topdown_frames):
        frame, _ = create_video_frame(
            rgb_frame,
            topdown_frame,
            instruction_text,
            max_width,  # Use the maximum width for all frames
            min_frame_width=max_width,
            max_frame_width=max_width
        )
        video_frames.append(frame)
    
    # Write video using imageio
    video_name = output_path.name
    if not video_name.endswith('.mp4'):
        video_name = f"{video_name}.mp4"
    
    writer = imageio.get_writer(
        str(output_path.parent / video_name),
        fps=fps,
        quality=quality,
    )
    
    for frame in video_frames:
        writer.append_data(frame)
    
    writer.close()
    
    print(f"✓ Video saved to: {output_path.parent / video_name}")


def annotate_topdown_map(
    info: dict,
    agent_state,
    waypoints: list,
    current_waypoint_idx: int,
    step_count: int,
    action: str,
    current_distance: float,
    goal_radius: float,
    config,
    topdown_frames: list = None
) -> np.ndarray:
    """Annotate top-down map with camera view bounds, waypoint circles, and text overlay.
    
    This function takes the top-down map from environment metrics and adds:
    - Camera view bounds (RGB sensor coverage area) as yellow rectangle
    - Waypoint threshold circles (yellow for current, gray for reached, green for future)
    - Text overlay with step information
    
    Args:
        info: Environment info dict containing metrics.
        agent_state: Current agent state with position and rotation.
        waypoints: List of waypoint positions [[lon, lat, alt], ...].
        current_waypoint_idx: Index of current waypoint being navigated to (0-indexed).
        step_count: Current step number.
        action: Action taken in this step.
        current_distance: Distance to current waypoint in meters.
        goal_radius: Goal radius threshold in meters.
        config: Configuration object with SIMULATOR.RGB_SENSOR settings.
        topdown_frames: Optional list to append RGB frame for video generation.
        
    Returns:
        Annotated top-down map as RGB numpy array (H, W, 3), or None if top_down_map
        is not available in metrics.
    """
    try:
        # Use metrics from info dict (already updated after step)
        env_metrics = info.get("metrics", {})
        if "top_down_map" not in env_metrics:
            return None
        
        top_down_info = env_metrics["top_down_map"]
        map_image = top_down_info["map"]
        bounds = top_down_info["bounds"]
        
        # Convert RGB to BGR for OpenCV
        map_bgr = cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR)
        
        # Draw camera view bounds (RGB sensor coverage area)
        draw_camera_view_bounds(
            map_bgr,
            agent_state.position.tolist(),
            agent_state.rotation,
            agent_state.position[2],  # altitude
            config.SIMULATOR.RGB_SENSOR.HFOV,
            config.SIMULATOR.RGB_SENSOR.WIDTH,
            config.SIMULATOR.RGB_SENSOR.HEIGHT,
            bounds,
            color=(0, 255, 255),  # Yellow (BGR)
            thickness=2
        )
        
        # Draw waypoint threshold circles
        for i, wp in enumerate(waypoints):
            # Use different colors for current waypoint vs others
            if i == current_waypoint_idx:
                # Current waypoint: yellow circle
                circle_color = (0, 255, 255)  # Yellow (BGR)
            elif i < current_waypoint_idx:
                # Reached waypoints: gray circle
                circle_color = (128, 128, 128)  # Gray (BGR)
            else:
                # Future waypoints: green circle
                circle_color = (0, 255, 0)  # Green (BGR)
            
            # Draw threshold circle for each waypoint
            draw_circle_outline(
                map_bgr,
                wp,
                radius_meters=goal_radius,
                bounds=bounds,
                color=circle_color,
                thickness=2
            )
        
        # Add text overlay with step info
        cv2.putText(map_bgr, f"Step {step_count}: {action}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(map_bgr, f"Waypoint {current_waypoint_idx+1}/{len(waypoints)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(map_bgr, f"Dist: {current_distance:.1f}m", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(map_bgr, f"Threshold: {goal_radius:.1f}m", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert BGR back to RGB for return
        map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
        
        # Store top-down frame for video if requested
        if topdown_frames is not None:
            topdown_frames.append(map_rgb.copy())
        
        return map_rgb
    except Exception:
        # Silently ignore errors
        return None


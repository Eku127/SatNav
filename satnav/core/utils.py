#!/usr/bin/env python3
from __future__ import annotations

"""Utility functions for SatNav."""

import math
from typing import Sequence

# Earth's radius in meters (WGS84 ellipsoid mean radius)
EARTH_RADIUS_METERS = 6371000.0


def geodesic_distance(
    position_a: Sequence[float],
    position_b: Sequence[float]
) -> float:
    """Calculate geodesic distance between two positions using Haversine formula.
    
    This function calculates the great-circle distance between two points on Earth's
    surface using their longitude and latitude coordinates. The altitude component
    is currently ignored (distance is calculated on Earth's surface).
    
    The Haversine formula accounts for Earth's curvature and provides accurate
    distance calculations for navigation purposes.
    
    Args:
        position_a: First position as [longitude, latitude, altitude] or [longitude, latitude].
        position_b: Second position as [longitude, latitude, altitude] or [longitude, latitude].
        
    Returns:
        Distance in meters between the two positions.
        
    Example:
        >>> # Distance between Beijing (116.3974°E, 39.9093°N) and Shanghai (121.4737°E, 31.2304°N)
        >>> pos_a = [116.3974, 39.9093, 0]
        >>> pos_b = [121.4737, 31.2304, 0]
        >>> distance = geodesic_distance(pos_a, pos_b)
        >>> print(f"Distance: {distance:.2f} meters")
        Distance: 1067000.00 meters  # Approximately 1067 km
    """
    lon1 = math.radians(float(position_a[0]))
    lat1 = math.radians(float(position_a[1]))
    lon2 = math.radians(float(position_b[0]))
    lat2 = math.radians(float(position_b[1]))
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance in meters
    distance = EARTH_RADIUS_METERS * c
    
    return float(distance)


def lonlat_to_ego_displacement(
    start_position: Sequence[float],
    start_heading_deg: float,
    current_position: Sequence[float]
) -> tuple:
    """Convert geographic displacement into ego-frame displacement (forward, right).
    
    The ego-frame is defined by the agent's initial heading at episode start:
    - Forward axis: the direction the agent was facing at t=0
    - Right axis: 90 degrees clockwise from forward
    
    This avoids any dependency on cardinal directions (N/S/E/W), which the
    model cannot observe from BEV images.
    
    Uses Mercator projection (consistent with SatSim's internal representation)
    for accurate displacement calculation, then rotates into the ego-frame.
    
    Args:
        start_position: Start position as [longitude, latitude, altitude].
        start_heading_deg: Start heading in degrees (0=North, clockwise).
        current_position: Current position as [longitude, latitude, altitude].
        
    Returns:
        Tuple of (delta_forward_m, delta_right_m):
            - delta_forward_m: Displacement along the initial forward direction (meters).
                Positive = moved in the direction the agent was initially facing.
            - delta_right_m: Displacement perpendicular to forward (meters).
                Positive = moved to the right of the initial forward direction.
    
    Example:
        >>> # Agent starts facing North (heading=0), moves 10m North
        >>> start = [114.0, 22.5, 100.0]
        >>> current = [114.0, 22.50009, 100.0]  # ~10m north
        >>> fwd, right = lonlat_to_ego_displacement(start, 0.0, current)
        >>> # fwd ≈ 10.0, right ≈ 0.0
    """
    lon0 = float(start_position[0])
    lat0 = float(start_position[1])
    lon_t = float(current_position[0])
    lat_t = float(current_position[1])
    
    # Calculate geographic displacement in meters using spherical approximation
    # This is simpler and avoids dependency on GeoUtils/pyproj
    lat_mid_rad = math.radians((lat0 + lat_t) / 2.0)
    
    # East displacement: longitude difference * cos(lat) * earth_radius
    delta_east_m = math.radians(lon_t - lon0) * EARTH_RADIUS_METERS * math.cos(lat_mid_rad)
    # North displacement: latitude difference * earth_radius
    delta_north_m = math.radians(lat_t - lat0) * EARTH_RADIUS_METERS
    
    # Rotate geographic (east, north) into ego-frame (right, forward)
    # heading_0 is measured from North, clockwise
    # When heading=0 (facing North): forward=north, right=east
    # When heading=90 (facing East): forward=east, right=-north (south)
    theta_0 = math.radians(start_heading_deg)
    
    delta_forward = delta_north_m * math.cos(theta_0) + delta_east_m * math.sin(theta_0)
    delta_right = delta_east_m * math.cos(theta_0) - delta_north_m * math.sin(theta_0)
    
    return (float(delta_forward), float(delta_right))


def wrap_heading_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180] degrees.
    
    Args:
        angle_deg: Angle in degrees.
        
    Returns:
        Wrapped angle in [-180, 180] degrees.
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def geodesic_distance_with_altitude(
    position_a: Sequence[float],
    position_b: Sequence[float]
) -> float:
    """Calculate geodesic distance between two positions including altitude difference.
    
    This function calculates the 3D distance between two points, accounting for
    both horizontal distance (using Haversine formula) and vertical distance (altitude).
    
    Args:
        position_a: First position as [longitude, latitude, altitude].
        position_b: Second position as [longitude, latitude, altitude].
        
    Returns:
        Distance in meters between the two positions (3D distance).
        
    Example:
        >>> pos_a = [116.3974, 39.9093, 100.0]  # 100m altitude
        >>> pos_b = [116.3975, 39.9094, 150.0]  # 150m altitude
        >>> distance = geodesic_distance_with_altitude(pos_a, pos_b)
    """
    # Calculate horizontal distance using Haversine
    horizontal_dist = geodesic_distance(position_a, position_b)
    
    # Extract altitudes
    alt_a = float(position_a[2]) if len(position_a) > 2 else 0.0
    alt_b = float(position_b[2]) if len(position_b) > 2 else 0.0
    
    # Calculate vertical distance
    vertical_dist = abs(alt_b - alt_a)
    
    # Calculate 3D distance (Pythagorean theorem)
    distance_3d = math.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)
    
    return float(distance_3d)


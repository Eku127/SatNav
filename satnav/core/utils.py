#!/usr/bin/env python3
"""Utility functions for SatNav."""

import math
from typing import Sequence, Union

import numpy as np

# Earth's radius in meters (WGS84 ellipsoid mean radius)
EARTH_RADIUS_METERS = 6371000.0


def geodesic_distance(
    position_a: Union[Sequence[float], np.ndarray],
    position_b: Union[Sequence[float], np.ndarray]
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
    # Convert to numpy arrays for easier manipulation
    pos_a = np.array(position_a, dtype=np.float64)
    pos_b = np.array(position_b, dtype=np.float64)
    
    # Extract longitude and latitude (ignore altitude for distance calculation)
    lon1, lat1 = np.radians(pos_a[0]), np.radians(pos_a[1])
    lon2, lat2 = np.radians(pos_b[0]), np.radians(pos_b[1])
    
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


def geodesic_distance_with_altitude(
    position_a: Union[Sequence[float], np.ndarray],
    position_b: Union[Sequence[float], np.ndarray]
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


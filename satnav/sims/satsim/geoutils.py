#!/usr/bin/env python3
"""Geographic coordinate transformation utilities for SatSim."""

import math
from typing import Tuple

import numpy as np
from pyproj import Transformer


class GeoUtils:
    """Geographic coordinate transformation utilities.
    
    This class provides static methods for converting between WGS84 (EPSG:4326)
    geographic coordinates and Web Mercator (EPSG:3857) projected coordinates.
    """
    
    # Initialize transformers (cached for efficiency)
    _wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    _mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    @staticmethod
    def wgs84_to_mercator(lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 geographic coordinates to Web Mercator.
        
        Args:
            lon: Longitude in degrees.
            lat: Latitude in degrees.
            
        Returns:
            Tuple of (x, y) in meters in Web Mercator projection.
        """
        x, y = GeoUtils._wgs84_to_mercator.transform(lon, lat)
        return (float(x), float(y))
    
    @staticmethod
    def mercator_to_wgs84(x: float, y: float) -> Tuple[float, float]:
        """Convert Web Mercator coordinates to WGS84 geographic coordinates.
        
        Args:
            x: X coordinate in meters (Web Mercator).
            y: Y coordinate in meters (Web Mercator).
            
        Returns:
            Tuple of (longitude, latitude) in degrees.
        """
        lon, lat = GeoUtils._mercator_to_wgs84.transform(x, y)
        return (float(lon), float(lat))
    
    @staticmethod
    def move_in_mercator(
        x: float,
        y: float,
        distance_m: float,
        heading_deg: float
    ) -> Tuple[float, float]:
        """Move a point in Mercator space by a given distance and heading.
        
        This function moves a point in Web Mercator coordinates by a specified
        distance (in meters) in a given heading direction (in degrees, 0=North).
        
        Args:
            x: Current X coordinate in meters (Web Mercator).
            y: Current Y coordinate in meters (Web Mercator).
            distance_m: Distance to move in meters.
            heading_deg: Heading direction in degrees (0=North, 90=East).
            
        Returns:
            Tuple of (x_new, y_new) in meters (Web Mercator).
        """
        # Convert heading to radians
        heading_rad = math.radians(heading_deg)
        
        # Calculate displacement
        dx = distance_m * math.sin(heading_rad)
        dy = distance_m * math.cos(heading_rad)
        
        # Apply displacement
        x_new = x + dx
        y_new = y + dy
        
        return (float(x_new), float(y_new))
    
    @staticmethod
    def position_wgs84_to_mercator(position: np.ndarray) -> Tuple[float, float, float]:
        """Convert WGS84 position [lon, lat, alt] to Mercator [x, y, alt].
        
        Args:
            position: Position array [longitude, latitude, altitude].
            
        Returns:
            Tuple of (x, y, altitude) where (x, y) are in meters (Web Mercator)
            and altitude is unchanged.
        """
        lon, lat, alt = position[0], position[1], position[2]
        x, y = GeoUtils.wgs84_to_mercator(lon, lat)
        return (x, y, float(alt))
    
    @staticmethod
    def position_mercator_to_wgs84(position: Tuple[float, float, float]) -> np.ndarray:
        """Convert Mercator position [x, y, alt] to WGS84 [lon, lat, alt].
        
        Args:
            position: Position tuple (x, y, altitude) where (x, y) are in meters
                (Web Mercator) and altitude is in meters.
            
        Returns:
            Numpy array [longitude, latitude, altitude] in degrees and meters.
        """
        x, y, alt = position
        lon, lat = GeoUtils.mercator_to_wgs84(x, y)
        return np.array([lon, lat, alt], dtype=np.float32)


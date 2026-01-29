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
    def get_mercator_scale_factor(y: float) -> float:
        """Get Mercator scale factor at given Mercator Y coordinate.
        
        The Mercator projection distorts distances by a factor of 1/cos(latitude).
        At the equator (lat=0), scale_factor=1, so 1 Mercator meter = 1 true meter.
        At higher latitudes, scale_factor > 1, so Mercator meters are "stretched".
        
        To convert from true ground meters to Mercator meters, multiply by this factor.
        To convert from Mercator meters to true ground meters, divide by this factor.
        
        Args:
            y: Y coordinate in meters (Web Mercator).
            
        Returns:
            Scale factor = 1/cos(latitude). 
            Multiply true ground meters by this to get Mercator meters.
            
        Example:
            At latitude 37.5° (San Francisco):
                scale_factor ≈ 1.26
                10 true ground meters = 12.6 Mercator meters
        """
        R = 6378137.0  # WGS84 semi-major axis in meters
        lat_rad = math.atan(math.sinh(y / R))
        return 1.0 / math.cos(lat_rad)
    
    @staticmethod
    def true_meters_to_mercator(distance_m: float, y: float) -> float:
        """Convert true ground meters to Mercator meters.
        
        Args:
            distance_m: Distance in true ground meters.
            y: Y coordinate in meters (Web Mercator) at the location.
            
        Returns:
            Distance in Mercator meters.
        """
        return distance_m * GeoUtils.get_mercator_scale_factor(y)
    
    @staticmethod
    def mercator_meters_to_true(distance_mercator: float, y: float) -> float:
        """Convert Mercator meters to true ground meters.
        
        Args:
            distance_mercator: Distance in Mercator meters.
            y: Y coordinate in meters (Web Mercator) at the location.
            
        Returns:
            Distance in true ground meters.
        """
        return distance_mercator / GeoUtils.get_mercator_scale_factor(y)
    
    @staticmethod
    def move_in_mercator(
        x: float,
        y: float,
        distance_m: float,
        heading_deg: float
    ) -> Tuple[float, float]:
        """Move a point in Mercator space by a given distance and heading.
        
        This function moves a point in Web Mercator coordinates by a specified
        distance (in TRUE GROUND meters) in a given heading direction.
        
        The distance is converted from true ground meters to Mercator meters
        using the Mercator scale factor at the current latitude, ensuring that
        the agent moves the expected real-world distance.
        
        Args:
            x: Current X coordinate in meters (Web Mercator).
            y: Current Y coordinate in meters (Web Mercator).
            distance_m: Distance to move in TRUE GROUND meters (not Mercator meters).
            heading_deg: Heading direction in degrees (0=North, 90=East).
            
        Returns:
            Tuple of (x_new, y_new) in meters (Web Mercator).
            
        Note:
            Before Mercator scale factor correction, a distance_m of 10m would
            result in moving 10 Mercator meters, which at latitude 37.5° equals
            only ~7.9 true ground meters. After correction, 10m means 10 true
            ground meters regardless of latitude.
        """
        # Convert true ground meters to Mercator meters
        # This ensures the agent moves the expected real-world distance
        distance_mercator = GeoUtils.true_meters_to_mercator(distance_m, y)
        
        # Convert heading to radians
        heading_rad = math.radians(heading_deg)
        
        # Calculate displacement in Mercator meters
        dx = distance_mercator * math.sin(heading_rad)
        dy = distance_mercator * math.cos(heading_rad)
        
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


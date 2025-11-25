#!/usr/bin/env python3
"""Tests for utility functions."""

import math
import pytest
import numpy as np

from satnav.core.utils import (
    geodesic_distance,
    geodesic_distance_with_altitude,
    EARTH_RADIUS_METERS,
)


class TestGeodesicDistance:
    """Test cases for geodesic_distance function."""

    def test_same_position(self):
        """Test distance between same position is zero."""
        pos = [116.3974, 39.9093, 100.0]
        distance = geodesic_distance(pos, pos)
        assert distance == 0.0

    def test_opposite_points_on_equator(self):
        """Test distance between opposite points on equator."""
        # Two points 180 degrees apart on equator
        pos_a = [0.0, 0.0, 0.0]
        pos_b = [180.0, 0.0, 0.0]
        distance = geodesic_distance(pos_a, pos_b)
        # Should be approximately half the Earth's circumference
        expected = math.pi * EARTH_RADIUS_METERS
        assert abs(distance - expected) < 1000.0  # Within 1km tolerance

    def test_north_pole_to_equator(self):
        """Test distance from North Pole to equator."""
        # North Pole to point on equator
        pos_a = [0.0, 90.0, 0.0]  # North Pole
        pos_b = [0.0, 0.0, 0.0]   # Equator
        distance = geodesic_distance(pos_a, pos_b)
        # Should be approximately quarter of Earth's circumference
        expected = math.pi * EARTH_RADIUS_METERS / 2
        assert abs(distance - expected) < 1000.0  # Within 1km tolerance

    def test_beijing_to_shanghai(self):
        """Test distance between Beijing and Shanghai (known cities)."""
        # Beijing: 116.3974°E, 39.9093°N
        # Shanghai: 121.4737°E, 31.2304°N
        beijing = [116.3974, 39.9093, 0.0]
        shanghai = [121.4737, 31.2304, 0.0]
        distance = geodesic_distance(beijing, shanghai)
        # Known distance is approximately 1067 km
        expected_km = 1067.0
        assert abs(distance / 1000.0 - expected_km) < 50.0  # Within 50km tolerance

    def test_small_distance(self):
        """Test distance calculation for nearby points."""
        # Two points approximately 1 km apart
        pos_a = [116.3974, 39.9093, 0.0]
        # Move approximately 0.01 degrees (roughly 1 km at this latitude)
        pos_b = [116.4074, 39.9093, 0.0]
        distance = geodesic_distance(pos_a, pos_b)
        # Should be approximately 1 km
        assert 500.0 < distance < 2000.0  # Between 500m and 2km

    def test_numpy_array_input(self):
        """Test that function works with numpy arrays."""
        pos_a = np.array([116.3974, 39.9093, 100.0])
        pos_b = np.array([116.3975, 39.9094, 100.0])
        distance = geodesic_distance(pos_a, pos_b)
        assert distance > 0.0
        assert isinstance(distance, float)

    def test_list_input(self):
        """Test that function works with Python lists."""
        pos_a = [116.3974, 39.9093, 100.0]
        pos_b = [116.3975, 39.9094, 100.0]
        distance = geodesic_distance(pos_a, pos_b)
        assert distance > 0.0
        assert isinstance(distance, float)

    def test_two_coordinates_only(self):
        """Test that function works with only longitude and latitude."""
        pos_a = [116.3974, 39.9093]  # No altitude
        pos_b = [116.3975, 39.9094]  # No altitude
        distance = geodesic_distance(pos_a, pos_b)
        assert distance > 0.0

    def test_altitude_ignored(self):
        """Test that altitude doesn't affect horizontal distance calculation."""
        pos_a_low = [116.3974, 39.9093, 0.0]
        pos_a_high = [116.3974, 39.9093, 10000.0]  # 10km altitude
        pos_b = [116.3975, 39.9094, 0.0]
        
        dist_low = geodesic_distance(pos_a_low, pos_b)
        dist_high = geodesic_distance(pos_a_high, pos_b)
        
        # Horizontal distances should be the same
        assert abs(dist_low - dist_high) < 0.01  # Should be identical


class TestGeodesicDistanceWithAltitude:
    """Test cases for geodesic_distance_with_altitude function."""

    def test_same_position_same_altitude(self):
        """Test distance between same position and altitude is zero."""
        pos = [116.3974, 39.9093, 100.0]
        distance = geodesic_distance_with_altitude(pos, pos)
        assert distance == 0.0

    def test_same_position_different_altitude(self):
        """Test distance when only altitude differs."""
        pos_a = [116.3974, 39.9093, 0.0]
        pos_b = [116.3974, 39.9093, 1000.0]  # 1km higher
        distance = geodesic_distance_with_altitude(pos_a, pos_b)
        # Should be exactly 1000 meters (vertical distance)
        assert abs(distance - 1000.0) < 0.01

    def test_different_position_same_altitude(self):
        """Test that 3D distance equals horizontal distance when altitudes are same."""
        pos_a = [116.3974, 39.9093, 100.0]
        pos_b = [116.3975, 39.9094, 100.0]  # Same altitude
        dist_3d = geodesic_distance_with_altitude(pos_a, pos_b)
        dist_horizontal = geodesic_distance(pos_a, pos_b)
        # Should be approximately equal (within small tolerance)
        assert abs(dist_3d - dist_horizontal) < 0.01

    def test_different_position_different_altitude(self):
        """Test 3D distance calculation with both horizontal and vertical components."""
        pos_a = [116.3974, 39.9093, 0.0]
        pos_b = [116.3975, 39.9094, 1000.0]  # Different position and altitude
        dist_3d = geodesic_distance_with_altitude(pos_a, pos_b)
        dist_horizontal = geodesic_distance(pos_a, pos_b)
        
        # 3D distance should be greater than horizontal distance
        assert dist_3d > dist_horizontal
        
        # Should satisfy Pythagorean theorem (approximately)
        vertical_dist = 1000.0
        expected_3d = math.sqrt(dist_horizontal ** 2 + vertical_dist ** 2)
        assert abs(dist_3d - expected_3d) < 0.01

    def test_numpy_array_input(self):
        """Test that function works with numpy arrays."""
        pos_a = np.array([116.3974, 39.9093, 100.0])
        pos_b = np.array([116.3975, 39.9094, 200.0])
        distance = geodesic_distance_with_altitude(pos_a, pos_b)
        assert distance > 0.0
        assert isinstance(distance, float)

    def test_altitude_difference_only(self):
        """Test that function correctly calculates altitude difference."""
        # Same horizontal position, different altitudes
        pos_a = [116.3974, 39.9093, 0.0]
        pos_b = [116.3974, 39.9093, 500.0]
        distance = geodesic_distance_with_altitude(pos_a, pos_b)
        assert abs(distance - 500.0) < 0.01


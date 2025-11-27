#!/usr/bin/env python3
"""Tests for GeoUtils coordinate transformation utilities.

Note: These tests require the following dependencies to be installed:
    - pyproj (for coordinate transformations)
    - numpy (for array operations)

Install dependencies with:
    pip install pyproj numpy
"""

import math
import pytest
import numpy as np

from satnav.sims.satsim.geoutils import GeoUtils


class TestWGS84ToMercator:
    """Test cases for wgs84_to_mercator conversion."""

    def test_origin(self):
        """Test conversion of origin (0, 0)."""
        lon, lat = 0.0, 0.0
        x, y = GeoUtils.wgs84_to_mercator(lon, lat)
        # Origin should map to (0, 0) in Mercator
        assert abs(x) < 1.0  # Very close to 0
        assert abs(y) < 1.0  # Very close to 0

    def test_beijing(self):
        """Test conversion of Beijing coordinates."""
        # Beijing: 116.3974°E, 39.9093°N
        lon, lat = 116.3974, 39.9093
        x, y = GeoUtils.wgs84_to_mercator(lon, lat)
        
        # Beijing should be in eastern hemisphere (positive x)
        assert x > 0
        # Beijing should be in northern hemisphere (positive y)
        assert y > 0
        # Values should be reasonable (in meters)
        assert 1e6 < abs(x) < 2e7  # Roughly 10^6 to 10^7 meters
        assert 1e6 < abs(y) < 2e7

    def test_round_trip(self):
        """Test round-trip conversion (WGS84 -> Mercator -> WGS84)."""
        lon_orig, lat_orig = 116.3974, 39.9093
        x, y = GeoUtils.wgs84_to_mercator(lon_orig, lat_orig)
        lon_back, lat_back = GeoUtils.mercator_to_wgs84(x, y)
        
        # Should recover original coordinates (within reasonable precision)
        assert abs(lon_back - lon_orig) < 1e-6  # Within 0.000001 degrees
        assert abs(lat_back - lat_orig) < 1e-6

    def test_multiple_locations(self):
        """Test conversion of multiple known locations."""
        test_cases = [
            (0.0, 0.0),  # Equator, Prime Meridian
            (180.0, 0.0),  # Equator, International Date Line
            (0.0, 90.0),  # North Pole
            (0.0, -90.0),  # South Pole
            (116.3974, 39.9093),  # Beijing
            (-74.0060, 40.7128),  # New York
        ]
        
        for lon, lat in test_cases:
            x, y = GeoUtils.wgs84_to_mercator(lon, lat)
            # Check that conversion produces valid results
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert not (math.isnan(x) or math.isnan(y))
            assert not (math.isinf(x) or math.isinf(y))

    def test_return_type(self):
        """Test that function returns tuple of floats."""
        lon, lat = 116.3974, 39.9093
        result = GeoUtils.wgs84_to_mercator(lon, lat)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestMercatorToWGS84:
    """Test cases for mercator_to_wgs84 conversion."""

    def test_origin(self):
        """Test conversion of origin (0, 0) in Mercator."""
        x, y = 0.0, 0.0
        lon, lat = GeoUtils.mercator_to_wgs84(x, y)
        # Origin should map to (0, 0) in WGS84
        assert abs(lon) < 1e-6
        assert abs(lat) < 1e-6

    def test_beijing_mercator(self):
        """Test conversion of Beijing's approximate Mercator coordinates."""
        # Approximate Mercator coordinates for Beijing area
        # These are rough estimates - we'll verify round-trip instead
        lon_orig, lat_orig = 116.3974, 39.9093
        x, y = GeoUtils.wgs84_to_mercator(lon_orig, lat_orig)
        lon_back, lat_back = GeoUtils.mercator_to_wgs84(x, y)
        
        # Should recover original coordinates
        assert abs(lon_back - lon_orig) < 1e-6
        assert abs(lat_back - lat_orig) < 1e-6

    def test_round_trip(self):
        """Test round-trip conversion (Mercator -> WGS84 -> Mercator)."""
        # Use known Mercator coordinates
        lon_orig, lat_orig = 116.3974, 39.9093
        x_orig, y_orig = GeoUtils.wgs84_to_mercator(lon_orig, lat_orig)
        
        lon, lat = GeoUtils.mercator_to_wgs84(x_orig, y_orig)
        x_back, y_back = GeoUtils.wgs84_to_mercator(lon, lat)
        
        # Should recover original Mercator coordinates
        assert abs(x_back - x_orig) < 0.1  # Within 0.1 meters
        assert abs(y_back - y_orig) < 0.1

    def test_return_type(self):
        """Test that function returns tuple of floats."""
        x, y = 1.0e7, 4.0e6  # Some Mercator coordinates
        result = GeoUtils.mercator_to_wgs84(x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestMoveInMercator:
    """Test cases for move_in_mercator function."""

    def test_move_north(self):
        """Test moving north (heading = 0 degrees)."""
        x, y = 0.0, 0.0
        distance = 1000.0  # 1 km
        heading = 0.0  # North
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        # Moving north should increase y coordinate
        assert y_new > y
        # X should remain approximately the same
        assert abs(x_new - x) < 0.1
        # Distance should be approximately correct
        dist_moved = math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)
        assert abs(dist_moved - distance) < 0.1

    def test_move_east(self):
        """Test moving east (heading = 90 degrees)."""
        x, y = 0.0, 0.0
        distance = 1000.0  # 1 km
        heading = 90.0  # East
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        # Moving east should increase x coordinate
        assert x_new > x
        # Y should remain approximately the same
        assert abs(y_new - y) < 0.1
        # Distance should be approximately correct
        dist_moved = math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)
        assert abs(dist_moved - distance) < 0.1

    def test_move_south(self):
        """Test moving south (heading = 180 degrees)."""
        x, y = 0.0, 0.0
        distance = 1000.0  # 1 km
        heading = 180.0  # South
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        # Moving south should decrease y coordinate
        assert y_new < y
        # X should remain approximately the same
        assert abs(x_new - x) < 0.1

    def test_move_west(self):
        """Test moving west (heading = 270 degrees)."""
        x, y = 0.0, 0.0
        distance = 1000.0  # 1 km
        heading = 270.0  # West
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        # Moving west should decrease x coordinate
        assert x_new < x
        # Y should remain approximately the same
        assert abs(y_new - y) < 0.1

    def test_move_northeast(self):
        """Test moving northeast (heading = 45 degrees)."""
        x, y = 0.0, 0.0
        distance = 1000.0  # 1 km
        heading = 45.0  # Northeast
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        # Moving northeast should increase both x and y
        assert x_new > x
        assert y_new > y
        # Distance should be correct
        dist_moved = math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)
        assert abs(dist_moved - distance) < 0.1

    def test_zero_distance(self):
        """Test that zero distance results in no movement."""
        x, y = 100.0, 200.0
        distance = 0.0
        heading = 45.0
        
        x_new, y_new = GeoUtils.move_in_mercator(x, y, distance, heading)
        
        assert abs(x_new - x) < 1e-10
        assert abs(y_new - y) < 1e-10

    def test_heading_wraparound(self):
        """Test that heading wraps around correctly (360 = 0)."""
        x, y = 0.0, 0.0
        distance = 1000.0
        
        x1, y1 = GeoUtils.move_in_mercator(x, y, distance, 0.0)
        x2, y2 = GeoUtils.move_in_mercator(x, y, distance, 360.0)
        
        # 0 degrees and 360 degrees should give same result
        assert abs(x1 - x2) < 0.1
        assert abs(y1 - y2) < 0.1

    def test_negative_heading(self):
        """Test that negative heading is handled correctly."""
        x, y = 0.0, 0.0
        distance = 1000.0
        
        # -90 degrees should be equivalent to 270 degrees (west)
        x_neg, y_neg = GeoUtils.move_in_mercator(x, y, distance, -90.0)
        x_270, y_270 = GeoUtils.move_in_mercator(x, y, distance, 270.0)
        
        assert abs(x_neg - x_270) < 0.1
        assert abs(y_neg - y_270) < 0.1

    def test_return_type(self):
        """Test that function returns tuple of floats."""
        x, y = 0.0, 0.0
        result = GeoUtils.move_in_mercator(x, y, 1000.0, 45.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestPositionWGS84ToMercator:
    """Test cases for position_wgs84_to_mercator conversion."""

    def test_basic_conversion(self):
        """Test basic position conversion."""
        position = np.array([116.3974, 39.9093, 100.0], dtype=np.float32)
        result = GeoUtils.position_wgs84_to_mercator(position)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], float)  # x
        assert isinstance(result[1], float)  # y
        assert isinstance(result[2], float)  # altitude
        assert result[2] == 100.0  # Altitude should be preserved

    def test_altitude_preserved(self):
        """Test that altitude is preserved in conversion."""
        altitudes = [0.0, 100.0, 1000.0, 10000.0]
        lon, lat = 116.3974, 39.9093
        
        for alt in altitudes:
            position = np.array([lon, lat, alt], dtype=np.float32)
            x, y, alt_result = GeoUtils.position_wgs84_to_mercator(position)
            assert abs(alt_result - alt) < 1e-6

    def test_round_trip(self):
        """Test round-trip conversion."""
        position_orig = np.array([116.3974, 39.9093, 100.0], dtype=np.float32)
        mercator_pos = GeoUtils.position_wgs84_to_mercator(position_orig)
        position_back = GeoUtils.position_mercator_to_wgs84(mercator_pos)
        
        # Should recover original position (within precision)
        assert abs(position_back[0] - position_orig[0]) < 1e-6
        assert abs(position_back[1] - position_orig[1]) < 1e-6
        assert abs(position_back[2] - position_orig[2]) < 1e-6

    def test_list_input(self):
        """Test that function works with list input."""
        position = [116.3974, 39.9093, 100.0]
        result = GeoUtils.position_wgs84_to_mercator(np.array(position))
        assert len(result) == 3

    def test_multiple_positions(self):
        """Test conversion of multiple positions."""
        positions = [
            [0.0, 0.0, 0.0],
            [116.3974, 39.9093, 100.0],
            [-74.0060, 40.7128, 50.0],
            [180.0, 0.0, 200.0],
        ]
        
        for pos in positions:
            position = np.array(pos, dtype=np.float32)
            result = GeoUtils.position_wgs84_to_mercator(position)
            assert len(result) == 3
            assert result[2] == pos[2]  # Altitude preserved


class TestPositionMercatorToWGS84:
    """Test cases for position_mercator_to_wgs84 conversion."""

    def test_basic_conversion(self):
        """Test basic position conversion."""
        # Convert Beijing to Mercator first
        lon_orig, lat_orig, alt = 116.3974, 39.9093, 100.0
        x, y = GeoUtils.wgs84_to_mercator(lon_orig, lat_orig)
        mercator_pos = (x, y, alt)
        
        result = GeoUtils.position_mercator_to_wgs84(mercator_pos)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 3
        # Use slightly relaxed tolerance (2e-6) due to coordinate transformation precision
        # and float32 limitations
        assert abs(result[0] - lon_orig) < 2e-6
        assert abs(result[1] - lat_orig) < 2e-6
        assert abs(result[2] - alt) < 1e-6

    def test_altitude_preserved(self):
        """Test that altitude is preserved in conversion."""
        lon, lat = 116.3974, 39.9093
        x, y = GeoUtils.wgs84_to_mercator(lon, lat)
        altitudes = [0.0, 100.0, 1000.0, 10000.0]
        
        for alt in altitudes:
            mercator_pos = (x, y, alt)
            result = GeoUtils.position_mercator_to_wgs84(mercator_pos)
            assert abs(result[2] - alt) < 1e-6

    def test_round_trip(self):
        """Test round-trip conversion."""
        position_orig = np.array([116.3974, 39.9093, 100.0], dtype=np.float32)
        mercator_pos = GeoUtils.position_wgs84_to_mercator(position_orig)
        position_back = GeoUtils.position_mercator_to_wgs84(mercator_pos)
        
        # Should recover original position
        assert abs(position_back[0] - position_orig[0]) < 1e-6
        assert abs(position_back[1] - position_orig[1]) < 1e-6
        assert abs(position_back[2] - position_orig[2]) < 1e-6

    def test_return_type(self):
        """Test that function returns numpy array with correct dtype."""
        x, y = 1.0e7, 4.0e6
        mercator_pos = (x, y, 100.0)
        result = GeoUtils.position_mercator_to_wgs84(mercator_pos)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 3

    def test_multiple_positions(self):
        """Test conversion of multiple positions."""
        # Create some Mercator positions
        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0e7, 4.0e6, 100.0),
            (-1.0e7, 4.0e6, 50.0),
        ]
        
        for x, y, alt in test_cases:
            mercator_pos = (x, y, alt)
            result = GeoUtils.position_mercator_to_wgs84(mercator_pos)
            assert len(result) == 3
            assert result[2] == alt  # Altitude preserved
            # Check that longitude and latitude are reasonable
            assert -180.0 <= result[0] <= 180.0
            assert -90.0 <= result[1] <= 90.0


class TestGeoUtilsIntegration:
    """Integration tests for GeoUtils methods working together."""

    def test_full_navigation_cycle(self):
        """Test a complete navigation cycle: convert, move, convert back."""
        # Start position
        start_pos_wgs84 = np.array([116.3974, 39.9093, 100.0], dtype=np.float32)
        
        # Convert to Mercator
        start_pos_mercator = GeoUtils.position_wgs84_to_mercator(start_pos_wgs84)
        x, y, alt = start_pos_mercator
        
        # Move forward 1000 meters north
        x_new, y_new = GeoUtils.move_in_mercator(x, y, 1000.0, 0.0)
        
        # Convert back to WGS84
        end_pos_mercator = (x_new, y_new, alt)
        end_pos_wgs84 = GeoUtils.position_mercator_to_wgs84(end_pos_mercator)
        
        # Verify results
        assert isinstance(end_pos_wgs84, np.ndarray)
        assert len(end_pos_wgs84) == 3
        # Should have moved north (latitude increased)
        assert end_pos_wgs84[1] > start_pos_wgs84[1]
        # Longitude should be approximately the same
        assert abs(end_pos_wgs84[0] - start_pos_wgs84[0]) < 0.01

    def test_coordinate_consistency(self):
        """Test that coordinate conversions are consistent."""
        # Test multiple points
        test_points = [
            [116.3974, 39.9093, 100.0],
            [0.0, 0.0, 0.0],
            [-74.0060, 40.7128, 50.0],
        ]
        
        for point in test_points:
            pos_wgs84 = np.array(point, dtype=np.float32)
            
            # Convert to Mercator and back
            pos_mercator = GeoUtils.position_wgs84_to_mercator(pos_wgs84)
            pos_wgs84_back = GeoUtils.position_mercator_to_wgs84(pos_mercator)
            
            # Should recover original (within precision)
            assert abs(pos_wgs84_back[0] - pos_wgs84[0]) < 1e-5
            assert abs(pos_wgs84_back[1] - pos_wgs84[1]) < 1e-5
            assert abs(pos_wgs84_back[2] - pos_wgs84[2]) < 1e-6


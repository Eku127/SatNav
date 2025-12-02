#!/usr/bin/env python3
"""Tests for TopDownMapSatNav measure and map utilities."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from satnav.utils.maps import (
    create_agent_sprite,
    geo_to_pixel,
    draw_point,
    draw_path,
    draw_agent,
    TOP_DOWN_MAP_COLORS,
)


class TestAgentSprite:
    """Tests for agent sprite creation."""
    
    def test_create_agent_sprite_shape(self):
        """Test that sprite has correct shape."""
        sprite = create_agent_sprite(size=30)
        assert sprite.shape == (30, 30, 4)
        assert sprite.dtype == np.uint8
    
    def test_create_agent_sprite_has_alpha(self):
        """Test that sprite has alpha channel."""
        sprite = create_agent_sprite(size=30)
        # Should have some transparent (0) and some opaque (255) pixels
        assert np.any(sprite[:, :, 3] == 0)  # Some transparent
        assert np.any(sprite[:, :, 3] == 255)  # Some opaque
    
    def test_create_agent_sprite_different_sizes(self):
        """Test sprite creation with different sizes."""
        for size in [20, 30, 50, 100]:
            sprite = create_agent_sprite(size=size)
            assert sprite.shape == (size, size, 4)


class TestGeoToPixel:
    """Tests for geographic to pixel coordinate conversion."""
    
    def test_geo_to_pixel_center(self):
        """Test conversion at map center."""
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        map_shape = (100, 100)
        
        # Center of map
        row, col = geo_to_pixel(-73.5, 40.5, bounds, map_shape)
        assert col == 49 or col == 50  # Center column
        assert row == 49 or row == 50  # Center row
    
    def test_geo_to_pixel_corners(self):
        """Test conversion at map corners."""
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        map_shape = (100, 100)
        
        # Top-left corner (max lat, min lon)
        row, col = geo_to_pixel(-74.0, 41.0, bounds, map_shape)
        assert row == 0
        assert col == 0
        
        # Bottom-right corner (min lat, max lon)
        row, col = geo_to_pixel(-73.0, 40.0, bounds, map_shape)
        assert row == 99
        assert col == 99
    
    def test_geo_to_pixel_clamping(self):
        """Test that out-of-bounds coordinates are clamped."""
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        map_shape = (100, 100)
        
        # Out of bounds
        row, col = geo_to_pixel(-75.0, 42.0, bounds, map_shape)
        assert 0 <= row < 100
        assert 0 <= col < 100


class TestDrawPoint:
    """Tests for drawing points on images."""
    
    def test_draw_point_modifies_image(self):
        """Test that draw_point modifies the image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_point(img, (50, 50), (255, 0, 0), radius=5)
        
        # Should have some non-zero pixels
        assert np.any(img > 0)
    
    def test_draw_point_correct_color(self):
        """Test that draw_point uses correct color."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (0, 255, 0)  # Green
        draw_point(img, (50, 50), color, radius=3)
        
        # Center pixel should be green
        assert img[50, 50, 1] == 255  # Green channel


class TestDrawPath:
    """Tests for drawing paths on images."""
    
    def test_draw_path_single_segment(self):
        """Test drawing a single line segment."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = [(10, 10), (90, 90)]
        draw_path(img, path, color=(255, 255, 255), thickness=2)
        
        # Should have some white pixels
        assert np.any(img == 255)
    
    def test_draw_path_gradient(self):
        """Test drawing a path with gradient colors."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = [(10, 10), (50, 50), (90, 90)]
        draw_path(img, path, color="gradient", thickness=2)
        
        # Should have some non-zero pixels
        assert np.any(img > 0)
    
    def test_draw_path_empty(self):
        """Test that empty path doesn't crash."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_path(img, [], color=(255, 255, 255))
        # Should be all zeros
        assert np.all(img == 0)
    
    def test_draw_path_single_point(self):
        """Test that single point path doesn't crash."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_path(img, [(50, 50)], color=(255, 255, 255))
        # Should be all zeros (no line to draw)
        assert np.all(img == 0)


class TestDrawAgent:
    """Tests for drawing agent on images."""
    
    def test_draw_agent_returns_image(self):
        """Test that draw_agent returns an image."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        
        result = draw_agent(img, [-73.5, 40.5, 100.0], 0.0, bounds)
        
        assert result.shape == img.shape
        assert result.dtype == np.uint8
    
    def test_draw_agent_modifies_image(self):
        """Test that draw_agent adds the agent sprite."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        
        result = draw_agent(img, [-73.5, 40.5, 100.0], 0.0, bounds)
        
        # Should have some non-zero pixels (the agent sprite)
        assert np.any(result > 0)
    
    def test_draw_agent_different_rotations(self):
        """Test drawing agent with different rotations."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bounds = {
            "lon_min": -74.0,
            "lon_max": -73.0,
            "lat_min": 40.0,
            "lat_max": 41.0,
        }
        
        for rotation in [0, 45, 90, 180, 270]:
            result = draw_agent(img.copy(), [-73.5, 40.5, 100.0], rotation, bounds)
            assert result.shape == img.shape


class TestTopDownMapColors:
    """Tests for color definitions."""
    
    def test_colors_defined(self):
        """Test that required colors are defined."""
        assert "source" in TOP_DOWN_MAP_COLORS
        assert "target" in TOP_DOWN_MAP_COLORS
        assert "reference_path" in TOP_DOWN_MAP_COLORS
        assert "agent_path_start" in TOP_DOWN_MAP_COLORS
        assert "agent_path_end" in TOP_DOWN_MAP_COLORS
    
    def test_colors_are_tuples(self):
        """Test that colors are BGR tuples."""
        for name, color in TOP_DOWN_MAP_COLORS.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255


class TestTopDownMapSatNavMeasure:
    """Tests for TopDownMapSatNav measure."""
    
    def test_measure_initialization(self):
        """Test that measure initializes correctly."""
        from satnav.task.measures import TopDownMapSatNav
        
        measure = TopDownMapSatNav(
            map_resolution=512,
            padding_meters=30.0,
        )
        
        assert measure._map_resolution == 512
        assert measure._padding_meters == 30.0
        assert measure._top_down_map is None
    
    def test_get_metric_before_reset(self):
        """Test get_metric returns default before reset."""
        from satnav.task.measures import TopDownMapSatNav
        
        measure = TopDownMapSatNav()
        metric = measure.get_metric()
        
        assert "map" in metric
        assert "agent_map_coord" in metric
        assert "agent_angle" in metric
        assert "bounds" in metric
        assert "step_count" in metric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


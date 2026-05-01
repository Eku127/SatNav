#!/usr/bin/env python3
"""Tests for SatelliteCamera rendering logic.

Note: These tests require the following dependencies to be installed:
    - rasterio (for TIF file handling)
    - numpy (for array operations)
    - scipy (for image rotation)
    - opencv-python (for image resizing)

Install dependencies with:
    pip install rasterio numpy scipy opencv-python
"""

import math
import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from satnav.sims.satsim.camera import SatelliteCamera


class TestSatelliteCameraInit:
    """Test cases for SatelliteCamera initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        camera = SatelliteCamera(width=224, height=224, hfov=90.0)
        assert camera.width == 224
        assert camera.height == 224
        assert camera.hfov == 90.0
        assert camera.aspect_ratio == 1.0

    def test_init_aspect_ratio(self):
        """Test aspect ratio calculation."""
        camera = SatelliteCamera(width=320, height=240, hfov=90.0)
        assert camera.aspect_ratio == pytest.approx(320 / 240)

    def test_init_different_hfov(self):
        """Test initialization with different HFOV values."""
        for hfov in [60.0, 90.0, 120.0]:
            camera = SatelliteCamera(width=224, height=224, hfov=hfov)
            assert camera.hfov == hfov


class TestGetViewBounds:
    """Test cases for get_view_bounds method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.camera = SatelliteCamera(width=224, height=224, hfov=90.0)
        self.position = (0.0, 0.0)  # Origin in Mercator

    def test_zero_rotation(self):
        """Test view bounds with zero rotation."""
        altitude = 100.0
        rotation = 0.0
        
        left, right, bottom, top = self.camera.get_view_bounds(
            self.position, altitude, rotation
        )
        
        # With zero rotation, bounds should be symmetric
        assert isinstance(left, float)
        assert isinstance(right, float)
        assert isinstance(bottom, float)
        assert isinstance(top, float)
        
        # Center should be at origin
        center_x = (left + right) / 2
        center_y = (bottom + top) / 2
        assert abs(center_x - self.position[0]) < 0.1
        assert abs(center_y - self.position[1]) < 0.1

    def test_90_degree_rotation(self):
        """Test view bounds with 90 degree rotation."""
        altitude = 100.0
        rotation = 90.0
        
        left, right, bottom, top = self.camera.get_view_bounds(
            self.position, altitude, rotation
        )
        
        # Bounds should still be valid
        assert left < right
        assert bottom < top
        
        # Width and height should be approximately equal (square view)
        width = right - left
        height = top - bottom
        assert abs(width - height) < 1.0  # Should be approximately equal

    def test_different_altitudes(self):
        """Test view bounds with different altitudes."""
        rotations = [0.0, 45.0, 90.0]
        altitudes = [50.0, 100.0, 200.0]
        
        for alt in altitudes:
            for rot in rotations:
                left, right, bottom, top = self.camera.get_view_bounds(
                    self.position, alt, rot
                )
                
                # Higher altitude should give larger bounds
                width = right - left
                height = top - bottom
                
                # Verify bounds are valid
                assert left < right
                assert bottom < top
                assert width > 0
                assert height > 0

    def test_altitude_scaling(self):
        """Test that bounds scale linearly with altitude."""
        altitude1 = 100.0
        altitude2 = 200.0
        
        left1, right1, bottom1, top1 = self.camera.get_view_bounds(
            self.position, altitude1, 0.0
        )
        left2, right2, bottom2, top2 = self.camera.get_view_bounds(
            self.position, altitude2, 0.0
        )
        
        width1 = right1 - left1
        width2 = right2 - left2
        height1 = top1 - bottom1
        height2 = top2 - bottom2
        
        # Width and height should approximately double when altitude doubles
        # (for small angles, tan(angle) ≈ angle, so linear scaling)
        ratio_w = width2 / width1
        ratio_h = height2 / height1
        
        assert 1.8 < ratio_w < 2.2  # Allow some tolerance
        assert 1.8 < ratio_h < 2.2

    def test_rotation_consistency(self):
        """Test that rotation at 90-degree intervals gives same bounds area."""
        altitude = 100.0
        # For square cameras, rotations at 90-degree intervals should give same bounds
        # For non-square cameras, 180-degree rotation should give same bounds
        rotations_90deg = [0.0, 90.0, 180.0, 270.0]
        
        areas_90deg = []
        for rot in rotations_90deg:
            left, right, bottom, top = self.camera.get_view_bounds(
                self.position, altitude, rot
            )
            width = right - left
            height = top - bottom
            area = width * height
            areas_90deg.append(area)
        
        # For square cameras, 90-degree rotations should give same bounds area
        # (because rotating a square by 90 degrees gives the same shape)
        if self.camera.aspect_ratio == 1.0:
            for area in areas_90deg[1:]:
                assert abs(area - areas_90deg[0]) / areas_90deg[0] < 0.1  # Within 10%
        
        # Test that 180-degree rotation gives same bounds as 0-degree
        # (this should hold for all aspect ratios)
        left_0, right_0, bottom_0, top_0 = self.camera.get_view_bounds(
            self.position, altitude, 0.0
        )
        left_180, right_180, bottom_180, top_180 = self.camera.get_view_bounds(
            self.position, altitude, 180.0
        )
        
        width_0 = right_0 - left_0
        height_0 = top_0 - bottom_0
        width_180 = right_180 - left_180
        height_180 = top_180 - bottom_180
        
        # 180-degree rotation should give same dimensions (just swapped)
        assert abs(width_0 - width_180) < 0.1
        assert abs(height_0 - height_180) < 0.1
        
        # Test that 45-degree rotation gives larger bounds (expected behavior)
        # When rotating a rectangle, the bounding box needs to expand
        left_45, right_45, bottom_45, top_45 = self.camera.get_view_bounds(
            self.position, altitude, 45.0
        )
        area_45 = (right_45 - left_45) * (top_45 - bottom_45)
        area_0 = width_0 * height_0
        
        # 45-degree rotation should give larger or equal bounds
        assert area_45 >= area_0 * 0.9  # At least 90% of original (could be larger)

    def test_360_degree_wraparound(self):
        """Test that 360 degrees equals 0 degrees."""
        altitude = 100.0
        
        left0, right0, bottom0, top0 = self.camera.get_view_bounds(
            self.position, altitude, 0.0
        )
        left360, right360, bottom360, top360 = self.camera.get_view_bounds(
            self.position, altitude, 360.0
        )
        
        # Should be approximately the same
        assert abs(left0 - left360) < 0.1
        assert abs(right0 - right360) < 0.1
        assert abs(bottom0 - bottom360) < 0.1
        assert abs(top0 - top360) < 0.1

    def test_negative_rotation(self):
        """Test that negative rotation is handled correctly."""
        altitude = 100.0
        
        left_pos, right_pos, bottom_pos, top_pos = self.camera.get_view_bounds(
            self.position, altitude, 45.0
        )
        left_neg, right_neg, bottom_neg, top_neg = self.camera.get_view_bounds(
            self.position, altitude, -45.0
        )
        
        # Negative rotation should give different but valid bounds
        assert left_neg < right_neg
        assert bottom_neg < top_neg

    def test_different_positions(self):
        """Test view bounds at different positions."""
        positions = [
            (0.0, 0.0),
            (1e6, 1e6),  # 1 million meters offset
            (-1e6, -1e6),
        ]
        altitude = 100.0
        rotation = 0.0
        
        for pos in positions:
            left, right, bottom, top = self.camera.get_view_bounds(
                pos, altitude, rotation
            )
            
            # Bounds should be centered around position
            center_x = (left + right) / 2
            center_y = (bottom + top) / 2
            assert abs(center_x - pos[0]) < 10.0  # Within 10 meters
            assert abs(center_y - pos[1]) < 10.0

    def test_aspect_ratio_effect(self):
        """Test that aspect ratio affects bounds correctly."""
        # Square camera
        camera_square = SatelliteCamera(width=224, height=224, hfov=90.0)
        # Wide camera
        camera_wide = SatelliteCamera(width=320, height=224, hfov=90.0)
        
        altitude = 100.0
        rotation = 0.0
        
        left_sq, right_sq, bottom_sq, top_sq = camera_square.get_view_bounds(
            self.position, altitude, rotation
        )
        left_w, right_w, bottom_w, top_w = camera_wide.get_view_bounds(
            self.position, altitude, rotation
        )
        
        width_sq = right_sq - left_sq
        width_w = right_w - left_w
        height_sq = top_sq - bottom_sq
        height_w = top_w - bottom_w
        
        # Wide camera should have wider bounds
        assert width_w > width_sq
        # With a fixed horizontal FOV, a wider aspect ratio reduces vertical coverage.
        assert height_w < height_sq


class TestRenderImage:
    """Test cases for render_image method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.camera = SatelliteCamera(width=224, height=224, hfov=90.0)
        self.position = (0.0, 0.0)
        self.altitude = 100.0

    def _create_mock_rasterio_dataset(self, bounds=None, shape=(3, 1000, 1000)):
        """Create a mock rasterio dataset for testing."""
        if bounds is None:
            bounds = (-1000.0, -1000.0, 1000.0, 1000.0)  # left, bottom, right, top
        
        mock_dataset = Mock(spec=rasterio.DatasetReader)
        mock_dataset.bounds = rasterio.coords.BoundingBox(
            left=bounds[0],
            bottom=bounds[1],
            right=bounds[2],
            top=bounds[3]
        )
        mock_dataset.transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            width=shape[2], height=shape[1]
        )
        mock_dataset.crs = rasterio.crs.CRS.from_epsg(3857)
        
        # Mock read method to return test image
        def mock_read(window=None):
            if window is None:
                return np.random.randint(0, 255, size=shape, dtype=np.uint8)
            else:
                # Return smaller image based on window
                return np.random.randint(0, 255, size=shape, dtype=np.uint8)
        
        mock_dataset.read = Mock(side_effect=mock_read)
        
        return mock_dataset

    def test_render_image_basic(self):
        """Test basic image rendering."""
        mock_dataset = self._create_mock_rasterio_dataset()
        
        # Mock the read method to return a predictable image
        test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        rotation = 0.0
        result = self.camera.render_image(
            mock_dataset, self.position, self.altitude, rotation
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (224, 224, 3)
        assert len(result.shape) == 3

    def test_render_image_with_rotation(self):
        """Test image rendering with rotation."""
        mock_dataset = self._create_mock_rasterio_dataset()
        
        test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        rotation = 45.0
        result = self.camera.render_image(
            mock_dataset, self.position, self.altitude, rotation
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (224, 224, 3)

    def test_render_image_different_rotations(self):
        """Test rendering with different rotation angles."""
        mock_dataset = self._create_mock_rasterio_dataset()
        
        test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        rotations = [0.0, 45.0, 90.0, 180.0, 270.0]
        
        for rotation in rotations:
            result = self.camera.render_image(
                mock_dataset, self.position, self.altitude, rotation
            )
            assert result.shape == (224, 224, 3)
            assert result.dtype == np.uint8

    def test_render_image_bounds_check(self):
        """Test that bounds checking works correctly."""
        # Create dataset with small bounds
        small_bounds = (-10.0, -10.0, 10.0, 10.0)
        mock_dataset = self._create_mock_rasterio_dataset(bounds=small_bounds)
        
        # Position far outside bounds
        position_outside = (1000.0, 1000.0)
        
        with pytest.raises(ValueError, match="Camera view bounds exceed image bounds"):
            self.camera.render_image(
                mock_dataset, position_outside, self.altitude, 0.0
            )

    def test_render_image_output_range(self):
        """Test that output image values are in valid range."""
        mock_dataset = self._create_mock_rasterio_dataset()
        
        test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        result = self.camera.render_image(
            mock_dataset, self.position, self.altitude, 0.0
        )
        
        # Check that values are in valid uint8 range
        assert result.min() >= 0
        assert result.max() <= 255

    def test_render_image_different_sizes(self):
        """Test rendering with different camera sizes."""
        sizes = [
            (224, 224),
            (320, 240),
            (640, 480),
        ]
        
        for width, height in sizes:
            camera = SatelliteCamera(width=width, height=height, hfov=90.0)
            mock_dataset = self._create_mock_rasterio_dataset()
            
            test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
            mock_dataset.read = Mock(return_value=test_image)
            
            result = camera.render_image(
                mock_dataset, self.position, self.altitude, 0.0
            )
            
            assert result.shape == (height, width, 3)

    def test_render_image_different_altitudes(self):
        """Test rendering with different altitudes."""
        mock_dataset = self._create_mock_rasterio_dataset()
        
        test_image = np.random.randint(0, 255, size=(3, 500, 500), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        altitudes = [50.0, 100.0, 200.0]
        
        for alt in altitudes:
            result = self.camera.render_image(
                mock_dataset, self.position, alt, 0.0
            )
            assert result.shape == (224, 224, 3)
            assert result.dtype == np.uint8

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_render_image_with_real_tif(self):
        """Test rendering with a real TIF file if available."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        with rasterio.open(tif_path) as dataset:
            # Reproject to EPSG:3857 if needed
            if dataset.crs and dataset.crs.to_epsg() != 3857:
                from rasterio.vrt import WarpedVRT
                with WarpedVRT(dataset, crs="EPSG:3857") as vrt:
                    # Get bounds in Mercator
                    bounds = vrt.bounds
                    # Use center of bounds as position
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.bottom + bounds.top) / 2
                    position = (center_x, center_y)
                    
                    result = self.camera.render_image(
                        vrt, position, 100.0, 0.0
                    )
                    
                    assert isinstance(result, np.ndarray)
                    assert result.dtype == np.uint8
                    assert result.shape == (224, 224, 3)
            else:
                # Already in EPSG:3857
                bounds = dataset.bounds
                center_x = (bounds.left + bounds.right) / 2
                center_y = (bounds.bottom + bounds.top) / 2
                position = (center_x, center_y)
                
                result = self.camera.render_image(
                    dataset, position, 100.0, 0.0
                )
                
                assert isinstance(result, np.ndarray)
                assert result.dtype == np.uint8
                assert result.shape == (224, 224, 3)


class TestSatelliteCameraIntegration:
    """Integration tests for SatelliteCamera."""

    def test_view_bounds_and_render_consistency(self):
        """Test that view bounds calculation is consistent with rendering."""
        camera = SatelliteCamera(width=224, height=224, hfov=90.0)
        position = (0.0, 0.0)
        altitude = 100.0
        rotation = 45.0
        
        # Calculate view bounds
        left, right, bottom, top = camera.get_view_bounds(
            position, altitude, rotation
        )
        
        # Verify bounds are reasonable
        assert left < right
        assert bottom < top
        
        # Create mock dataset that covers these bounds
        expanded_bounds = (
            left - 100.0, bottom - 100.0,
            right + 100.0, top + 100.0
        )
        mock_dataset = Mock(spec=rasterio.DatasetReader)
        mock_dataset.bounds = rasterio.coords.BoundingBox(
            left=expanded_bounds[0],
            bottom=expanded_bounds[1],
            right=expanded_bounds[2],
            top=expanded_bounds[3]
        )
        mock_dataset.transform = from_bounds(
            expanded_bounds[0], expanded_bounds[1],
            expanded_bounds[2], expanded_bounds[3],
            width=1000, height=1000
        )
        
        test_image = np.random.randint(0, 255, size=(3, 1000, 1000), dtype=np.uint8)
        mock_dataset.read = Mock(return_value=test_image)
        
        # Should be able to render without errors
        result = camera.render_image(mock_dataset, position, altitude, rotation)
        assert result.shape == (224, 224, 3)

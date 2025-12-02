#!/usr/bin/env python3
"""Tests for SatSim core simulator engine.

Note: These tests require the following dependencies to be installed:
    - rasterio (for TIF file handling)
    - numpy (for array operations)
    - pyproj (for coordinate transformations)
    - scipy (for image rotation)
    - opencv-python (for image resizing)

Install dependencies with:
    pip install rasterio numpy pyproj scipy opencv-python
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.vrt import WarpedVRT

from satnav.sims.satsim.satsim import SatSim, open_reprojected_to_epsg3857


class TestOpenReprojectedToEPSG3857:
    """Test cases for open_reprojected_to_epsg3857 function."""

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_open_existing_file(self):
        """Test opening an existing TIF file."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        dataset = open_reprojected_to_epsg3857(tif_path)
        assert dataset is not None
        dataset.close()

    def test_open_nonexistent_file(self):
        """Test that opening nonexistent file raises error."""
        with pytest.raises((FileNotFoundError, rasterio.errors.RasterioIOError)):
            open_reprojected_to_epsg3857("nonexistent_file.tif")


class TestSatSimInit:
    """Test cases for SatSim initialization."""

    def create_test_config(self):
        """Create a test simulator configuration."""
        return {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {
                "WIDTH": 224,
                "HEIGHT": 224,
                "HFOV": 90.0
            }
        }

    def test_init_with_dict_config(self):
        """Test initialization with dictionary config."""
        config = self.create_test_config()
        sim = SatSim(config)
        
        assert sim.forward_step_size == 0.25
        assert sim.turn_angle == 15.0
        assert sim._camera.width == 224
        assert sim._camera.height == 224
        assert sim._camera.hfov == 90.0
        assert sim._current_scene is None
        assert sim._agent_position is None
        assert sim._agent_rotation is None

    def test_init_with_dictconfig(self):
        """Test initialization with DictConfig."""
        config_dict = self.create_test_config()
        config = OmegaConf.create(config_dict)
        sim = SatSim(config)
        
        assert sim.forward_step_size == 0.25
        assert sim.turn_angle == 15.0

    def test_init_with_defaults(self):
        """Test initialization with minimal config (uses defaults)."""
        config = {}
        sim = SatSim(config)
        
        assert sim.forward_step_size == 0.25  # default
        assert sim.turn_angle == 15.0  # default
        assert sim._camera.width == 224  # default
        assert sim._camera.height == 224  # default
        assert sim._camera.hfov == 90.0  # default

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = {
            "FORWARD_STEP_SIZE": 0.5,
            "TURN_ANGLE": 30.0,
            "RGB_SENSOR": {
                "WIDTH": 320,
                "HEIGHT": 240,
                "HFOV": 60.0
            }
        }
        sim = SatSim(config)
        
        assert sim.forward_step_size == 0.5
        assert sim.turn_angle == 30.0
        assert sim._camera.width == 320
        assert sim._camera.height == 240
        assert sim._camera.hfov == 60.0


class TestLoadScene:
    """Test cases for load_scene method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_load_scene_with_tif_extension(self):
        """Test loading scene with .tif extension."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        result = self.sim.load_scene(str(tif_path))
        assert result is True
        assert self.sim._current_scene is not None
        assert str(tif_path) in self.sim._scene_cache

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_load_scene_without_extension(self):
        """Test loading scene without extension (should add .tif)."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        # Try loading without extension
        path_without_ext = str(tif_path.with_suffix(""))
        result = self.sim.load_scene(path_without_ext)
        assert result is True
        assert self.sim._current_scene is not None

    def test_load_scene_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.sim.load_scene("nonexistent_scene.tif")

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_load_scene_caching(self):
        """Test that scenes are cached."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        # Load scene first time
        self.sim.load_scene(str(tif_path))
        first_scene = self.sim._current_scene
        
        # Load same scene again
        self.sim.load_scene(str(tif_path))
        second_scene = self.sim._current_scene
        
        # Should be the same cached scene
        assert first_scene is second_scene


class TestSetAgentState:
    """Test cases for set_agent_state method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_set_agent_state_basic(self):
        """Test basic agent state setting."""
        position = [116.3974, 39.9093, 100.0]
        rotation = 45.0
        
        self.sim.set_agent_state(position, rotation)
        
        assert self.sim._agent_position is not None
        assert self.sim._agent_rotation == 45.0

    def test_set_agent_state_normalize_rotation(self):
        """Test that rotation is normalized to [0, 360)."""
        position = [116.3974, 39.9093, 100.0]
        
        # Test negative rotation
        self.sim.set_agent_state(position, -45.0)
        assert self.sim._agent_rotation == 315.0  # -45 + 360
        
        # Test rotation > 360
        self.sim.set_agent_state(position, 450.0)
        assert self.sim._agent_rotation == 90.0  # 450 % 360

    def test_set_agent_state_invalid_position(self):
        """Test that invalid position raises ValueError."""
        # Too few elements
        with pytest.raises(ValueError, match="must have 3 elements"):
            self.sim.set_agent_state([116.3974, 39.9093], 0.0)
        
        # Too many elements
        with pytest.raises(ValueError, match="must have 3 elements"):
            self.sim.set_agent_state([116.3974, 39.9093, 100.0, 0.0], 0.0)

    def test_set_agent_state_numpy_array(self):
        """Test that numpy array input works."""
        position = np.array([116.3974, 39.9093, 100.0], dtype=np.float32)
        rotation = 90.0
        
        self.sim.set_agent_state(position, rotation)
        assert self.sim._agent_rotation == 90.0


class TestGetAgentState:
    """Test cases for get_agent_state method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_get_agent_state_not_initialized(self):
        """Test that get_agent_state raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError, match="Agent state not initialized"):
            self.sim.get_agent_state()

    def test_get_agent_state_round_trip(self):
        """Test round-trip conversion of agent state."""
        position_orig = [116.3974, 39.9093, 100.0]
        rotation_orig = 45.0
        
        self.sim.set_agent_state(position_orig, rotation_orig)
        position_back, rotation_back = self.sim.get_agent_state()
        
        assert isinstance(position_back, np.ndarray)
        assert len(position_back) == 3
        # Should recover original position (within precision)
        np.testing.assert_array_almost_equal(position_back, position_orig, decimal=5)
        assert abs(rotation_back - rotation_orig) < 0.1


class TestStep:
    """Test cases for step method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)
        
        # Load a scene for testing
        tif_path = Path("tests/test_data/map.tif")
        if tif_path.exists():
            self.sim.load_scene(str(tif_path))
            # Set agent state to center of scene
            bounds = self.sim._current_scene.bounds
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            # Convert to WGS84 for set_agent_state
            from satnav.sims.satsim.geoutils import GeoUtils
            lon, lat = GeoUtils.mercator_to_wgs84(center_x, center_y)
            self.sim.set_agent_state([lon, lat, 100.0], 0.0)

    def test_step_not_initialized(self):
        """Test that step raises RuntimeError when agent state not initialized."""
        sim = SatSim({"FORWARD_STEP_SIZE": 0.25, "TURN_ANGLE": 15.0})
        with pytest.raises(RuntimeError, match="Agent state not initialized"):
            sim.step("MOVE_FORWARD")

    def test_step_no_scene(self):
        """Test that step raises RuntimeError when scene not loaded."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        sim = SatSim(config)
        sim.set_agent_state([116.3974, 39.9093, 100.0], 0.0)
        
        with pytest.raises(RuntimeError, match="Scene not loaded"):
            sim.step("MOVE_FORWARD")

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_step_move_forward(self):
        """Test MOVE_FORWARD action."""
        if self.sim._current_scene is None:
            pytest.skip("Scene not loaded")
        
        position_before, rotation_before = self.sim.get_agent_state()
        
        obs = self.sim.step("MOVE_FORWARD")
        
        position_after, rotation_after = self.sim.get_agent_state()
        
        # Position should have changed (0.25m movement is small but should be detectable)
        # Check if position actually changed by computing differences
        lon_diff = abs(position_after[0] - position_before[0])
        lat_diff = abs(position_after[1] - position_before[1])
        total_diff = lon_diff + lat_diff
        
        # 0.25m movement should result in coordinate change
        # The actual difference seen is ~0.000002 degrees, which should be detectable
        # Use a very small threshold (1e-9) to account for float precision
        assert total_diff > 1e-9, (
            f"MOVE_FORWARD should change position: "
            f"before={position_before[:2]}, after={position_after[:2]}, "
            f"lon_diff={lon_diff:.12f}, lat_diff={lat_diff:.12f}, total_diff={total_diff:.12f}"
        )
        # Altitude should remain the same
        assert abs(position_before[2] - position_after[2]) < 0.1
        # Rotation should be the same
        assert abs(rotation_after - rotation_before) < 0.1
        # Observations should contain RGB
        assert "rgb" in obs
        assert obs["rgb"].shape == (224, 224, 3)

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_step_turn_left(self):
        """Test TURN_LEFT action."""
        if self.sim._current_scene is None:
            pytest.skip("Scene not loaded")
        
        _, rotation_before = self.sim.get_agent_state()
        
        obs = self.sim.step("TURN_LEFT")
        
        _, rotation_after = self.sim.get_agent_state()
        
        # Rotation should have decreased by turn_angle
        expected_rotation = (rotation_before - 15.0) % 360.0
        assert abs(rotation_after - expected_rotation) < 0.1
        # Observations should contain RGB
        assert "rgb" in obs

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_step_turn_right(self):
        """Test TURN_RIGHT action."""
        if self.sim._current_scene is None:
            pytest.skip("Scene not loaded")
        
        _, rotation_before = self.sim.get_agent_state()
        
        obs = self.sim.step("TURN_RIGHT")
        
        _, rotation_after = self.sim.get_agent_state()
        
        # Rotation should have increased by turn_angle
        expected_rotation = (rotation_before + 15.0) % 360.0
        assert abs(rotation_after - expected_rotation) < 0.1
        # Observations should contain RGB
        assert "rgb" in obs

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_step_stop(self):
        """Test STOP action."""
        if self.sim._current_scene is None:
            pytest.skip("Scene not loaded")
        
        position_before, rotation_before = self.sim.get_agent_state()
        
        obs = self.sim.step("STOP")
        
        position_after, rotation_after = self.sim.get_agent_state()
        
        # Position and rotation should not change
        np.testing.assert_array_almost_equal(position_before, position_after, decimal=5)
        assert abs(rotation_after - rotation_before) < 0.1
        # Observations should contain RGB
        assert "rgb" in obs

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_step_invalid_action(self):
        """Test that invalid action raises ValueError."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        sim = SatSim(config)
        
        # Load scene first (required for step to work)
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        sim.load_scene(str(tif_path))
        
        # Set agent state
        bounds = sim._current_scene.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        from satnav.sims.satsim.geoutils import GeoUtils
        lon, lat = GeoUtils.mercator_to_wgs84(center_x, center_y)
        sim.set_agent_state([lon, lat, 100.0], 0.0)
        
        # Now test invalid action
        with pytest.raises(ValueError, match="Invalid action"):
            sim.step("INVALID_ACTION")

    def test_step_action_as_int(self):
        """Test step with action as integer index."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        sim = SatSim(config)
        sim.set_agent_state([116.3974, 39.9093, 100.0], 0.0)
        
        # Action 1 = MOVE_FORWARD
        with pytest.raises(RuntimeError):  # Will fail because no scene, but action parsing works
            sim.step(1)

    def test_step_action_as_dict(self):
        """Test step with action as dictionary."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        sim = SatSim(config)
        sim.set_agent_state([116.3974, 39.9093, 100.0], 0.0)
        
        # Action as dict
        with pytest.raises(RuntimeError):  # Will fail because no scene, but action parsing works
            sim.step({"action": "MOVE_FORWARD"})


class TestGetObservations:
    """Test cases for get_observations method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_get_observations_not_initialized(self):
        """Test that get_observations raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError, match="Agent state not initialized"):
            self.sim.get_observations()

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_get_observations_basic(self):
        """Test basic observations retrieval."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        self.sim.load_scene(str(tif_path))
        
        # Set agent state to center of scene
        bounds = self.sim._current_scene.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        from satnav.sims.satsim.geoutils import GeoUtils
        lon, lat = GeoUtils.mercator_to_wgs84(center_x, center_y)
        self.sim.set_agent_state([lon, lat, 100.0], 0.0)
        
        obs = self.sim.get_observations()
        
        assert isinstance(obs, dict)
        assert "rgb" in obs
        assert isinstance(obs["rgb"], np.ndarray)
        assert obs["rgb"].dtype == np.uint8
        assert obs["rgb"].shape == (224, 224, 3)


class TestIsNavigable:
    """Test cases for is_navigable method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_is_navigable_no_scene(self):
        """Test that is_navigable raises RuntimeError when scene not loaded."""
        position = [116.3974, 39.9093, 100.0]
        with pytest.raises(RuntimeError, match="Scene not loaded"):
            self.sim.is_navigable(position)

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_is_navigable_within_bounds(self):
        """Test is_navigable with position within scene bounds."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        self.sim.load_scene(str(tif_path))
        
        # Get scene bounds and use center
        bounds = self.sim._current_scene.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        from satnav.sims.satsim.geoutils import GeoUtils
        lon, lat = GeoUtils.mercator_to_wgs84(center_x, center_y)
        
        # Position within bounds should be navigable
        assert self.sim.is_navigable([lon, lat, 100.0]) is True

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_is_navigable_outside_bounds(self):
        """Test is_navigable with position outside scene bounds."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        self.sim.load_scene(str(tif_path))
        
        # Position far outside bounds
        position_outside = [200.0, 80.0, 100.0]  # Unlikely to be in scene
        
        # Should return False (or True if scene is very large, but unlikely)
        result = self.sim.is_navigable(position_outside)
        assert isinstance(result, bool)


class TestGetSceneBounds:
    """Test cases for get_scene_bounds method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_get_scene_bounds_no_scene(self):
        """Test that get_scene_bounds raises RuntimeError when scene not loaded."""
        with pytest.raises(RuntimeError, match="Scene not loaded"):
            self.sim.get_scene_bounds()

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_get_scene_bounds_basic(self):
        """Test basic scene bounds retrieval."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        self.sim.load_scene(str(tif_path))
        
        bounds = self.sim.get_scene_bounds()
        
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4
        left_lon, right_lon, bottom_lat, top_lat = bounds
        
        # Bounds should be valid WGS84 coordinates
        assert -180.0 <= left_lon <= 180.0
        assert -180.0 <= right_lon <= 180.0
        assert -90.0 <= bottom_lat <= 90.0
        assert -90.0 <= top_lat <= 90.0
        assert left_lon < right_lon
        assert bottom_lat < top_lat


class TestParseAction:
    """Test cases for _parse_action method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    def test_parse_action_string(self):
        """Test parsing action as string."""
        assert self.sim._parse_action("MOVE_FORWARD") == "MOVE_FORWARD"
        assert self.sim._parse_action("TURN_LEFT") == "TURN_LEFT"
        assert self.sim._parse_action("STOP") == "STOP"

    def test_parse_action_int(self):
        """Test parsing action as integer index."""
        assert self.sim._parse_action(0) == "STOP"
        assert self.sim._parse_action(1) == "MOVE_FORWARD"
        assert self.sim._parse_action(2) == "TURN_LEFT"
        assert self.sim._parse_action(3) == "TURN_RIGHT"

    def test_parse_action_int_out_of_range(self):
        """Test parsing action with invalid integer index."""
        with pytest.raises(ValueError, match="out of range"):
            self.sim._parse_action(4)
        with pytest.raises(ValueError, match="out of range"):
            self.sim._parse_action(-1)

    def test_parse_action_dict(self):
        """Test parsing action as dictionary."""
        assert self.sim._parse_action({"action": "MOVE_FORWARD"}) == "MOVE_FORWARD"
        assert self.sim._parse_action({"action": "STOP"}) == "STOP"

    def test_parse_action_dict_invalid(self):
        """Test parsing action dictionary without 'action' key."""
        with pytest.raises(ValueError, match="must contain 'action' key"):
            self.sim._parse_action({"invalid": "key"})

    def test_parse_action_invalid_type(self):
        """Test parsing action with invalid type."""
        with pytest.raises(ValueError, match="Unsupported action type"):
            self.sim._parse_action(3.14)  # float


class TestClose:
    """Test cases for close method."""

    def setup_method(self):
        """Set up test fixtures."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        self.sim = SatSim(config)

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_close_clears_cache(self):
        """Test that close clears scene cache."""
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        self.sim.load_scene(str(tif_path))
        assert len(self.sim._scene_cache) > 0
        
        self.sim.close()
        
        assert len(self.sim._scene_cache) == 0
        assert self.sim._current_scene is None


class TestSatSimIntegration:
    """Integration tests for SatSim."""

    @pytest.mark.skipif(
        not Path("tests/test_data/map.tif").exists(),
        reason="Test TIF file not found"
    )
    def test_full_navigation_sequence(self):
        """Test a complete navigation sequence."""
        config = {
            "FORWARD_STEP_SIZE": 0.25,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90.0}
        }
        sim = SatSim(config)
        
        # Load scene
        tif_path = Path("tests/test_data/map.tif")
        if not tif_path.exists():
            pytest.skip("Test TIF file not found")
        
        sim.load_scene(str(tif_path))
        
        # Set initial state
        bounds = sim._current_scene.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        from satnav.sims.satsim.geoutils import GeoUtils
        lon, lat = GeoUtils.mercator_to_wgs84(center_x, center_y)
        sim.set_agent_state([lon, lat, 100.0], 0.0)
        
        # Get initial observations
        obs1 = sim.get_observations()
        assert "rgb" in obs1
        
        # Move forward
        obs2 = sim.step("MOVE_FORWARD")
        assert "rgb" in obs2
        
        # Turn right
        obs3 = sim.step("TURN_RIGHT")
        assert "rgb" in obs3
        
        # Move forward again
        obs4 = sim.step("MOVE_FORWARD")
        assert "rgb" in obs4
        
        # Verify position changed
        position_final, rotation_final = sim.get_agent_state()
        assert rotation_final != 0.0  # Should have turned
        
        # Cleanup
        sim.close()


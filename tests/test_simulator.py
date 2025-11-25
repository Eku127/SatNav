#!/usr/bin/env python3
"""Tests for simulator wrapper implementation."""

import pytest
import numpy as np
from omegaconf import OmegaConf

from satnav.sims.satsim_wrapper import SatSimWrapper
from satnav.core.simulator import AgentState


class TestSatSimWrapper:
    """Test cases for SatSimWrapper class."""
    
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
        sim = SatSimWrapper(config)
        
        assert sim.forward_step_size == 0.25
        assert sim.turn_angle == 15.0
        assert sim.rgb_width == 224
        assert sim.rgb_height == 224
        assert sim.rgb_hfov == 90.0
        assert sim._agent_state is None
        assert sim._scene_id is None
    
    def test_init_with_dictconfig(self):
        """Test initialization with DictConfig."""
        config_dict = self.create_test_config()
        config = OmegaConf.create({"SIMULATOR": config_dict})
        sim = SatSimWrapper(config.SIMULATOR)
        
        assert sim.forward_step_size == 0.25
        assert sim.turn_angle == 15.0
        assert sim.rgb_width == 224
        assert sim.rgb_height == 224
        assert sim.rgb_hfov == 90.0
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config (uses defaults)."""
        config = {}
        sim = SatSimWrapper(config)
        
        assert sim.forward_step_size == 0.25  # default
        assert sim.turn_angle == 15.0  # default
        assert sim.rgb_width == 224  # default
        assert sim.rgb_height == 224  # default
        assert sim.rgb_hfov == 90.0  # default
    
    def test_reset_not_implemented(self):
        """Test that reset() raises NotImplementedError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        with pytest.raises(NotImplementedError):
            sim.reset("test_scene_001")
    
    def test_step_not_implemented(self):
        """Test that step() raises NotImplementedError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        with pytest.raises(NotImplementedError):
            sim.step("MOVE_FORWARD")
    
    def test_get_agent_state_not_initialized(self):
        """Test that get_agent_state() raises RuntimeError when not initialized."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        with pytest.raises(RuntimeError, match="Agent state not initialized"):
            sim.get_agent_state()
    
    def test_set_agent_state(self):
        """Test setting agent state."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        position = [116.3974, 39.9093, 100.0]
        rotation = 45.0
        
        sim.set_agent_state(position, rotation)
        
        state = sim.get_agent_state()
        assert isinstance(state, AgentState)
        np.testing.assert_array_almost_equal(state.position, position, decimal=5)
        assert state.rotation == 45.0
    
    def test_set_agent_state_normalize_rotation(self):
        """Test that rotation is normalized to [0, 360)."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        position = [116.3974, 39.9093, 100.0]
        
        # Test negative rotation
        sim.set_agent_state(position, -45.0)
        state = sim.get_agent_state()
        assert state.rotation == 315.0  # -45 + 360
        
        # Test rotation > 360
        sim.set_agent_state(position, 450.0)
        state = sim.get_agent_state()
        assert state.rotation == 90.0  # 450 % 360
    
    def test_set_agent_state_invalid_position(self):
        """Test that invalid position raises ValueError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        # Too few elements
        with pytest.raises(ValueError, match="must have 3 elements"):
            sim.set_agent_state([116.3974, 39.9093], 0.0)
        
        # Too many elements
        with pytest.raises(ValueError, match="must have 3 elements"):
            sim.set_agent_state([116.3974, 39.9093, 100.0, 0.0], 0.0)
    
    def test_get_observations_not_initialized(self):
        """Test that get_observations() raises RuntimeError when not initialized."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        with pytest.raises(RuntimeError, match="Agent state not initialized"):
            sim.get_observations()
    
    def test_get_observations_not_implemented(self):
        """Test that get_observations() raises NotImplementedError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        sim.set_agent_state([116.3974, 39.9093, 100.0], 0.0)
        
        with pytest.raises(NotImplementedError):
            sim.get_observations()
    
    def test_geodesic_distance(self):
        """Test geodesic distance calculation."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        # Test with known coordinates (Beijing to Shanghai approximate)
        pos_a = [116.3974, 39.9093, 100.0]  # Beijing
        pos_b = [121.4737, 31.2304, 50.0]   # Shanghai
        
        distance = sim.geodesic_distance(pos_a, pos_b)
        
        # Distance should be approximately 1067 km (with altitude difference)
        assert distance > 1000000  # > 1000 km
        assert distance < 1100000  # < 1100 km
    
    def test_geodesic_distance_same_position(self):
        """Test geodesic distance with same position."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        pos = [116.3974, 39.9093, 100.0]
        distance = sim.geodesic_distance(pos, pos)
        
        # Should be approximately the altitude difference (0 in this case)
        assert abs(distance) < 1.0  # Very small
    
    def test_is_navigable_not_implemented(self):
        """Test that is_navigable() raises NotImplementedError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        position = [116.3974, 39.9093, 100.0]
        
        with pytest.raises(NotImplementedError):
            sim.is_navigable(position)
    
    def test_sample_navigable_point_no_scene(self):
        """Test that sample_navigable_point() raises RuntimeError when no scene loaded."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        with pytest.raises(RuntimeError, match="Scene not loaded"):
            sim.sample_navigable_point()
    
    def test_sample_navigable_point_not_implemented(self):
        """Test that sample_navigable_point() raises NotImplementedError."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        # Set scene_id (simulating reset without actually calling it)
        sim._scene_id = "test_scene_001"
        
        with pytest.raises(NotImplementedError):
            sim.sample_navigable_point()
    
    def test_sensor_suite(self):
        """Test sensor_suite property."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        sensor_suite = sim.sensor_suite
        
        assert "rgb" in sensor_suite
        assert sensor_suite["rgb"]["width"] == 224
        assert sensor_suite["rgb"]["height"] == 224
        assert sensor_suite["rgb"]["hfov"] == 90.0
    
    def test_action_space(self):
        """Test action_space property."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        action_space = sim.action_space
        
        assert isinstance(action_space, list)
        assert "STOP" in action_space
        assert "MOVE_FORWARD" in action_space
        assert "TURN_LEFT" in action_space
        assert "TURN_RIGHT" in action_space
        assert len(action_space) == 4
    
    def test_scene_id_property(self):
        """Test scene_id property."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        # Initially None
        assert sim.scene_id is None
        
        # Set scene_id (simulating reset)
        sim._scene_id = "test_scene_001"
        assert sim.scene_id == "test_scene_001"
    
    def test_geodesic_distance_with_altitude_difference(self):
        """Test that geodesic_distance considers altitude difference."""
        config = self.create_test_config()
        sim = SatSimWrapper(config)
        
        # Same longitude/latitude, different altitude
        pos_a = [116.3974, 39.9093, 100.0]
        pos_b = [116.3974, 39.9093, 200.0]
        
        distance = sim.geodesic_distance(pos_a, pos_b)
        
        # Should be approximately 100 meters (altitude difference)
        assert abs(distance - 100.0) < 1.0


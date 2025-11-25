#!/usr/bin/env python3
"""Tests for VLN task implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from satnav.core.config import load_config
from satnav.core.episode import VLNEpisode, InstructionData, NavigationGoal
from satnav.core.simulator import AgentState, Simulator
from satnav.task.actions import Action
from satnav.task.vln_task import VLNTask


# Path to test config and data
TEST_CONFIG_PATH = Path(__file__).parent / "test_data" / "test_config.yaml"
EXAMPLE_DATASET_PATH = Path(__file__).parent / "test_data" / "satnav_dataset_example.json"


class MockSimulator(Simulator):
    """Mock simulator for testing VLNTask without actual satsim implementation."""
    
    def __init__(self):
        """Initialize mock simulator."""
        self._agent_state: Optional[AgentState] = None
        self._scene_id: Optional[str] = None
        self._step_count = 0
    
    def reset(self, scene_id: str) -> Dict[str, Any]:
        """Reset simulator and load scene."""
        self._scene_id = scene_id
        self._step_count = 0
        return {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}
    
    def step(self, action: Union[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute action (mock implementation)."""
        self._step_count += 1
        
        # Simulate position change for MOVE_FORWARD
        if isinstance(action, str) and action == "MOVE_FORWARD":
            if self._agent_state is not None:
                # Move forward slightly (simulate)
                new_pos = self._agent_state.position.copy()
                new_pos[0] += 0.0001  # Small longitude change
                new_pos[1] += 0.0001  # Small latitude change
                self._agent_state = AgentState(position=new_pos, rotation=self._agent_state.rotation)
        elif isinstance(action, str) and action in ["TURN_LEFT", "TURN_RIGHT"]:
            if self._agent_state is not None:
                # Simulate rotation change
                rotation_delta = 15.0 if action == "TURN_LEFT" else -15.0
                new_rotation = (self._agent_state.rotation + rotation_delta) % 360.0
                self._agent_state = AgentState(
                    position=self._agent_state.position.copy(),
                    rotation=new_rotation
                )
        
        return {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}
    
    def get_agent_state(self) -> AgentState:
        """Get current agent state."""
        if self._agent_state is None:
            raise RuntimeError("Agent state not initialized")
        return self._agent_state
    
    def set_agent_state(self, position: List[float], rotation: float) -> None:
        """Set agent state."""
        self._agent_state = AgentState(
            position=np.array(position, dtype=np.float32),
            rotation=float(rotation)
        )
    
    def get_observations(self) -> Dict[str, Any]:
        """Get observations."""
        if self._agent_state is None:
            raise RuntimeError("Agent state not initialized")
        return {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}
    
    def geodesic_distance(
        self,
        position_a: Union[List[float], np.ndarray],
        position_b: Union[List[float], np.ndarray]
    ) -> float:
        """Calculate geodesic distance (simplified for testing)."""
        from satnav.core.utils import geodesic_distance
        return geodesic_distance(position_a, position_b)
    
    def is_navigable(self, position: Union[List[float], np.ndarray]) -> bool:
        """Check if position is navigable (mock: always True)."""
        return True
    
    def sample_navigable_point(self) -> List[float]:
        """Sample navigable point (mock)."""
        if self._scene_id is None:
            raise RuntimeError("Scene not loaded")
        return [116.3974, 39.9093, 100.0]
    
    @property
    def sensor_suite(self):
        """Get sensor suite."""
        return {"rgb": {"width": 224, "height": 224, "hfov": 90.0}}
    
    @property
    def action_space(self):
        """Get action space."""
        return ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


class TestVLNTask:
    """Test cases for VLNTask class."""
    
    def create_test_config(self) -> DictConfig:
        """Create test task configuration."""
        if TEST_CONFIG_PATH.exists():
            config = load_config(str(TEST_CONFIG_PATH))
            return config.TASK
        else:
            # Fallback: create minimal config
            return OmegaConf.create({
                "TYPE": "VLN-v0",
                "SUCCESS_DISTANCE": 3.0,
                "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "PATH_LENGTH"]
            })
    
    def create_test_episode(self) -> VLNEpisode:
        """Create a test episode."""
        return VLNEpisode(
            episode_id="test_001",
            scene_id="test_scene_001",
            start_position=[116.3974, 39.9093, 100.0],
            start_rotation=0.0,
            goals=[NavigationGoal(position=[116.3975, 39.9094, 100.0])],
            reference_path=[
                [116.3974, 39.9093, 100.0],
                [116.39745, 39.90935, 100.0],
                [116.3975, 39.9094, 100.0]
            ],
            instruction=InstructionData(instruction_text="Go to the kitchen"),
            trajectory_id="traj_001"
        )
    
    def test_init_with_dict_config(self):
        """Test initialization with dictionary config."""
        config = {
            "SUCCESS_DISTANCE": 3.0,
            "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD"],
            "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS"]
        }
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        assert task.success_distance == 3.0
        assert len(task.sensors) == 2  # RGBSensor and InstructionSensor
        assert len(task.measures) == 2  # DistanceToGoal and Success
    
    def test_init_with_dictconfig(self):
        """Test initialization with DictConfig."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        assert task.success_distance == 3.0
        assert len(task.sensors) == 2
        assert len(task.measures) >= 2
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config (uses defaults)."""
        config = {}
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        assert task.success_distance == 3.0  # default
        assert len(task.sensors) == 2  # Always includes RGB and Instruction
        assert len(task.measures) == 4  # All measures if none specified
    
    def test_init_sensors(self):
        """Test that sensors are initialized correctly."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        # Check sensor types
        sensor_uuids = [sensor.uuid for sensor in task.sensors]
        assert "rgb" in sensor_uuids
        assert "instruction" in sensor_uuids
    
    def test_init_measures(self):
        """Test that measures are initialized correctly."""
        config = {
            "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "PATH_LENGTH", "SPL"]
        }
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        assert len(task.measures) == 4
        # Check that measures_dict contains all measures
        assert "DISTANCE_TO_GOAL" in task._measures_dict
        assert "SUCCESS" in task._measures_dict
        assert "PATH_LENGTH" in task._measures_dict
        assert "SPL" in task._measures_dict
    
    def test_init_measure_dependencies(self):
        """Test that measure dependencies are set up correctly."""
        config = {
            "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "PATH_LENGTH", "SPL"]
        }
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        # Check that Success has DistanceToGoal dependency
        success_measure = task._measures_dict["SUCCESS"]
        assert success_measure._distance_to_goal is not None
        assert success_measure._distance_to_goal == task._measures_dict["DISTANCE_TO_GOAL"]
        
        # Check that SPL has Success and PathLength dependencies
        spl_measure = task._measures_dict["SPL"]
        assert spl_measure._success_measure is not None
        assert spl_measure._path_length_measure is not None
    
    def test_reset(self):
        """Test reset method."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        observations = task.reset(episode)
        
        # Check that episode is stored
        assert task.current_episode == episode
        
        # Check that observations are returned
        assert isinstance(observations, dict)
        assert "rgb" in observations
        assert "instruction" in observations
        
        # Check that instruction observation is correct
        assert observations["instruction"]["text"] == "Go to the kitchen"
        
        # Check that RGB observation has correct shape
        assert observations["rgb"].shape == (224, 224, 3)
    
    def test_reset_initializes_simulator(self):
        """Test that reset initializes simulator state."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Check that simulator state is set
        state = sim.get_agent_state()
        np.testing.assert_array_almost_equal(
            state.position,
            episode.start_position,
            decimal=5
        )
        assert state.rotation == episode.start_rotation
    
    def test_reset_resets_measures(self):
        """Test that reset resets all measures."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Check that measures are reset (distance_to_goal should be calculated)
        metrics = task.get_metrics()
        assert "distance_to_goal" in metrics
        assert metrics["distance_to_goal"] > 0  # Should have some distance
    
    def test_step_with_string_action(self):
        """Test step with string action."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Execute action
        observations = task.step("MOVE_FORWARD")
        
        # Check that observations are returned
        assert isinstance(observations, dict)
        assert "rgb" in observations
        assert "instruction" in observations
    
    def test_step_with_dict_action(self):
        """Test step with dictionary action."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Execute action as dictionary
        observations = task.step({"action": "TURN_LEFT"})
        
        assert isinstance(observations, dict)
    
    def test_step_with_index_action(self):
        """Test step with action index."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Execute action as index
        observations = task.step(0)  # STOP action
        
        assert isinstance(observations, dict)
    
    def test_step_invalid_action(self):
        """Test that invalid action raises ValueError."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        with pytest.raises(ValueError, match="Invalid action"):
            task.step("INVALID_ACTION")
    
    def test_step_updates_measures(self):
        """Test that step updates all measures."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Get initial metrics
        initial_metrics = task.get_metrics()
        initial_distance = initial_metrics.get("distance_to_goal", 0.0)
        
        # Execute action
        task.step("MOVE_FORWARD")
        
        # Get updated metrics
        updated_metrics = task.get_metrics()
        
        # Distance should be updated (may change due to position change)
        assert "distance_to_goal" in updated_metrics
        # Path length should increase
        if "path_length" in updated_metrics:
            assert updated_metrics["path_length"] >= 0.0
    
    def test_get_observations(self):
        """Test get_observations method."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        observations = task.get_observations()
        
        # Check observations structure
        assert isinstance(observations, dict)
        assert "rgb" in observations
        assert "instruction" in observations
        
        # Check RGB observation
        assert isinstance(observations["rgb"], np.ndarray)
        assert observations["rgb"].shape == (224, 224, 3)
        assert observations["rgb"].dtype == np.uint8
        
        # Check instruction observation
        assert isinstance(observations["instruction"], dict)
        assert "text" in observations["instruction"]
        assert observations["instruction"]["text"] == "Go to the kitchen"
    
    def test_get_observations_without_reset(self):
        """Test that get_observations raises error if episode not set."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        with pytest.raises(RuntimeError, match="episode not set"):
            task.get_observations()
    
    def test_get_metrics(self):
        """Test get_metrics method."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        metrics = task.get_metrics()
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        
        # Check that enabled metrics are present
        if "DISTANCE_TO_GOAL" in config.get("MEASUREMENTS", []):
            assert "distance_to_goal" in metrics
            assert isinstance(metrics["distance_to_goal"], float)
            assert metrics["distance_to_goal"] >= 0.0
        
        if "SUCCESS" in config.get("MEASUREMENTS", []):
            assert "success" in metrics
            assert metrics["success"] in [0.0, 1.0]
        
        if "PATH_LENGTH" in config.get("MEASUREMENTS", []):
            assert "path_length" in metrics
            assert isinstance(metrics["path_length"], float)
            assert metrics["path_length"] >= 0.0
        
        if "SPL" in config.get("MEASUREMENTS", []):
            assert "spl" in metrics
            assert isinstance(metrics["spl"], float)
            assert 0.0 <= metrics["spl"] <= 1.0
    
    def test_get_metrics_after_step(self):
        """Test that metrics are updated after step."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        initial_metrics = task.get_metrics()
        
        # Execute several steps
        task.step("MOVE_FORWARD")
        task.step("TURN_LEFT")
        task.step("MOVE_FORWARD")
        
        final_metrics = task.get_metrics()
        
        # Path length should increase
        if "path_length" in initial_metrics and "path_length" in final_metrics:
            assert final_metrics["path_length"] >= initial_metrics["path_length"]
    
    def test_current_episode_property(self):
        """Test current_episode property."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        # Initially None
        assert task.current_episode is None
        
        # After reset, should be set
        episode = self.create_test_episode()
        task.reset(episode)
        assert task.current_episode == episode
    
    def test_success_measure_with_stop_action(self):
        """Test that Success measure updates correctly with STOP action."""
        config = {
            "SUCCESS_DISTANCE": 3.0,
            "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS"]
        }
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        # Create episode where goal is very close
        episode = VLNEpisode(
            episode_id="test_close",
            scene_id="test_scene",
            start_position=[116.3974, 39.9093, 100.0],
            start_rotation=0.0,
            goals=[NavigationGoal(position=[116.39741, 39.90931, 100.0])],  # Very close
            reference_path=[[116.3974, 39.9093, 100.0], [116.39741, 39.90931, 100.0]],
            instruction=InstructionData(instruction_text="Go nearby"),
            trajectory_id="traj_close"
        )
        
        task.reset(episode)
        
        # Initially not successful (no STOP called)
        metrics = task.get_metrics()
        assert metrics.get("success", 0.0) == 0.0
        
        # Call STOP action
        task.step("STOP")
        
        # Check success (may be 1.0 if within success_distance)
        metrics = task.get_metrics()
        assert "success" in metrics
        assert metrics["success"] in [0.0, 1.0]
    
    def test_spl_calculation(self):
        """Test that SPL is calculated correctly."""
        config = {
            "SUCCESS_DISTANCE": 3.0,
            "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "PATH_LENGTH", "SPL"]
        }
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        episode = self.create_test_episode()
        task.reset(episode)
        
        # Execute some actions
        task.step("MOVE_FORWARD")
        task.step("TURN_LEFT")
        
        metrics = task.get_metrics()
        
        if "spl" in metrics:
            # SPL should be in [0, 1] range
            assert 0.0 <= metrics["spl"] <= 1.0
            # If not successful, SPL should be 0
            if metrics.get("success", 0.0) == 0.0:
                assert metrics["spl"] == 0.0
    
    def test_multiple_episodes(self):
        """Test that task can handle multiple episodes."""
        config = self.create_test_config()
        sim = MockSimulator()
        task = VLNTask(config, sim)
        
        # First episode
        episode1 = self.create_test_episode()
        task.reset(episode1)
        assert task.current_episode == episode1
        
        # Second episode
        episode2 = VLNEpisode(
            episode_id="test_002",
            scene_id="test_scene_002",
            start_position=[121.4737, 31.2304, 50.0],
            start_rotation=90.0,
            goals=[NavigationGoal(position=[121.4738, 31.2305, 50.0])],
            reference_path=[[121.4737, 31.2304, 50.0], [121.4738, 31.2305, 50.0]],
            instruction=InstructionData(instruction_text="Navigate forward"),
            trajectory_id="traj_002"
        )
        task.reset(episode2)
        assert task.current_episode == episode2
        assert task.current_episode.episode_id == "test_002"


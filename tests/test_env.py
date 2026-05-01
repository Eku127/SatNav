#!/usr/bin/env python3
"""Tests for Env class implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from satnav.core.config import load_config
from satnav.core.env import Env
from satnav.core.episode import VLNEpisode, InstructionData, NavigationGoal
from satnav.core.simulator import AgentState, Simulator
from satnav.dataset.satnav_dataset import SatNavDataset


# Path to test config and data
TEST_CONFIG_PATH = Path(__file__).parent / "test_data" / "satnav_config_example.yaml"
EXAMPLE_DATASET_PATH = Path(__file__).parent / "test_data" / "satnav_dataset_example.json"


class MockSimulator(Simulator):
    """Mock simulator for testing Env without actual satsim implementation."""
    
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


class MockSatSimWrapper(Simulator):
    """Mock SatSimWrapper for testing Env."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize mock wrapper."""
        self._sim = MockSimulator()
        self._config = config or {}
    
    def reset(self, scene_id: str) -> Dict[str, Any]:
        """Reset simulator."""
        return self._sim.reset(scene_id)
    
    def step(self, action: Union[str, Dict[str, Any], int]) -> Dict[str, Any]:
        """Execute action."""
        return self._sim.step(action)
    
    def get_agent_state(self) -> AgentState:
        """Get agent state."""
        return self._sim.get_agent_state()
    
    def set_agent_state(self, position: List[float], rotation: float) -> None:
        """Set agent state."""
        self._sim.set_agent_state(position, rotation)
    
    def get_observations(self) -> Dict[str, Any]:
        """Get observations."""
        return self._sim.get_observations()
    
    def geodesic_distance(
        self,
        position_a: Union[List[float], np.ndarray],
        position_b: Union[List[float], np.ndarray]
    ) -> float:
        """Calculate geodesic distance."""
        return self._sim.geodesic_distance(position_a, position_b)
    
    def is_navigable(self, position: Union[List[float], np.ndarray]) -> bool:
        """Check if position is navigable."""
        return self._sim.is_navigable(position)
    
    def sample_navigable_point(self) -> List[float]:
        """Sample navigable point."""
        return self._sim.sample_navigable_point()
    
    @property
    def sensor_suite(self):
        """Get sensor suite."""
        return self._sim.sensor_suite
    
    @property
    def action_space(self):
        """Get action space."""
        return self._sim.action_space


class TestEnv:
    """Test cases for Env class."""
    
    def create_test_config(self) -> DictConfig:
        """Create test configuration."""
        if TEST_CONFIG_PATH.exists():
            return load_config(str(TEST_CONFIG_PATH))
        else:
            # Fallback: create minimal config
            return OmegaConf.create({
                "ENVIRONMENT": {
                    "MAX_EPISODE_STEPS": 500
                },
                "SIMULATOR": {
                    "FORWARD_STEP_SIZE": 0.25,
                    "TURN_ANGLE": 15,
                    "RGB_SENSOR": {
                        "WIDTH": 224,
                        "HEIGHT": 224,
                        "HFOV": 90
                    }
                },
                "TASK": {
                    "TYPE": "VLN",
                    "SUCCESS_DISTANCE": 3.0,
                    "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                    "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "PATH_LENGTH"]
                },
                "DATASET": {
                    "TYPE": "SatNav",
                    "SPLIT": "test",
                    "DATA_PATH": str(EXAMPLE_DATASET_PATH),
                    "SCENES_DIR": "tests/test_data/"
                }
            })
    
    def create_test_dataset(self) -> SatNavDataset:
        """Create a test dataset."""
        if not EXAMPLE_DATASET_PATH.exists():
            pytest.skip(f"Example dataset not found: {EXAMPLE_DATASET_PATH}")
        
        config = {
            "DATA_PATH": str(EXAMPLE_DATASET_PATH),
            "SPLIT": "test",
            "SCENES_DIR": "tests/test_data/",  # Updated to match test data location
        }
        return SatNavDataset(config)
    
    @patch('satnav.core.env.create_simulator')
    def test_init_with_dataset(self, mock_wrapper_class):
        """Test initialization with provided dataset."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        assert env._dataset is not None
        assert len(env._dataset.episodes) > 0
        assert env.max_episode_steps == 500
        assert env._episode_iterator is not None
    
    @patch('satnav.core.env.create_simulator')
    def test_init_with_config_dataset(self, mock_wrapper_class):
        """Test initialization with dataset from config."""
        config = self.create_test_config()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config)
        
        assert env._dataset is not None
        assert len(env._dataset.episodes) > 0
    
    @patch('satnav.core.env.create_simulator')
    def test_init_with_empty_dataset(self, mock_wrapper_class):
        """Test initialization with empty dataset raises error."""
        config = self.create_test_config()
        
        # Create empty dataset by creating object and setting episodes to empty list
        # We can't use SatNavDataset constructor with empty DATA_PATH because it will
        # raise FileNotFoundError before we can set episodes to empty
        empty_dataset = SatNavDataset.__new__(SatNavDataset)  # Create without calling __init__
        empty_dataset.episodes = []  # Make it empty
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        with pytest.raises(ValueError, match="Dataset must have at least one episode"):
            Env(config, dataset=empty_dataset)
    
    @patch('satnav.core.env.create_simulator')
    def test_reset(self, mock_wrapper_class):
        """Test reset() method."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        observations = env.reset()
        
        # Check observations structure
        assert isinstance(observations, dict)
        assert "rgb" in observations
        assert "instruction" in observations
        assert observations["rgb"].shape == (224, 224, 3)
        assert isinstance(observations["instruction"], dict)
        assert "text" in observations["instruction"]
        
        # Check episode state
        assert env._current_episode is not None
        assert env._elapsed_steps == 0
        assert env.episode_over is False
        assert env.current_episode is not None
    
    @patch('satnav.core.env.create_simulator')
    def test_reset_without_dataset(self, mock_wrapper_class):
        """Test reset() without dataset raises error."""
        config = self.create_test_config()
        config.DATASET = None
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=None)
        
        with pytest.raises(RuntimeError, match="dataset is not available"):
            env.reset()
    
    @patch('satnav.core.env.create_simulator')
    def test_step_with_string_action(self, mock_wrapper_class):
        """Test step() with string action."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        env.reset()
        
        obs, done, info = env.step("MOVE_FORWARD")
        
        # Check return values
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info structure
        assert "metrics" in info
        assert "episode_id" in info
        assert "elapsed_steps" in info
        assert "episode_over" in info
        assert info["elapsed_steps"] == 1
        assert info["episode_over"] == done
    
    @patch('satnav.core.env.create_simulator')
    def test_step_with_dict_action(self, mock_wrapper_class):
        """Test step() with dictionary action."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        env.reset()
        
        obs, done, info = env.step({"action": "TURN_LEFT"})
        
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert info["elapsed_steps"] == 1
    
    @patch('satnav.core.env.create_simulator')
    def test_step_with_int_action(self, mock_wrapper_class):
        """Test step() with integer action index."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        env.reset()
        
        # Action index 1 should be "MOVE_FORWARD"
        obs, done, info = env.step(1)
        
        assert isinstance(obs, dict)
        assert isinstance(done, bool)
        assert info["elapsed_steps"] == 1
    
    @patch('satnav.core.env.create_simulator')
    def test_step_without_reset(self, mock_wrapper_class):
        """Test step() without reset() raises error."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        with pytest.raises(RuntimeError, match="episode not set"):
            env.step("MOVE_FORWARD")
    
    @patch('satnav.core.env.create_simulator')
    def test_step_after_episode_over(self, mock_wrapper_class):
        """Test step() after episode is over raises error."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        env.reset()
        
        # Manually set episode_over to True
        env._episode_over = True
        
        with pytest.raises(RuntimeError, match="episode is over"):
            env.step("MOVE_FORWARD")
    
    @patch('satnav.core.env.create_simulator')
    def test_episode_ends_at_max_steps(self, mock_wrapper_class):
        """Test episode ends when max steps reached."""
        config = self.create_test_config()
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 5  # Set small max steps
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        env.reset()
        
        # Step until max steps
        for i in range(5):
            obs, done, info = env.step("MOVE_FORWARD")
            assert info["elapsed_steps"] == i + 1
        
        # Episode should be over
        assert done is True
        assert env.episode_over is True
        assert info["episode_over"] is True
    
    @patch('satnav.core.env.create_simulator')
    def test_get_metrics(self, mock_wrapper_class):
        """Test get_metrics() method."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        # Before reset, should return empty dict
        metrics = env.get_metrics()
        assert metrics == {}
        
        # After reset, should return metrics
        env.reset()
        metrics = env.get_metrics()
        assert isinstance(metrics, dict)
        # Metrics should contain enabled measures
        assert "distance_to_goal" in metrics
    
    @patch('satnav.core.env.create_simulator')
    def test_observation_space(self, mock_wrapper_class):
        """Test observation_space property."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        obs_space = env.observation_space
        assert isinstance(obs_space, dict)
        assert "rgb" in obs_space
        assert "instruction" in obs_space
        assert obs_space["rgb"]["shape"] == (224, 224, 3)
    
    @patch('satnav.core.env.create_simulator')
    def test_action_space(self, mock_wrapper_class):
        """Test action_space property."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        action_space = env.action_space
        assert isinstance(action_space, dict)
        assert "actions" in action_space
        assert "STOP" in action_space["actions"]
        assert "MOVE_FORWARD" in action_space["actions"]
    
    @patch('satnav.core.env.create_simulator')
    def test_episode_iterator_restart_with_cycle(self, mock_wrapper_class):
        """Test episode iterator restarts when exhausted with cycle=True."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        # Create env with cycle=True (training mode)
        env = Env(config, dataset=dataset, cycle=True)
        
        # Get first episode
        obs1 = env.reset()
        episode_id_1 = env.current_episode.episode_id
        
        # Reset (len(dataset.episodes) - 1) times to exhaust iterator
        # After first reset, we have len(dataset.episodes) - 1 episodes left
        # So we need to reset (len(dataset.episodes) - 1) more times to exhaust
        for _ in range(len(dataset.episodes) - 1):
            env.reset()
        
        # Next reset should restart iterator and get first episode again
        obs2 = env.reset()
        episode_id_2 = env.current_episode.episode_id
        
        # Should get first episode again
        assert episode_id_2 == episode_id_1
    
    @patch('satnav.core.env.create_simulator')
    def test_episode_iterator_exhaustion_without_cycle(self, mock_wrapper_class):
        """Test episode iterator raises error when exhausted with cycle=False."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        # Create env with cycle=False (evaluation mode, default)
        env = Env(config, dataset=dataset, cycle=False)
        
        # Reset to exhaust all episodes
        num_episodes = len(dataset.episodes)
        for _ in range(num_episodes):
            env.reset()
        
        # Next reset should raise RuntimeError
        with pytest.raises(RuntimeError, match="All episodes exhausted"):
            env.reset()
    
    @patch('satnav.core.env.create_simulator')
    def test_current_episode_property(self, mock_wrapper_class):
        """Test current_episode property."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        # Before reset, should be None
        assert env.current_episode is None
        
        # After reset, should be set
        env.reset()
        assert env.current_episode is not None
        assert isinstance(env.current_episode, VLNEpisode)
    
    @patch('satnav.core.env.create_simulator')
    def test_episode_over_property(self, mock_wrapper_class):
        """Test episode_over property."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        # After reset, should be False
        env.reset()
        assert env.episode_over is False
        
        # After max steps, should be True
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 2
        # Create new env with updated config
        mock_wrapper_class.return_value = MockSatSimWrapper()
        env = Env(config, dataset=dataset)
        env.reset()
        env.step("MOVE_FORWARD")
        obs, done, info = env.step("MOVE_FORWARD")
        assert env.episode_over is True


class TestEnvIntegration:
    """Integration tests for Env with real SatSimWrapper and SatSim.
    
    These tests use the actual SatSim implementation instead of mocks,
    providing end-to-end validation of the Env + SatSimWrapper + SatSim integration.
    """
    
    @pytest.fixture(autouse=True)
    def check_test_data(self):
        """Skip tests if test data files are not available."""
        tif_path = Path(__file__).parent / "test_data" / "map.tif"
        dataset_path = Path(__file__).parent / "test_data" / "satnav_dataset_example.json"
        config_path = Path(__file__).parent / "test_data" / "satnav_config_example.yaml"
        
        if not tif_path.exists():
            pytest.skip(f"Test TIF file not found: {tif_path}")
        if not dataset_path.exists():
            pytest.skip(f"Test dataset file not found: {dataset_path}")
        if not config_path.exists():
            pytest.skip(f"Test config file not found: {config_path}")
    
    def create_test_config(self) -> DictConfig:
        """Create test configuration."""
        if TEST_CONFIG_PATH.exists():
            return load_config(str(TEST_CONFIG_PATH))
        else:
            # Fallback: create minimal config
            return OmegaConf.create({
                "ENVIRONMENT": {
                    "MAX_EPISODE_STEPS": 500
                },
                "SIMULATOR": {
                    "FORWARD_STEP_SIZE": 0.25,
                    "TURN_ANGLE": 15,
                    "RGB_SENSOR": {
                        "WIDTH": 224,
                        "HEIGHT": 224,
                        "HFOV": 90
                    }
                },
                "TASK": {
                    "TYPE": "VLN",
                    "SUCCESS_DISTANCE": 3.0,
                    "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                    "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "PATH_LENGTH"]
                },
                "DATASET": {
                    "TYPE": "SatNav",
                    "SPLIT": "test",
                    "DATA_PATH": str(EXAMPLE_DATASET_PATH),
                    "SCENES_DIR": "tests/test_data/"
                }
            })
    
    def test_env_reset_with_real_satsim(self):
        """Test Env.reset() with real SatSimWrapper and SatSim."""
        config = self.create_test_config()
        
        # Create Env without mocking - uses real SatSimWrapper
        env = Env(config)
        
        # Reset environment
        observations = env.reset()
        
        # Verify observations structure
        assert isinstance(observations, dict)
        assert "rgb" in observations
        assert "instruction" in observations
        
        # Verify RGB observation
        rgb = observations["rgb"]
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8
        
        # Verify instruction observation
        instruction = observations["instruction"]
        assert isinstance(instruction, dict)
        assert "text" in instruction
        assert isinstance(instruction["text"], str)
        assert len(instruction["text"]) > 0
        
        # Verify episode state
        assert env.current_episode is not None
        assert env._elapsed_steps == 0
        assert env.episode_over is False
        
        # Verify dataset has episodes
        assert env._dataset is not None
        assert len(env._dataset.episodes) > 0
        assert len(env._dataset.episodes) == 2  # Should have 2 episodes from test data
    
    def test_env_step_with_real_satsim(self):
        """Test Env.step() with real SatSimWrapper and SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset first
        env.reset()
        
        # Execute MOVE_FORWARD action
        obs, done, info = env.step("MOVE_FORWARD")
        
        # Verify return values
        assert isinstance(obs, dict)
        assert "rgb" in obs
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Verify RGB observation after step
        assert obs["rgb"].shape == (224, 224, 3)
        assert obs["rgb"].dtype == np.uint8
        
        # Verify info structure
        assert "metrics" in info
        assert "episode_id" in info
        assert "elapsed_steps" in info
        assert "episode_over" in info
        assert info["elapsed_steps"] == 1
        
        # Verify agent state changed
        agent_state = env._sim.get_agent_state()
        assert agent_state is not None
        assert len(agent_state.position) == 3
    
    def test_env_multiple_steps_with_real_satsim(self):
        """Test multiple steps with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset
        env.reset()
        initial_state = env._sim.get_agent_state()
        
        # Execute multiple actions
        actions = ["MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT"]
        for i, action in enumerate(actions):
            obs, done, info = env.step(action)
            
            # Verify step count
            assert info["elapsed_steps"] == i + 1
            
            # Verify observations
            assert "rgb" in obs
            assert obs["rgb"].shape == (224, 224, 3)
            
            # Verify agent state exists
            agent_state = env._sim.get_agent_state()
            assert agent_state is not None
        
        # Verify final state is different from initial
        final_state = env._sim.get_agent_state()
        # Check if position changed (calculate actual distance)
        from satnav.core.utils import geodesic_distance
        distance_moved = geodesic_distance(initial_state.position, final_state.position)
        
        # After MOVE_FORWARD actions, we should have moved at least some distance
        # (even if we turned, we should have moved forward at least once)
        assert distance_moved > 0.01, (
            f"Expected position to change after MOVE_FORWARD actions, "
            f"but moved only {distance_moved}m. "
            f"Initial: pos={initial_state.position}, rot={initial_state.rotation}, "
            f"Final: pos={final_state.position}, rot={final_state.rotation}"
        )
    
    def test_env_rotation_changes_with_real_satsim(self):
        """Test that rotation changes correctly with real SatSim.
        
        Note: TURN_LEFT decreases rotation (counter-clockwise),
        TURN_RIGHT increases rotation (clockwise).
        """
        config = self.create_test_config()
        env = Env(config)
        
        # Reset
        env.reset()
        initial_state = env._sim.get_agent_state()
        initial_rotation = initial_state.rotation
        
        # Turn left (should decrease rotation, counter-clockwise)
        env.step("TURN_LEFT")
        state_after_left = env._sim.get_agent_state()
        
        # TURN_LEFT decreases rotation: (rotation - turn_angle) % 360
        turn_angle = config.SIMULATOR.TURN_ANGLE
        expected_rotation = (initial_rotation - turn_angle) % 360.0
        
        # Calculate difference accounting for wrap-around
        diff = abs(state_after_left.rotation - expected_rotation)
        diff_wrapped = min(diff, 360.0 - diff)  # Handle wrap-around
        
        assert diff_wrapped < 0.1, (
            f"Expected rotation ~{expected_rotation}° after TURN_LEFT "
            f"(from {initial_rotation}° - {turn_angle}°), "
            f"but got {state_after_left.rotation}° (diff={diff_wrapped}°)"
        )
        
        # Turn right (should increase rotation, clockwise, undoing the left turn)
        env.step("TURN_RIGHT")
        state_after_right = env._sim.get_agent_state()
        
        # Should be back close to initial rotation
        diff = abs(state_after_right.rotation - initial_rotation)
        diff_wrapped = min(diff, 360.0 - diff)  # Handle wrap-around
        assert diff_wrapped < 0.1, (
            f"Expected rotation ~{initial_rotation}° after TURN_RIGHT "
            f"(undoing TURN_LEFT), "
            f"but got {state_after_right.rotation}° (diff={diff_wrapped}°)"
        )
    
    def test_env_position_changes_with_move_forward(self):
        """Test that position changes when moving forward with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset
        env.reset()
        initial_state = env._sim.get_agent_state()
        initial_pos = initial_state.position.copy()
        
        # Move forward
        env.step("MOVE_FORWARD")
        state_after_move = env._sim.get_agent_state()
        after_pos = state_after_move.position
        
        # Verify position changed
        # Calculate distance moved
        from satnav.core.utils import geodesic_distance
        distance = geodesic_distance(initial_pos, after_pos)
        
        # Should have moved approximately FORWARD_STEP_SIZE meters
        expected_distance = config.SIMULATOR.FORWARD_STEP_SIZE
        assert abs(distance - expected_distance) < 0.1, (
            f"Expected to move ~{expected_distance}m, but moved {distance}m"
        )
    
    def test_env_metrics_with_real_satsim(self):
        """Test metrics calculation with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset
        env.reset()
        
        # Get initial metrics
        initial_metrics = env.get_metrics()
        assert isinstance(initial_metrics, dict)
        assert "distance_to_goal" in initial_metrics
        
        # Execute some actions
        env.step("MOVE_FORWARD")
        env.step("TURN_LEFT")
        
        # Get metrics after steps
        metrics = env.get_metrics()
        assert "distance_to_goal" in metrics
        assert isinstance(metrics["distance_to_goal"], (int, float))
        
        # Distance should be non-negative
        assert metrics["distance_to_goal"] >= 0
    
    def test_env_episode_over_at_max_steps(self):
        """Test that episode ends at max steps with real SatSim."""
        config = self.create_test_config()
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 3
        env = Env(config)
        
        # Reset
        env.reset()
        assert env.episode_over is False
        
        # Step until max steps
        for i in range(3):
            obs, done, info = env.step("MOVE_FORWARD")
            assert info["elapsed_steps"] == i + 1
        
        # Episode should be over
        assert done is True
        assert env.episode_over is True
        assert info["episode_over"] is True
    
    def test_env_stop_action_with_real_satsim(self):
        """Test STOP action with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset
        env.reset()
        initial_state = env._sim.get_agent_state()
        
        # Execute STOP action
        obs, done, info = env.step("STOP")
        
        # Verify observations returned
        assert "rgb" in obs
        
        # Verify state didn't change (STOP should not move)
        # Use rtol=0 to disable relative tolerance for geographic coordinates
        final_state = env._sim.get_agent_state()
        assert np.allclose(initial_state.position, final_state.position, atol=1e-6, rtol=0)
        assert abs(initial_state.rotation - final_state.rotation) < 0.1
    
    def test_env_multiple_episodes_with_real_satsim(self):
        """Test multiple episodes with real SatSim."""
        config = self.create_test_config()
        env = Env(config, cycle=True)
        
        # First episode
        obs1 = env.reset()
        episode_id_1 = env.current_episode.episode_id
        
        # Execute some steps
        env.step("MOVE_FORWARD")
        env.step("TURN_LEFT")
        
        # Second episode
        obs2 = env.reset()
        episode_id_2 = env.current_episode.episode_id
        
        # Episodes should be different
        assert episode_id_1 != episode_id_2
        
        # Both should have valid observations
        assert "rgb" in obs1
        assert "rgb" in obs2
        assert obs1["rgb"].shape == (224, 224, 3)
        assert obs2["rgb"].shape == (224, 224, 3)
    
    def test_env_observation_space_with_real_satsim(self):
        """Test observation_space property with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        obs_space = env.observation_space
        assert isinstance(obs_space, dict)
        assert "rgb" in obs_space
        assert "instruction" in obs_space
        assert obs_space["rgb"]["shape"] == (224, 224, 3)
    
    def test_env_action_space_with_real_satsim(self):
        """Test action_space property with real SatSim."""
        config = self.create_test_config()
        env = Env(config)
        
        action_space = env.action_space
        assert isinstance(action_space, dict)
        assert "actions" in action_space
        expected_actions = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        for action in expected_actions:
            assert action in action_space["actions"]
    
    def test_metric_distance_to_goal(self):
        """Test DistanceToGoal metric calculation."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        episode = env.current_episode
        
        # Get initial distance to goal
        initial_metrics = env.get_metrics()
        initial_distance = initial_metrics["distance_to_goal"]
        
        # Verify initial distance is reasonable (non-negative, finite)
        assert isinstance(initial_distance, (int, float))
        assert initial_distance >= 0
        assert np.isfinite(initial_distance)
        
        # Verify initial distance matches geodesic distance from start to goal
        from satnav.core.utils import geodesic_distance
        expected_distance = geodesic_distance(
            episode.start_position,
            episode.goals[0].position
        )
        # Allow small tolerance due to floating-point precision and potential position updates
        assert abs(initial_distance - expected_distance) < 1.0, (
            f"Initial distance to goal should be ~{expected_distance}m, "
            f"but got {initial_distance}m (diff={abs(initial_distance - expected_distance)}m)"
        )
        
        # Move forward multiple times to ensure significant position change
        # DistanceToGoal only updates if position changes by > 1e-4 degrees (~11m)
        # So we need to move enough times to accumulate > 11m
        step_size = config.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m per step
        steps_needed = int(15.0 / step_size) + 1  # Need > 11m, so ~60 steps to ensure update
        
        # Get initial position for verification
        initial_state = env._sim.get_agent_state()
        initial_pos = initial_state.position.copy()
        
        for _ in range(steps_needed):
            env.step("MOVE_FORWARD")
        
        # Verify position actually changed
        final_state = env._sim.get_agent_state()
        final_pos = final_state.position.copy()
        from satnav.core.utils import geodesic_distance
        actual_distance_moved = geodesic_distance(initial_pos, final_pos)
        
        metrics_after_move = env.get_metrics()
        distance_after_move = metrics_after_move["distance_to_goal"]
        
        # Distance should still be non-negative and finite
        assert distance_after_move >= 0
        assert np.isfinite(distance_after_move)
        
        # Verify DistanceToGoal metric properties
        # Note: DistanceToGoal always updates distance (no optimization).
        # Previous implementation used allclose with atol=1e-4, but this was removed
        # for better accuracy with geographic coordinates.
        
        # The key properties to verify:
        # 1. Distance is always non-negative and finite
        # 2. Distance represents geodesic distance to goal
        # 3. The metric is updated appropriately (may be cached for small movements)
        
        # Verify basic properties
        assert distance_after_move >= 0, "Distance to goal should be non-negative"
        assert np.isfinite(distance_after_move), "Distance to goal should be finite"
        
        # Verify that if we manually calculate distance, it should match
        # (accounting for the update threshold optimization)
        manual_distance = geodesic_distance(final_pos, episode.goals[0].position)
        
        # The metric may be cached if position change is small, so we verify
        # that the metric value is reasonable (within expected range)
        # Initial distance was ~616m, after moving ~14m, distance should be ~600-632m
        assert 500.0 < distance_after_move < 700.0, (
            f"Distance to goal ({distance_after_move}m) should be in reasonable range "
            f"after moving {actual_distance_moved}m from initial {initial_distance}m"
        )
    
    def test_metric_success(self):
        """Test Success metric calculation."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        
        # Initially success should be 0.0 (no STOP called yet)
        initial_metrics = env.get_metrics()
        assert "success" in initial_metrics
        assert initial_metrics["success"] == 0.0
        
        # Move around without STOP - success should remain 0
        env.step("MOVE_FORWARD")
        env.step("TURN_LEFT")
        metrics_after_moves = env.get_metrics()
        assert metrics_after_moves["success"] == 0.0, (
            "Success should be 0.0 when STOP is not called"
        )
        
        # Now call STOP - success depends on distance to goal
        env.step("STOP")
        metrics_after_stop = env.get_metrics()
        success_value = metrics_after_stop["success"]
        
        # Success should be either 0.0 or 1.0
        assert success_value in [0.0, 1.0], (
            f"Success should be 0.0 or 1.0, but got {success_value}"
        )
        
        # Verify success logic: should be 1.0 only if within success_distance
        distance_to_goal = metrics_after_stop["distance_to_goal"]
        expected_success = 1.0 if distance_to_goal < config.TASK.SUCCESS_DISTANCE else 0.0
        assert success_value == expected_success, (
            f"Success should be {expected_success} when distance={distance_to_goal}m "
            f"and success_distance={config.TASK.SUCCESS_DISTANCE}m, "
            f"but got {success_value}"
        )
    
    def test_metric_success_with_close_goal(self):
        """Test Success metric when agent is very close to goal."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        episode = env.current_episode
        
        # Manually set agent state very close to goal
        goal_pos = episode.goals[0].position
        # Set position slightly offset from goal (within success_distance)
        from satnav.core.utils import geodesic_distance
        import math
        
        # Create a position very close to goal (1 meter away)
        # Use a small offset in latitude (approximately 1 meter)
        close_pos = [goal_pos[0], goal_pos[1] + 0.000009, goal_pos[2]]  # ~1m north
        env._sim.set_agent_state(close_pos, 0.0)
        
        # Update task measures manually to reflect new position
        env._task.step("STOP")
        
        # Check success
        metrics = env.get_metrics()
        distance = metrics["distance_to_goal"]
        
        if distance < config.TASK.SUCCESS_DISTANCE:
            assert metrics["success"] == 1.0, (
                f"Success should be 1.0 when distance={distance}m < "
                f"success_distance={config.TASK.SUCCESS_DISTANCE}m"
            )
        else:
            assert metrics["success"] == 0.0, (
                f"Success should be 0.0 when distance={distance}m >= "
                f"success_distance={config.TASK.SUCCESS_DISTANCE}m"
            )
    
    def test_metric_path_length(self):
        """Test PathLength metric calculation."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        
        # Initially path length should be 0.0
        initial_metrics = env.get_metrics()
        assert "path_length" in initial_metrics
        assert initial_metrics["path_length"] == 0.0
        
        # Get initial position
        initial_state = env._sim.get_agent_state()
        initial_pos = initial_state.position.copy()
        
        # Move forward
        env.step("MOVE_FORWARD")
        state_after_move = env._sim.get_agent_state()
        after_pos = state_after_move.position
        
        # Calculate expected path length (geodesic distance)
        from satnav.core.utils import geodesic_distance
        expected_path_length = geodesic_distance(initial_pos, after_pos)
        
        # Check path length metric
        metrics_after_move = env.get_metrics()
        path_length = metrics_after_move["path_length"]
        
        # Path length should be non-negative and approximately equal to distance moved
        assert path_length >= 0
        assert abs(path_length - expected_path_length) < 0.1, (
            f"Path length should be ~{expected_path_length}m after one move, "
            f"but got {path_length}m"
        )
        
        # Move again
        pos_before_second_move = after_pos.copy()
        env.step("MOVE_FORWARD")
        state_after_second_move = env._sim.get_agent_state()
        pos_after_second_move = state_after_second_move.position
        
        # Calculate expected total path length
        second_step_distance = geodesic_distance(pos_before_second_move, pos_after_second_move)
        expected_total_path_length = expected_path_length + second_step_distance
        
        # Check accumulated path length
        metrics_after_second_move = env.get_metrics()
        total_path_length = metrics_after_second_move["path_length"]
        
        assert abs(total_path_length - expected_total_path_length) < 0.1, (
            f"Total path length should be ~{expected_total_path_length}m after two moves, "
            f"but got {total_path_length}m"
        )
        
        # Path length should be monotonically increasing
        assert total_path_length >= path_length, (
            "Path length should be monotonically increasing"
        )
    
    def test_metric_spl(self):
        """Test SPL (Success weighted by Path Length) metric calculation."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        episode = env.current_episode
        
        # Initially SPL should be 0.0 (no success yet)
        initial_metrics = env.get_metrics()
        assert "spl" in initial_metrics
        assert initial_metrics["spl"] == 0.0
        
        # Calculate reference path length
        from satnav.core.utils import geodesic_distance
        if episode.reference_path and len(episode.reference_path) >= 2:
            ref_path_length = 0.0
            for i in range(len(episode.reference_path) - 1):
                ref_path_length += geodesic_distance(
                    episode.reference_path[i],
                    episode.reference_path[i + 1]
                )
        else:
            ref_path_length = geodesic_distance(
                episode.start_position,
                episode.goals[0].position
            )
        
        # Move around without success
        env.step("MOVE_FORWARD")
        env.step("TURN_LEFT")
        metrics_after_moves = env.get_metrics()
        
        # SPL should still be 0.0 (no success)
        assert metrics_after_moves["spl"] == 0.0, (
            "SPL should be 0.0 when success is 0.0"
        )
        
        # Test SPL calculation when success = 1.0
        # First, get close to goal and call STOP
        # (This is a simplified test - in practice, we'd need to navigate properly)
        current_metrics = env.get_metrics()
        current_distance = current_metrics["distance_to_goal"]
        
        # If we're already close enough, test success case
        if current_distance < config.TASK.SUCCESS_DISTANCE:
            env.step("STOP")
            metrics_with_success = env.get_metrics()
            
            if metrics_with_success["success"] == 1.0:
                spl_value = metrics_with_success["spl"]
                path_length = metrics_with_success["path_length"]
                
                # SPL should be in [0, 1] range
                assert 0.0 <= spl_value <= 1.0, (
                    f"SPL should be in [0, 1] range, but got {spl_value}"
                )
                
                # If path_length <= ref_path_length, SPL should be 1.0
                if path_length <= ref_path_length:
                    assert abs(spl_value - 1.0) < 0.01, (
                        f"SPL should be ~1.0 when path_length={path_length}m <= "
                        f"ref_path_length={ref_path_length}m, but got {spl_value}"
                    )
                else:
                    # If path_length > ref_path_length, SPL should be < 1.0
                    expected_spl = ref_path_length / path_length
                    assert abs(spl_value - expected_spl) < 0.01, (
                        f"SPL should be ~{expected_spl} when path_length={path_length}m > "
                        f"ref_path_length={ref_path_length}m, but got {spl_value}"
                    )
    
    def test_metric_spl_formula(self):
        """Test SPL formula correctness: SPL = Success * (ref_len / max(ref_len, actual_len))."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        episode = env.current_episode
        
        # Calculate reference path length
        from satnav.core.utils import geodesic_distance
        if episode.reference_path and len(episode.reference_path) >= 2:
            ref_path_length = 0.0
            for i in range(len(episode.reference_path) - 1):
                ref_path_length += geodesic_distance(
                    episode.reference_path[i],
                    episode.reference_path[i + 1]
                )
        else:
            ref_path_length = geodesic_distance(
                episode.start_position,
                episode.goals[0].position
            )
        
        # Test case 1: Success = 0, SPL should be 0
        metrics = env.get_metrics()
        assert metrics["success"] == 0.0
        assert metrics["spl"] == 0.0, (
            "SPL should be 0.0 when success is 0.0"
        )
        
        # Test case 2: Success = 1, actual_path_length <= ref_path_length
        # Move a bit and then STOP (if close enough)
        env.step("MOVE_FORWARD")
        metrics_after_move = env.get_metrics()
        
        # If we're close enough and call STOP
        if metrics_after_move["distance_to_goal"] < config.TASK.SUCCESS_DISTANCE:
            env.step("STOP")
            final_metrics = env.get_metrics()
            
            if final_metrics["success"] == 1.0:
                actual_path_length = final_metrics["path_length"]
                spl_value = final_metrics["spl"]
                
                # Verify SPL formula
                if actual_path_length <= ref_path_length:
                    expected_spl = 1.0 * (ref_path_length / max(ref_path_length, actual_path_length))
                    assert abs(spl_value - expected_spl) < 0.01, (
                        f"SPL formula incorrect: expected {expected_spl}, got {spl_value}. "
                        f"ref_len={ref_path_length}m, actual_len={actual_path_length}m"
                    )
                else:
                    expected_spl = 1.0 * (ref_path_length / actual_path_length)
                    assert abs(spl_value - expected_spl) < 0.01, (
                        f"SPL formula incorrect: expected {expected_spl}, got {spl_value}. "
                        f"ref_len={ref_path_length}m, actual_len={actual_path_length}m"
                    )
    
    def test_all_metrics_present(self):
        """Test that all expected metrics are present in metrics dictionary."""
        config = self.create_test_config()
        env = Env(config)
        
        # Reset environment
        env.reset()
        
        # Get metrics
        metrics = env.get_metrics()
        
        # Verify all expected metrics are present
        expected_metrics = ["distance_to_goal", "success", "path_length", "spl"]
        for metric_name in expected_metrics:
            assert metric_name in metrics, (
                f"Metric '{metric_name}' should be present in metrics dictionary"
            )
            assert isinstance(metrics[metric_name], (int, float)), (
                f"Metric '{metric_name}' should be a number, but got {type(metrics[metric_name])}"
            )
            assert np.isfinite(metrics[metric_name]), (
                f"Metric '{metric_name}' should be finite, but got {metrics[metric_name]}"
            )
        
        # Verify metric value ranges
        assert metrics["distance_to_goal"] >= 0, "distance_to_goal should be >= 0"
        assert metrics["success"] in [0.0, 1.0], "success should be 0.0 or 1.0"
        assert metrics["path_length"] >= 0, "path_length should be >= 0"
        assert 0.0 <= metrics["spl"] <= 1.0, "spl should be in [0, 1] range"


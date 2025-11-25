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
TEST_CONFIG_PATH = Path(__file__).parent / "test_data" / "test_config.yaml"
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
                    "SCENES_DIR": "data/scene_datasets/"
                }
            })
    
    def create_test_dataset(self) -> SatNavDataset:
        """Create a test dataset."""
        if not EXAMPLE_DATASET_PATH.exists():
            pytest.skip(f"Example dataset not found: {EXAMPLE_DATASET_PATH}")
        
        config = {
            "DATA_PATH": str(EXAMPLE_DATASET_PATH),
            "SPLIT": "test",
        }
        return SatNavDataset(config)
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
    def test_init_with_config_dataset(self, mock_wrapper_class):
        """Test initialization with dataset from config."""
        config = self.create_test_config()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config)
        
        assert env._dataset is not None
        assert len(env._dataset.episodes) > 0
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
    def test_reset_without_dataset(self, mock_wrapper_class):
        """Test reset() without dataset raises error."""
        config = self.create_test_config()
        config.DATASET = None
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=None)
        
        with pytest.raises(RuntimeError, match="dataset is not available"):
            env.reset()
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
    def test_step_without_reset(self, mock_wrapper_class):
        """Test step() without reset() raises error."""
        config = self.create_test_config()
        dataset = self.create_test_dataset()
        
        # Setup mock
        mock_wrapper_class.return_value = MockSatSimWrapper()
        
        env = Env(config, dataset=dataset)
        
        with pytest.raises(RuntimeError, match="episode not set"):
            env.step("MOVE_FORWARD")
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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
    
    @patch('satnav.core.env.SatSimWrapper')
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


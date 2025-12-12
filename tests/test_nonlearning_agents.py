#!/usr/bin/env python3
"""Tests for non-learning agents (RandomAgent, GreedyAgent).

This module tests the implementation of non-learning baseline agents.
"""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from satnav.models import ModelRegistry, RandomAgent, GreedyAgent
from satnav.models.baselines.random_agent import DummyNet


def test_dummy_net_instantiation():
    """Test that DummyNet can be instantiated."""
    net = DummyNet(output_size=512)
    assert net.output_size == 512
    assert net.is_blind == True
    
    # Test get_initial_state
    device = torch.device('cpu')
    initial_state = net.get_initial_state(batch_size=2, device=device)
    assert initial_state.shape == (1, 2, 512)  # (layers, batch, hidden)


def test_dummy_net_forward():
    """Test DummyNet forward pass."""
    net = DummyNet(output_size=512)
    
    # Create dummy inputs
    batch_size = 2
    rnn_states = torch.zeros(1, batch_size, 512)
    observations = {}
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)
    
    # Forward pass
    features, new_rnn_states = net.forward(observations, rnn_states, prev_actions, masks)
    
    # Check outputs
    assert features.shape == (batch_size, 512)
    assert torch.all(features == 0)  # Should be zeros
    assert torch.equal(rnn_states, new_rnn_states)  # Should be unchanged


def test_random_agent_instantiation():
    """Test RandomAgent instantiation."""
    action_probs = [0.02, 0.68, 0.15, 0.15]
    agent = RandomAgent(action_probs, dim_actions=4)
    
    assert agent.dim_actions == 4
    assert torch.allclose(agent.action_probs, torch.tensor(action_probs))


def test_random_agent_invalid_probs():
    """Test RandomAgent rejects invalid probabilities."""
    # Wrong length
    with pytest.raises(ValueError, match="Length of action_probs"):
        RandomAgent([0.5, 0.5], dim_actions=4)
    
    # Doesn't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        RandomAgent([0.3, 0.3, 0.3, 0.3], dim_actions=4)


def test_random_agent_act_deterministic():
    """Test RandomAgent ignores deterministic parameter and always samples randomly."""
    action_probs = [0.0, 0.5, 0.5, 0.0]  # MOVE_FORWARD or TURN_LEFT
    torch.manual_seed(42)  # Set seed for reproducible test
    agent = RandomAgent(action_probs, dim_actions=4, seed=None)
    
    # Create dummy inputs
    batch_size = 20
    rnn_states = torch.zeros(1, batch_size, 512)
    observations = {}
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)
    
    # Act in deterministic mode - RandomAgent should IGNORE this and still be random
    action, new_rnn_states = agent.act(
        observations, rnn_states, prev_actions, masks, deterministic=True
    )
    
    # Check output
    assert action.shape == (batch_size, 1)
    # Should have both actions (not all the same) since RandomAgent ignores deterministic
    unique_actions = torch.unique(action)
    assert len(unique_actions) > 1, "RandomAgent should sample randomly even with deterministic=True"
    # Only actions 1 and 2 should appear (based on probs)
    assert torch.all((action == 1) | (action == 2))
    assert torch.equal(rnn_states, new_rnn_states)


def test_random_agent_act_stochastic():
    """Test RandomAgent act method in stochastic mode."""
    action_probs = [0.25, 0.25, 0.25, 0.25]  # Equal probability
    agent = RandomAgent(action_probs, dim_actions=4)
    
    # Create dummy inputs
    batch_size = 10
    rnn_states = torch.zeros(1, batch_size, 512)
    observations = {}
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)
    
    # Act in stochastic mode
    action, new_rnn_states = agent.act(
        observations, rnn_states, prev_actions, masks, deterministic=False
    )
    
    # Check output shape
    assert action.shape == (batch_size, 1)
    
    # Check actions are in valid range
    assert torch.all(action >= 0)
    assert torch.all(action < 4)


def test_random_agent_distribution():
    """Test that RandomAgent samples approximately according to distribution."""
    action_probs = [0.1, 0.6, 0.2, 0.1]
    agent = RandomAgent(action_probs, dim_actions=4)
    
    # Sample many times
    num_samples = 1000
    rnn_states = torch.zeros(1, 1, 512)
    observations = {}
    prev_actions = torch.zeros(1, 1)
    masks = torch.ones(1, 1)
    
    action_counts = torch.zeros(4)
    for _ in range(num_samples):
        action, _ = agent.act(observations, rnn_states, prev_actions, masks, deterministic=False)
        action_counts[action.item()] += 1
    
    # Check distribution is approximately correct (within 10% tolerance)
    empirical_probs = action_counts / num_samples
    expected_probs = torch.tensor(action_probs)
    
    assert torch.allclose(empirical_probs, expected_probs, atol=0.1)


def test_random_agent_from_config_with_probs():
    """Test creating RandomAgent from config with manual probabilities."""
    config = OmegaConf.create({
        'MODEL': {
            'RANDOM_AGENT': {
                'action_probs': [0.1, 0.5, 0.2, 0.2]
            }
        }
    })
    
    class DummyActionSpace:
        n = 4
    
    agent = RandomAgent.from_config(config, None, DummyActionSpace())
    
    assert agent.dim_actions == 4
    assert torch.allclose(
        agent.action_probs,
        torch.tensor([0.1, 0.5, 0.2, 0.2])
    )


def test_greedy_agent_instantiation():
    """Test GreedyAgent instantiation."""
    agent = GreedyAgent(goal_radius=3.0, turn_angle=15.0, dim_actions=4)
    
    assert agent.dim_actions == 4
    assert agent.path_follower.goal_radius == 3.0
    assert agent.path_follower.turn_angle == 15.0
    assert agent._env is None


def test_greedy_agent_set_env():
    """Test GreedyAgent set_env method."""
    agent = GreedyAgent(goal_radius=3.0, turn_angle=15.0, dim_actions=4)
    
    class DummyEnv:
        pass
    
    env = DummyEnv()
    agent.set_env(env)
    
    assert agent._env is env


def test_greedy_agent_act_without_env():
    """Test GreedyAgent raises error if env not set."""
    agent = GreedyAgent(goal_radius=3.0, turn_angle=15.0, dim_actions=4)
    
    # Create dummy inputs
    batch_size = 1
    rnn_states = torch.zeros(1, batch_size, 512)
    observations = {}
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)
    
    # Should raise error
    with pytest.raises(RuntimeError, match="Environment not set"):
        agent.act(observations, rnn_states, prev_actions, masks)


def test_greedy_agent_from_config():
    """Test creating GreedyAgent from config."""
    config = OmegaConf.create({
        'MODEL': {
            'GREEDY_AGENT': {
                'goal_radius': 5.0,
                'turn_angle': 30.0
            }
        }
    })
    
    class DummyActionSpace:
        n = 4
    
    agent = GreedyAgent.from_config(config, None, DummyActionSpace())
    
    assert agent.dim_actions == 4
    assert agent.path_follower.goal_radius == 5.0
    assert agent.path_follower.turn_angle == 30.0


def test_agents_registry():
    """Test that agents are properly registered."""
    assert "random" in ModelRegistry.list_models()['baseline']
    assert "greedy" in ModelRegistry.list_models()['baseline']
    
    # Test retrieval
    RandomAgentClass = ModelRegistry.get_model("random")
    GreedyAgentClass = ModelRegistry.get_model("greedy")
    
    assert RandomAgentClass is RandomAgent
    assert GreedyAgentClass is GreedyAgent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


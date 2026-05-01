"""Tests for CMA policy implementation.

This module tests the CMA (Cross-Modal Attention) policy to ensure:
1. Policy can be instantiated correctly
2. Forward pass produces correct output shapes
3. Compatible with RecollectTrainer
4. RNN states are handled correctly (CMA has 2 RNN encoders)
"""

import pytest
import torch
from omegaconf import OmegaConf

from satnav.models import ModelRegistry, CMAPolicy
from satnav.models.baselines.cma_policy import CMANet


def test_cma_policy_instantiation():
    """Test that CMA policy can be instantiated."""
    # Create minimal config
    config = OmegaConf.create({
        'MODEL': {
            'policy_name': 'cma',
            'normalize_rgb': False,
            'ablate_instruction': False,
            'ablate_rgb': False,
            'INSTRUCTION_ENCODER': {
                'sensor_uuid': 'instruction',
                'vocab_size': 100,
                'use_pretrained_embeddings': False,
                'embedding_size': 50,
                'hidden_size': 128,
                'rnn_type': 'LSTM',
                'bidirectional': True,
                'num_layers': 1,
                'final_state_only': False,
            },
            'RGB_ENCODER': {
                'cnn_type': 'TorchVisionResNet50',
                'output_size': 256,
                'trainable': False,
                'normalize_visual_inputs': False,
            },
            'CMA': {
                'hidden_size': 512,
                'rnn_type': 'GRU',
                'use_prev_action': True,
            },
        }
    })
    
    observation_space = {
        'rgb': {'shape': (224, 224, 3)},
        'instruction': {'max_length': 80},
    }
    
    action_space = {
        'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    }
    
    # Create policy
    policy = CMAPolicy.from_config(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
    )
    
    assert policy is not None
    assert isinstance(policy, CMAPolicy)
    assert isinstance(policy.net, CMANet)
    assert policy.dim_actions == 4


def test_cma_net_properties():
    """Test CMANet properties."""
    config = OmegaConf.create({
        'INSTRUCTION_ENCODER': {
            'sensor_uuid': 'instruction',
            'vocab_size': 100,
            'use_pretrained_embeddings': False,
            'embedding_size': 50,
            'hidden_size': 128,
            'rnn_type': 'LSTM',
            'bidirectional': True,
            'num_layers': 1,
            'final_state_only': False,
        },
        'RGB_ENCODER': {
            'output_size': 256,
            'trainable': False,
        },
        'CMA': {
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'use_prev_action': True,
        },
        'normalize_rgb': False,
        'ablate_instruction': False,
        'ablate_rgb': False,
    })
    
    observation_space = {}
    num_actions = 4
    
    net = CMANet(
        observation_space=observation_space,
        model_config=config,
        num_actions=num_actions,
    )
    
    # Check properties
    assert net.output_size == 512
    assert net.is_blind == False
    
    # Test get_initial_state
    device = torch.device('cpu')
    initial_state = net.get_initial_state(batch_size=2, device=device)
    # CMA has 2 GRU encoders, each with 1 layer = 2 total layers
    assert initial_state.shape == (2, 2, 512)  # (layers, batch, hidden)


def test_cma_forward_pass():
    """Test CMA forward pass with dummy data."""
    config = OmegaConf.create({
        'INSTRUCTION_ENCODER': {
            'sensor_uuid': 'instruction',
            'vocab_size': 100,
            'use_pretrained_embeddings': False,
            'embedding_size': 50,
            'hidden_size': 128,
            'rnn_type': 'LSTM',
            'bidirectional': True,
            'num_layers': 1,
            'final_state_only': False,
        },
        'RGB_ENCODER': {
            'output_size': 256,
            'trainable': False,
        },
        'CMA': {
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'use_prev_action': True,
        },
        'normalize_rgb': False,
        'ablate_instruction': False,
        'ablate_rgb': False,
    })
    
    observation_space = {}
    action_space = {'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']}
    
    policy = CMAPolicy(
        observation_space=observation_space,
        action_space=action_space,
        model_config=config,
    )
    
    # Create dummy observations
    batch_size = 2
    observations = {
        'rgb': torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        'instruction': torch.randint(0, 100, (batch_size, 80), dtype=torch.long),
    }
    
    # Initialize model states
    device = torch.device("cpu")
    rnn_states = policy.net.get_initial_state(batch_size, device)
    
    prev_actions = torch.zeros(batch_size, 1, dtype=torch.long)
    masks = torch.ones(batch_size, 1)
    
    # Forward pass
    with torch.no_grad():
        features, rnn_states_out = policy.net(
            observations, rnn_states, prev_actions, masks
        )
    
    # Check output shapes
    assert features.shape == (batch_size, 512)
    assert rnn_states_out.shape == (2, batch_size, 512)


def test_cma_act():
    """Test CMA policy act method."""
    config = OmegaConf.create({
        'INSTRUCTION_ENCODER': {
            'sensor_uuid': 'instruction',
            'vocab_size': 100,
            'use_pretrained_embeddings': False,
            'embedding_size': 50,
            'hidden_size': 128,
            'rnn_type': 'LSTM',
            'bidirectional': True,
            'num_layers': 1,
            'final_state_only': False,
        },
        'RGB_ENCODER': {
            'output_size': 256,
            'trainable': False,
        },
        'CMA': {
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'use_prev_action': True,
        },
        'normalize_rgb': False,
        'ablate_instruction': False,
        'ablate_rgb': False,
    })
    
    observation_space = {}
    action_space = {'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']}
    
    policy = CMAPolicy(
        observation_space=observation_space,
        action_space=action_space,
        model_config=config,
    )
    
    # Create dummy observations
    batch_size = 2
    observations = {
        'rgb': torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        'instruction': torch.randint(0, 100, (batch_size, 80), dtype=torch.long),
    }
    
    rnn_states = torch.zeros(2, batch_size, 512)
    prev_actions = torch.zeros(batch_size, 1, dtype=torch.long)
    masks = torch.ones(batch_size, 1)
    
    # Test act (sampling)
    with torch.no_grad():
        action, rnn_states_out = policy.act(
            observations, rnn_states, prev_actions, masks, deterministic=False
        )
    
    assert action.shape == (batch_size, 1)
    assert rnn_states_out.shape == (2, batch_size, 512)
    assert action.dtype == torch.long
    assert torch.all(action >= 0) and torch.all(action < 4)
    
    # Test act (deterministic)
    with torch.no_grad():
        action_det, _ = policy.act(
            observations, rnn_states, prev_actions, masks, deterministic=True
        )
    
    assert action_det.shape == (batch_size, 1)


def test_cma_registry():
    """Test that CMA is registered in ModelRegistry."""
    policy_class = ModelRegistry.get_model('cma')
    assert policy_class == CMAPolicy


def test_cma_parameter_count():
    """Test CMA parameter count (should be reasonable)."""
    config = OmegaConf.create({
        'INSTRUCTION_ENCODER': {
            'sensor_uuid': 'instruction',
            'vocab_size': 2504,  # VLN-CE vocab size
            'use_pretrained_embeddings': False,
            'embedding_size': 50,
            'hidden_size': 128,
            'rnn_type': 'LSTM',
            'bidirectional': True,
            'num_layers': 1,
            'final_state_only': False,
        },
        'RGB_ENCODER': {
            'output_size': 256,
            'trainable': False,
        },
        'CMA': {
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'use_prev_action': True,
        },
        'normalize_rgb': False,
        'ablate_instruction': False,
        'ablate_rgb': False,
    })
    
    observation_space = {}
    action_space = {'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']}
    
    policy = CMAPolicy(
        observation_space=observation_space,
        action_space=action_space,
        model_config=config,
    )
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"\nCMA Policy Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # CMA should have more parameters than Seq2Seq due to attention mechanisms
    # Rough estimate: 30-40M total (including frozen ResNet)
    assert total_params > 20_000_000, "CMA should have > 20M parameters"
    assert trainable_params > 1_000_000, "CMA should have > 1M trainable parameters"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

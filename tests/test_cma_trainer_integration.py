"""Integration tests for CMA with RecollectTrainer.

This module tests that CMA policy works correctly with the RecollectTrainer,
especially handling the dual RNN architecture.
"""

import pytest
import torch
from omegaconf import OmegaConf

from satnav.models import CMAPolicy
from satnav.training.base_il_trainer import BaseILTrainer


def test_cma_trainer_rnn_states():
    """Test that trainer correctly initializes RNN states for CMA."""
    # Create minimal config
    config = OmegaConf.create({
        'TORCH_GPU_ID': 0,
        'CHECKPOINT_FOLDER': '/tmp/test_checkpoints',
        'IL': {
            'lr': 1e-4,
            'RECOLLECT_TRAINER': {
                'use_scheduled_sampling': False,
                'scheduled_sampling_ratio': 0.0,
                'scheduled_sampling_p': 0.5,
            },
            'use_class_weighting': False,
        },
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
            'STATE_ENCODER': {
                'hidden_size': 512,
                'rnn_type': 'GRU',
                'num_layers': 1,
            },
        }
    })
    
    # Create trainer
    trainer = BaseILTrainer(config)
    
    # Initialize policy
    observation_space = {
        'rgb': {'shape': (224, 224, 3)},
        'instruction': {'max_length': 80},
    }
    action_space = {
        'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    }
    
    trainer._initialize_policy(
        config=config,
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
    )
    
    # Check that policy is CMA
    assert isinstance(trainer.policy, CMAPolicy)
    
    # Check num_recurrent_layers
    assert trainer.policy.net.num_recurrent_layers == 2
    
    # Simulate trainer's RNN state initialization
    N = 2  # batch size
    rnn_states = torch.zeros(
        trainer.policy.net.num_recurrent_layers,
        N,
        config.MODEL.STATE_ENCODER.hidden_size,
        device=trainer.device
    )
    
    # Check shape
    assert rnn_states.shape == (2, N, 512)
    
    # Test forward pass with these states
    observations = {
        'rgb': torch.randint(0, 256, (N, 224, 224, 3), dtype=torch.uint8).to(trainer.device),
        'instruction': torch.randint(0, 100, (N, 80), dtype=torch.long).to(trainer.device),
    }
    prev_actions = torch.zeros(N, 1, dtype=torch.long).to(trainer.device)
    masks = torch.ones(N, 1).to(trainer.device)
    
    with torch.no_grad():
        distribution = trainer.policy.build_distribution(
            observations, rnn_states, prev_actions, masks
        )
    
    # Check that distribution is valid
    assert distribution.logits.shape == (N, 4)


def test_cma_update_agent():
    """Test that _update_agent works with CMA policy."""
    config = OmegaConf.create({
        'TORCH_GPU_ID': 0,
        'CHECKPOINT_FOLDER': '/tmp/test_checkpoints',
        'IL': {
            'lr': 1e-4,
            'RECOLLECT_TRAINER': {
                'use_scheduled_sampling': False,
                'scheduled_sampling_ratio': 0.0,
                'scheduled_sampling_p': 0.5,
            },
            'use_class_weighting': False,
        },
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
            'STATE_ENCODER': {
                'hidden_size': 512,
                'rnn_type': 'GRU',
                'num_layers': 1,
            },
        }
    })
    
    trainer = BaseILTrainer(config)
    
    observation_space = {
        'rgb': {'shape': (224, 224, 3)},
        'instruction': {'max_length': 80},
    }
    action_space = {
        'actions': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    }
    
    trainer._initialize_policy(
        config=config,
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
    )
    
    # Create dummy batch data
    T, N = 5, 2  # 5 timesteps, 2 episodes
    observations = {
        'rgb': torch.randint(0, 256, (T*N, 224, 224, 3), dtype=torch.uint8).to(trainer.device),
        'instruction': torch.randint(0, 100, (T*N, 80), dtype=torch.long).to(trainer.device),
    }
    prev_actions = torch.randint(0, 4, (T*N, 1), dtype=torch.long).to(trainer.device)
    not_done_masks = torch.ones(T*N, 1).to(trainer.device)
    teacher_actions = torch.randint(0, 4, (T, N), dtype=torch.long).to(trainer.device)
    
    # Run update
    loss = trainer._update_agent(
        observations=observations,
        prev_actions=prev_actions,
        not_done_masks=not_done_masks,
        teacher_actions=teacher_actions,
        epoch=0,
    )
    
    # Check that loss is valid
    assert isinstance(loss, float)
    assert loss > 0
    print(f"\nCMA training loss: {loss:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])


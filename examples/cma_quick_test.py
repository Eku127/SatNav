#!/usr/bin/env python3
"""Quick test script for CMA model.

This script demonstrates how to instantiate and use the CMA model.
"""

import torch
from omegaconf import OmegaConf

from satnav.models import ModelRegistry, CMAPolicy


def main():
    """Run a quick test of the CMA model."""
    print("="*80)
    print("CMA Model Quick Test")
    print("="*80)
    
    # Create minimal config
    config = OmegaConf.create({
        'MODEL': {
            'policy_name': 'cma',
            'normalize_rgb': False,
            'ablate_instruction': False,
            'ablate_rgb': False,
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
    
    # 1. Test model registry
    print("\n1. Testing Model Registry...")
    model_class = ModelRegistry.get_model('cma')
    print(f"   ✓ Retrieved model class: {model_class.__name__}")
    
    # 2. Create policy
    print("\n2. Creating CMA Policy...")
    policy = CMAPolicy.from_config(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
    )
    print(f"   ✓ Policy created successfully")
    
    # 3. Check properties
    print("\n3. Checking Model Properties...")
    print(f"   - Output size: {policy.net.output_size}")
    print(f"   - Is blind: {policy.net.is_blind}")
    print(f"   - Num actions: {policy.dim_actions}")
    
    # 4. Count parameters
    print("\n4. Counting Parameters...")
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Frozen parameters: {total_params - trainable_params:,}")
    
    # 5. Test forward pass
    print("\n5. Testing Forward Pass...")
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    
    observations = {
        'rgb': torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8).to(device),
        'instruction': torch.randint(0, 100, (batch_size, 80), dtype=torch.long).to(device),
    }
    
    # Initialize model states using get_initial_state
    rnn_states = policy.net.get_initial_state(batch_size, device)
    prev_actions = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
    masks = torch.ones(batch_size, 1).to(device)
    
    with torch.no_grad():
        # Test net forward
        features, rnn_states_out = policy.net(
            observations, rnn_states, prev_actions, masks
        )
        print(f"   ✓ Net forward pass successful")
        print(f"     - Features shape: {features.shape}")
        print(f"     - RNN states shape: {rnn_states_out.shape}")
        
        # Test policy act
        action, rnn_states_out = policy.act(
            observations, rnn_states, prev_actions, masks, deterministic=True
        )
        print(f"   ✓ Policy act successful")
        print(f"     - Action shape: {action.shape}")
        print(f"     - Action values: {action.squeeze().tolist()}")
    
    print("\n" + "="*80)
    print("✓ All tests passed! CMA model is working correctly.")
    print("="*80)


if __name__ == '__main__':
    main()


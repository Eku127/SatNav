#!/usr/bin/env python3
"""Test if model relies on prev_action pattern vs RGB+instruction.

This script simulates what happens when we change prev_action
but keep RGB+instruction the same, and vice versa.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from omegaconf import OmegaConf

from satnav.models import ModelRegistry
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.core.env import Env
from satnav.utils.build_vocab import VocabDict, build_vocab_from_dataset


def test_prev_action_pattern():
    """Test if model relies on prev_action pattern."""
    print("=" * 80)
    print("Test: Does Model Rely on Prev_Action Pattern?")
    print("=" * 80)
    
    # Load config
    config_path = 'configs/baselines/seq2seq.yaml'
    config = OmegaConf.load(config_path)
    if '_base_' in config:
        base_config = OmegaConf.load(config._base_)
        config = OmegaConf.merge(base_config, config)
        del config._base_
    if 'BASE_TASK_CONFIG_PATH' in config:
        task_config = OmegaConf.load(config.BASE_TASK_CONFIG_PATH)
        config = OmegaConf.merge(task_config, config)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and environment
    dataset = SatNavDataset(config.DATASET)
    env = Env(config, dataset=dataset, cycle=False)
    
    # Load vocabulary
    vocab_file = config.DATASET.get('vocab_file', None)
    if vocab_file and os.path.exists(vocab_file):
        vocab = VocabDict.load(vocab_file)
    else:
        dataset_path = config.DATASET.DATA_PATH.format(split=config.DATASET.SPLIT)
        vocab = build_vocab_from_dataset(dataset_path)
    
    # Initialize policy
    policy_name = config.MODEL.policy_name
    policy_class = ModelRegistry.get_model(policy_name)
    
    # Check checkpoint
    ckpt_path = config.EVAL.CKPT_PATH
    has_prev_action = False
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        has_prev_action = 'net.prev_action_embedding.weight' in ckpt['state_dict']
        config.MODEL.SEQ2SEQ.use_prev_action = has_prev_action
    
    policy = policy_class.from_config(
        config=config,
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    policy.to(device)
    policy.eval()
    
    if os.path.exists(ckpt_path):
        policy.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    
    # Test 1: Same RGB+Instruction, Different Prev_Action History
    print("\n" + "=" * 80)
    print("Test 1: Same RGB+Instruction, Different Action History")
    print("=" * 80)
    
    obs = env.reset()
    from satnav.utils.build_vocab import tokenize_instruction_in_observation
    obs = tokenize_instruction_in_observation(obs, vocab, max_length=None, output_format="numpy")
    
    # Initialize model states
    # Scenario A: History of MOVE_FORWARD actions
    print("\nScenario A: History = [MOVE_FORWARD, MOVE_FORWARD, MOVE_FORWARD]")
    rnn_state_A = policy.net.get_initial_state(1, device)
    prev_action_A = torch.zeros(1, 1, device=device, dtype=torch.long)
    
    # Simulate 3 steps of MOVE_FORWARD (update RNN state properly)
    for step in range(3):
        obs_batch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
                obs_batch[key] = tensor_value
        
        with torch.no_grad():
            # Forward pass updates RNN state
            features, rnn_state_A = policy.net(
                obs_batch, rnn_state_A, prev_action_A,
                torch.ones(1, 1, device=device, dtype=torch.uint8)
            )
        
        # Update prev_action for next step
        prev_action_A = torch.tensor([[1]], device=device, dtype=torch.long)  # MOVE_FORWARD
    
    # Now make prediction with this history
    obs_batch = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
            obs_batch[key] = tensor_value
    
    with torch.no_grad():
        distribution_A = policy.build_distribution(
            obs_batch, rnn_state_A, prev_action_A,
            torch.ones(1, 1, device=device, dtype=torch.uint8)
        )
        probs_A = distribution_A.probs
        action_A = distribution_A.mode().item()
    
    action_names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    print(f"  Prediction: {action_names[action_A]}")
    print(f"  Probabilities:")
    for i, name in action_names.items():
        print(f"    {name}: {probs_A[0, i].item():.4f}")
    
    # Scenario B: History of TURN_LEFT actions
    print("\nScenario B: History = [TURN_LEFT, TURN_LEFT, TURN_LEFT]")
    rnn_state_B = torch.zeros(rnn_num_layers, 1, hidden_size, device=device)
    prev_action_B = torch.zeros(1, 1, device=device, dtype=torch.long)
    
    # Simulate 3 steps of TURN_LEFT (update RNN state properly)
    for step in range(3):
        obs_batch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
                obs_batch[key] = tensor_value
        
        with torch.no_grad():
            # Forward pass updates RNN state
            features, rnn_state_B = policy.net(
                obs_batch, rnn_state_B, prev_action_B,
                torch.ones(1, 1, device=device, dtype=torch.uint8)
            )
        
        # Update prev_action for next step
        prev_action_B = torch.tensor([[2]], device=device, dtype=torch.long)  # TURN_LEFT
    
    # Now make prediction with this history
    obs_batch = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
            obs_batch[key] = tensor_value
    
    with torch.no_grad():
        distribution_B = policy.build_distribution(
            obs_batch, rnn_state_B, prev_action_B,
            torch.ones(1, 1, device=device, dtype=torch.uint8)
        )
        probs_B = distribution_B.probs
        action_B = distribution_B.mode().item()
    
    print(f"  Prediction: {action_names[action_B]}")
    print(f"  Probabilities:")
    for i, name in action_names.items():
        print(f"    {name}: {probs_B[0, i].item():.4f}")
    
    # Compare
    diff = (probs_A - probs_B).abs().sum().item()
    print(f"\n  Difference between scenarios: {diff:.4f}")
    if diff < 0.05:
        print("  → Model ignores action history! (Same prediction despite different history)")
    else:
        print("  → Model uses action history (Different predictions)")
    
    # Test 2: Different RGB+Instruction, Same Prev_Action
    print("\n" + "=" * 80)
    print("Test 2: Different RGB+Instruction, Same Prev_Action")
    print("=" * 80)
    
    # Get two different observations
    obs1 = env.reset()
    obs1 = tokenize_instruction_in_observation(obs1, vocab, max_length=None, output_format="numpy")
    
    # Reset to get a different episode
    if len(env._dataset.episodes) > 1:
        env._current_episode_idx = 1
        obs2 = env.reset()
        obs2 = tokenize_instruction_in_observation(obs2, vocab, max_length=None, output_format="numpy")
    else:
        obs2 = obs1.copy()
        print("  Warning: Only one episode available, using same observation")
    
    rnn_state = torch.zeros(rnn_num_layers, 1, hidden_size, device=device)
    prev_action = torch.tensor([[1]], device=device, dtype=torch.long)  # Same prev_action
    
    # Prediction with obs1
    obs_batch1 = {}
    for key, value in obs1.items():
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
            obs_batch1[key] = tensor_value
    
    with torch.no_grad():
        distribution1 = policy.build_distribution(
            obs_batch1, rnn_state, prev_action,
            torch.ones(1, 1, device=device, dtype=torch.uint8)
        )
        probs1 = distribution1.probs
        action1 = distribution1.mode().item()
    
    print(f"\nObservation 1:")
    print(f"  Prediction: {action_names[action1]}")
    print(f"  Probabilities:")
    for i, name in action_names.items():
        print(f"    {name}: {probs1[0, i].item():.4f}")
    
    # Prediction with obs2
    obs_batch2 = {}
    for key, value in obs2.items():
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
            obs_batch2[key] = tensor_value
    
    with torch.no_grad():
        distribution2 = policy.build_distribution(
            obs_batch2, rnn_state, prev_action,
            torch.ones(1, 1, device=device, dtype=torch.uint8)
        )
        probs2 = distribution2.probs
        action2 = distribution2.mode().item()
    
    print(f"\nObservation 2:")
    print(f"  Prediction: {action_names[action2]}")
    print(f"  Probabilities:")
    for i, name in action_names.items():
        print(f"    {name}: {probs2[0, i].item():.4f}")
    
    diff_obs = (probs1 - probs2).abs().sum().item()
    print(f"\n  Difference between observations: {diff_obs:.4f}")
    if diff_obs < 0.05:
        print("  → Model ignores RGB+Instruction! (Same prediction despite different observations)")
    else:
        print("  → Model uses RGB+Instruction (Different predictions)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)
    
    if diff < 0.05 and diff_obs > 0.1:
        print("✓ Model relies MORE on RGB+Instruction")
        print("✗ Model ignores action history")
        print("→ Model learns navigation from visual/linguistic cues")
    elif diff > 0.1 and diff_obs < 0.05:
        print("✓ Model relies MORE on action history (prev_action pattern)")
        print("✗ Model ignores RGB+Instruction")
        print("→ Model learned pattern matching, not navigation!")
    elif diff < 0.05 and diff_obs < 0.05:
        print("✗ Model ignores both history and observations")
        print("→ Model may be predicting based on bias/prior")
    else:
        print("→ Model uses both history and observations")


if __name__ == "__main__":
    test_prev_action_pattern()


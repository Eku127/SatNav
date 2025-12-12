#!/usr/bin/env python3
"""Visualize feature usage across training trajectory.

This script visualizes how the model uses different features
(RGB, instruction, prev_action) at different timesteps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from satnav.models import ModelRegistry
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.core.env import Env
from satnav.utils.build_vocab import VocabDict, build_vocab_from_dataset


def visualize_feature_usage():
    """Visualize how model uses features."""
    print("=" * 80)
    print("Feature Usage Visualization")
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
    policy = policy_class.from_config(
        config=config,
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    policy.to(device)
    policy.eval()
    
    # Load checkpoint
    ckpt_path = config.EVAL.CKPT_PATH
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['state_dict'])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        return
    
    # Collect data
    obs = env.reset()
    from satnav.utils.build_vocab import tokenize_instruction_in_observation
    obs = tokenize_instruction_in_observation(obs, vocab, max_length=None, output_format="numpy")
    
    # Initialize model state
    rnn_state = policy.net.get_initial_state(1, device)
    prev_action = torch.zeros(1, 1, device=device, dtype=torch.long)
    not_done_mask = torch.ones(1, 1, device=device, dtype=torch.uint8)
    
    timesteps = []
    inst_impacts = []
    rgb_impacts = []
    prev_impacts = []
    predicted_actions = []
    ground_truth_actions = []
    
    max_steps = 20
    for step in range(max_steps):
        # Prepare observation batch
        obs_batch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
                obs_batch[key] = tensor_value
        
        # Baseline prediction
        with torch.no_grad():
            distribution_baseline = policy.build_distribution(
                obs_batch, rnn_state, prev_action, not_done_mask
            )
            probs_baseline = distribution_baseline.probs
            action_baseline = distribution_baseline.mode().item()
        
        # Test removing instruction
        obs_batch_no_inst = obs_batch.copy()
        if 'instruction' in obs_batch_no_inst:
            obs_batch_no_inst['instruction'] = torch.zeros_like(obs_batch_no_inst['instruction'])
            obs_batch_no_inst['instruction'][0, 0] = 1
        
        with torch.no_grad():
            distribution_no_inst = policy.build_distribution(
                obs_batch_no_inst, rnn_state, prev_action, not_done_mask
            )
            probs_no_inst = distribution_no_inst.probs
        
        inst_impact = (probs_baseline - probs_no_inst).abs().sum().item()
        
        # Test removing RGB
        obs_batch_no_rgb = obs_batch.copy()
        if 'rgb' in obs_batch_no_rgb:
            obs_batch_no_rgb['rgb'] = torch.zeros_like(obs_batch_no_rgb['rgb'])
        
        with torch.no_grad():
            distribution_no_rgb = policy.build_distribution(
                obs_batch_no_rgb, rnn_state, prev_action, not_done_mask
            )
            probs_no_rgb = distribution_no_rgb.probs
        
        rgb_impact = (probs_baseline - probs_no_rgb).abs().sum().item()
        
        # Test changing prev_action
        if config.MODEL.SEQ2SEQ.use_prev_action:
            prev_action_alt = torch.randint(1, 4, (1, 1), device=device, dtype=torch.long)
            
            with torch.no_grad():
                distribution_diff_prev = policy.build_distribution(
                    obs_batch, rnn_state, prev_action_alt, not_done_mask
                )
                probs_diff_prev = distribution_diff_prev.probs
            
            prev_impact = (probs_baseline - probs_diff_prev).abs().sum().item()
        else:
            prev_impact = 0.0
        
        # Store results
        timesteps.append(step)
        inst_impacts.append(inst_impact)
        rgb_impacts.append(rgb_impact)
        prev_impacts.append(prev_impact)
        predicted_actions.append(action_baseline)
        
        # Execute action
        action_idx = action_baseline
        obs, done, info = env.step(action_idx)
        obs = tokenize_instruction_in_observation(obs, vocab, max_length=None, output_format="numpy")
        
        prev_action = torch.tensor([[action_idx]], device=device, dtype=torch.long)
        not_done_mask = torch.tensor(
            [[0] if done else [1]],
            dtype=torch.uint8,
            device=device
        )
        
        if done:
            break
    
    # Print results
    print("\nFeature Impact Across Timesteps:")
    print("-" * 80)
    print(f"{'Step':<6} {'Instruction':<12} {'RGB':<12} {'Prev_Action':<12} {'Predicted':<10}")
    print("-" * 80)
    for i in range(len(timesteps)):
        action_names = {0: "STOP", 1: "MOVE", 2: "LEFT", 3: "RIGHT"}
        print(f"{timesteps[i]:<6} {inst_impacts[i]:<12.4f} {rgb_impacts[i]:<12.4f} "
              f"{prev_impacts[i]:<12.4f} {action_names.get(predicted_actions[i], 'UNK'):<10}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Average Instruction Impact: {np.mean(inst_impacts):.4f}")
    print(f"Average RGB Impact: {np.mean(rgb_impacts):.4f}")
    print(f"Average Prev_Action Impact: {np.mean(prev_impacts):.4f}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    impacts = {
        'Instruction': np.mean(inst_impacts),
        'RGB': np.mean(rgb_impacts),
        'Prev_Action': np.mean(prev_impacts)
    }
    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance Ranking:")
    for i, (feature, impact) in enumerate(sorted_impacts, 1):
        print(f"  {i}. {feature}: {impact:.4f}")
    
    if sorted_impacts[0][0] == 'Prev_Action':
        print("\n✓ Model relies MOST on prev_action")
        print("✗ Model relies LESS on RGB and instruction")
        print("→ This confirms: Model uses prev_action as shortcut!")
    elif sorted_impacts[0][0] == 'RGB':
        print("\n✓ Model relies MORE on RGB")
        print("→ Model uses visual cues")
    else:
        print("\n→ Model uses instruction most")
    
    # Save visualization data
    output_file = 'output/feature_importance_data.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Step\tInstruction_Impact\tRGB_Impact\tPrevAction_Impact\tPredicted_Action\n")
        for i in range(len(timesteps)):
            f.write(f"{timesteps[i]}\t{inst_impacts[i]:.6f}\t{rgb_impacts[i]:.6f}\t"
                   f"{prev_impacts[i]:.6f}\t{predicted_actions[i]}\n")
    print(f"\nData saved to: {output_file}")


if __name__ == "__main__":
    visualize_feature_usage()


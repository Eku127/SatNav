#!/usr/bin/env python3
"""Analyze feature importance: RGB, Instruction, Prev_Action.

This script performs ablation studies to determine which features
the model relies on most for action prediction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from collections import defaultdict

from satnav.models import ModelRegistry
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.core.env import Env
from satnav.utils.build_vocab import VocabDict, build_vocab_from_dataset


class FeatureImportanceAnalyzer:
    """Analyze which features the model relies on."""
    
    def __init__(self, config_path, ckpt_path):
        """Initialize analyzer.
        
        Args:
            config_path: Path to config file
            ckpt_path: Path to checkpoint file
        """
        # Load config
        config = OmegaConf.load(config_path)
        if '_base_' in config:
            base_config = OmegaConf.load(config._base_)
            config = OmegaConf.merge(base_config, config)
            del config._base_
        if 'BASE_TASK_CONFIG_PATH' in config:
            task_config = OmegaConf.load(config.BASE_TASK_CONFIG_PATH)
            config = OmegaConf.merge(task_config, config)
        
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
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
        self.vocab = vocab
        
        # Initialize policy
        policy_name = config.MODEL.policy_name
        policy_class = ModelRegistry.get_model(policy_name)
        self.policy = policy_class.from_config(
            config=config,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        self.policy.to(self.device)
        self.policy.eval()
        
        # Load checkpoint
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # Check if checkpoint was trained with use_prev_action
            has_prev_action = 'net.prev_action_embedding.weight' in ckpt['state_dict']
            current_use_prev = config.MODEL.SEQ2SEQ.use_prev_action
            
            if has_prev_action != current_use_prev:
                print(f"Warning: Checkpoint use_prev_action ({has_prev_action}) != config ({current_use_prev})")
                print("Adjusting config to match checkpoint...")
                # Temporarily adjust config to match checkpoint
                config.MODEL.SEQ2SEQ.use_prev_action = has_prev_action
                # Recreate policy with correct config
                policy_class = ModelRegistry.get_model(config.MODEL.policy_name)
                self.policy = policy_class.from_config(
                    config=config,
                    observation_space=env.observation_space,
                    action_space=env.action_space
                )
                self.policy.to(self.device)
                self.policy.eval()
            
            try:
                self.policy.load_state_dict(ckpt['state_dict'], strict=True)
                print(f"Loaded checkpoint: {ckpt_path}")
            except RuntimeError as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Skipping checkpoint loading, using random weights")
        else:
            print(f"Warning: Checkpoint not found: {ckpt_path}")
        
        self.env = env
    
    def analyze_gradient_importance(self, num_samples=10):
        """Analyze gradient magnitudes for different features.
        
        Returns:
            Dict with gradient statistics for each feature
        """
        print("\n" + "=" * 80)
        print("Gradient Importance Analysis")
        print("=" * 80)
        
        self.policy.train()  # Enable gradients
        
        gradient_stats = {
            'instruction': [],
            'rgb': [],
            'prev_action': []
        }
        
        # Collect samples
        samples_collected = 0
        for ep_idx in range(min(num_samples, len(self.env._dataset.episodes))):
            obs = self.env.reset()
            episode = self.env.current_episode
            
            # Tokenize instruction
            from satnav.utils.build_vocab import tokenize_instruction_in_observation
            obs = tokenize_instruction_in_observation(obs, self.vocab, max_length=None, output_format="numpy")
            
            # Initialize RNN state
            # Use policy.net.num_recurrent_layers for compatibility with both Seq2Seq and CMA
            hidden_size = self.config.MODEL.STATE_ENCODER.hidden_size
            
            rnn_state = torch.zeros(
                self.policy.net.num_recurrent_layers, 1, hidden_size,
                device=self.device,
                requires_grad=True
            )
            prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
            not_done_mask = torch.ones(1, 1, device=self.device, dtype=torch.uint8)
            
            # Prepare observation batch
            obs_batch = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_value = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    if key == 'instruction':
                        tensor_value.requires_grad_(True)
                    elif key == 'rgb':
                        tensor_value.requires_grad_(True)
                    obs_batch[key] = tensor_value
            
            if self.config.MODEL.SEQ2SEQ.use_prev_action:
                prev_action.requires_grad_(False)  # Can't compute grad for discrete
            
            # Forward pass
            distribution = self.policy.build_distribution(
                obs_batch,
                rnn_state,
                prev_action,
                not_done_mask
            )
            
            # Compute loss
            # Use a dummy target (we just want gradients)
            target = torch.randint(0, 4, (1, 1), device=self.device)
            loss = F.cross_entropy(distribution.logits, target)
            
            # Backward pass
            self.policy.zero_grad()
            loss.backward()
            
            # Collect gradients
            if 'instruction' in obs_batch and obs_batch['instruction'].grad is not None:
                inst_grad = obs_batch['instruction'].grad.abs().mean().item()
                gradient_stats['instruction'].append(inst_grad)
            
            if 'rgb' in obs_batch and obs_batch['rgb'].grad is not None:
                rgb_grad = obs_batch['rgb'].grad.abs().mean().item()
                gradient_stats['rgb'].append(rgb_grad)
            
            # For prev_action, we can't compute gradient directly
            # Instead, check prev_action embedding gradient
            if self.config.MODEL.SEQ2SEQ.use_prev_action:
                if hasattr(self.policy.net, 'prev_action_embedding'):
                    if self.policy.net.prev_action_embedding.weight.grad is not None:
                        prev_grad = self.policy.net.prev_action_embedding.weight.grad.abs().mean().item()
                        gradient_stats['prev_action'].append(prev_grad)
            
            samples_collected += 1
            if samples_collected >= num_samples:
                break
        
        self.policy.eval()
        
        # Print statistics
        print("\nGradient Statistics (higher = more important):")
        for feature, grads in gradient_stats.items():
            if grads:
                mean_grad = np.mean(grads)
                std_grad = np.std(grads)
                print(f"  {feature:15s}: mean={mean_grad:.6f}, std={std_grad:.6f}")
            else:
                print(f"  {feature:15s}: No gradients collected")
        
        return gradient_stats
    
    def ablation_study(self, num_steps=10):
        """Perform ablation study: remove features one by one.
        
        Returns:
            Dict with prediction differences for each ablation
        """
        print("\n" + "=" * 80)
        print("Ablation Study: Feature Removal Impact")
        print("=" * 80)
        
        results = defaultdict(list)
        
        # Collect baseline predictions
        obs = self.env.reset()
        from satnav.utils.build_vocab import tokenize_instruction_in_observation
        obs = tokenize_instruction_in_observation(obs, self.vocab, max_length=None, output_format="numpy")
        
        # Use policy.net.num_recurrent_layers for compatibility with both Seq2Seq and CMA
        hidden_size = self.config.MODEL.STATE_ENCODER.hidden_size
        
        rnn_state = torch.zeros(
            self.policy.net.num_recurrent_layers, 1, hidden_size,
            device=self.device
        )
        prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        not_done_mask = torch.ones(1, 1, device=self.device, dtype=torch.uint8)
        
        # Baseline prediction
        obs_batch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).unsqueeze(0).to(self.device)
                obs_batch[key] = tensor_value
        
        with torch.no_grad():
            distribution_baseline = self.policy.build_distribution(
                obs_batch, rnn_state, prev_action, not_done_mask
            )
            probs_baseline = distribution_baseline.probs
        
        print(f"\nBaseline prediction probabilities:")
        action_names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
        for i, name in action_names.items():
            print(f"  {name}: {probs_baseline[0, i].item():.4f}")
        
        # Ablation 1: Remove instruction
        print(f"\n1. Ablation: Remove Instruction")
        obs_batch_no_inst = obs_batch.copy()
        if 'instruction' in obs_batch_no_inst:
            # Zero out instruction (but keep shape)
            obs_batch_no_inst['instruction'] = torch.zeros_like(obs_batch_no_inst['instruction'])
            # Keep first token to avoid length=0 error
            obs_batch_no_inst['instruction'][0, 0] = 1
        
        with torch.no_grad():
            distribution_no_inst = self.policy.build_distribution(
                obs_batch_no_inst, rnn_state, prev_action, not_done_mask
            )
            probs_no_inst = distribution_no_inst.probs
        
        diff_no_inst = (probs_baseline - probs_no_inst).abs().sum().item()
        print(f"  Prediction difference: {diff_no_inst:.4f}")
        print(f"  New probabilities:")
        for i, name in action_names.items():
            print(f"    {name}: {probs_no_inst[0, i].item():.4f}")
        results['no_instruction'] = diff_no_inst
        
        # Ablation 2: Remove RGB
        print(f"\n2. Ablation: Remove RGB")
        obs_batch_no_rgb = obs_batch.copy()
        if 'rgb' in obs_batch_no_rgb:
            obs_batch_no_rgb['rgb'] = torch.zeros_like(obs_batch_no_rgb['rgb'])
        
        with torch.no_grad():
            distribution_no_rgb = self.policy.build_distribution(
                obs_batch_no_rgb, rnn_state, prev_action, not_done_mask
            )
            probs_no_rgb = distribution_no_rgb.probs
        
        diff_no_rgb = (probs_baseline - probs_no_rgb).abs().sum().item()
        print(f"  Prediction difference: {diff_no_rgb:.4f}")
        print(f"  New probabilities:")
        for i, name in action_names.items():
            print(f"    {name}: {probs_no_rgb[0, i].item():.4f}")
        results['no_rgb'] = diff_no_rgb
        
        # Ablation 3: Remove prev_action (if enabled)
        if self.config.MODEL.SEQ2SEQ.use_prev_action:
            print(f"\n3. Ablation: Remove Prev_Action")
            # Use a different prev_action (random)
            prev_action_random = torch.randint(1, 4, (1, 1), device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                distribution_diff_prev = self.policy.build_distribution(
                    obs_batch, rnn_state, prev_action_random, not_done_mask
                )
                probs_diff_prev = distribution_diff_prev.probs
            
            diff_prev = (probs_baseline - probs_diff_prev).abs().sum().item()
            print(f"  Prediction difference (prev_action changed): {diff_prev:.4f}")
            print(f"  New probabilities:")
            for i, name in action_names.items():
                print(f"    {name}: {probs_diff_prev[0, i].item():.4f}")
            results['change_prev_action'] = diff_prev
        
        # Ablation 4: Reset RNN state (remove history)
        print(f"\n4. Ablation: Reset RNN State (Remove History)")
        rnn_state_reset = torch.zeros_like(rnn_state)
        
        with torch.no_grad():
            distribution_no_history = self.policy.build_distribution(
                obs_batch, rnn_state_reset, prev_action, not_done_mask
            )
            probs_no_history = distribution_no_history.probs
        
        diff_history = (probs_baseline - probs_no_history).abs().sum().item()
        print(f"  Prediction difference (RNN state reset): {diff_history:.4f}")
        print(f"  New probabilities:")
        for i, name in action_names.items():
            print(f"    {name}: {probs_no_history[0, i].item():.4f}")
        results['reset_rnn_state'] = diff_history
        
        # Summary
        print(f"\n" + "=" * 80)
        print("Feature Importance Ranking (by prediction difference):")
        print("=" * 80)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, diff) in enumerate(sorted_results, 1):
            print(f"  {i}. {feature}: {diff:.4f}")
        
        return results
    
    def analyze_feature_contributions(self, num_steps=20):
        """Analyze feature contributions across multiple steps.
        
        Returns:
            Dict with feature contribution statistics
        """
        print("\n" + "=" * 80)
        print("Feature Contribution Analysis Across Steps")
        print("=" * 80)
        
        contributions = {
            'instruction': [],
            'rgb': [],
            'prev_action': [],
            'rnn_history': []
        }
        
        obs = self.env.reset()
        from satnav.utils.build_vocab import tokenize_instruction_in_observation
        obs = tokenize_instruction_in_observation(obs, self.vocab, max_length=None, output_format="numpy")
        
        # Use policy.net.num_recurrent_layers for compatibility with both Seq2Seq and CMA
        hidden_size = self.config.MODEL.STATE_ENCODER.hidden_size
        
        rnn_state = torch.zeros(
            self.policy.net.num_recurrent_layers, 1, hidden_size,
            device=self.device
        )
        prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        not_done_mask = torch.ones(1, 1, device=self.device, dtype=torch.uint8)
        
        for step in range(num_steps):
            # Prepare observation batch
            obs_batch = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_value = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    obs_batch[key] = tensor_value
            
            # Baseline prediction
            with torch.no_grad():
                distribution_baseline = self.policy.build_distribution(
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
                distribution_no_inst = self.policy.build_distribution(
                    obs_batch_no_inst, rnn_state, prev_action, not_done_mask
                )
                probs_no_inst = distribution_no_inst.probs
                action_no_inst = distribution_no_inst.mode().item()
            
            inst_diff = (probs_baseline - probs_no_inst).abs().sum().item()
            inst_action_change = 1 if action_baseline != action_no_inst else 0
            contributions['instruction'].append({
                'prob_diff': inst_diff,
                'action_changed': inst_action_change
            })
            
            # Test removing RGB
            obs_batch_no_rgb = obs_batch.copy()
            if 'rgb' in obs_batch_no_rgb:
                obs_batch_no_rgb['rgb'] = torch.zeros_like(obs_batch_no_rgb['rgb'])
            
            with torch.no_grad():
                distribution_no_rgb = self.policy.build_distribution(
                    obs_batch_no_rgb, rnn_state, prev_action, not_done_mask
                )
                probs_no_rgb = distribution_no_rgb.probs
                action_no_rgb = distribution_no_rgb.mode().item()
            
            rgb_diff = (probs_baseline - probs_no_rgb).abs().sum().item()
            rgb_action_change = 1 if action_baseline != action_no_rgb else 0
            contributions['rgb'].append({
                'prob_diff': rgb_diff,
                'action_changed': rgb_action_change
            })
            
            # Test changing prev_action
            if self.config.MODEL.SEQ2SEQ.use_prev_action:
                prev_action_alt = torch.randint(1, 4, (1, 1), device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    distribution_diff_prev = self.policy.build_distribution(
                        obs_batch, rnn_state, prev_action_alt, not_done_mask
                    )
                    probs_diff_prev = distribution_diff_prev.probs
                    action_diff_prev = distribution_diff_prev.mode().item()
                
                prev_diff = (probs_baseline - probs_diff_prev).abs().sum().item()
                prev_action_change = 1 if action_baseline != action_diff_prev else 0
                contributions['prev_action'].append({
                    'prob_diff': prev_diff,
                    'action_changed': prev_action_change
                })
            
            # Test resetting RNN state (removing history)
            rnn_state_reset = torch.zeros_like(rnn_state)
            
            with torch.no_grad():
                distribution_no_history = self.policy.build_distribution(
                    obs_batch, rnn_state_reset, prev_action, not_done_mask
                )
                probs_no_history = distribution_no_history.probs
                action_no_history = distribution_no_history.mode().item()
            
            history_diff = (probs_baseline - probs_no_history).abs().sum().item()
            history_action_change = 1 if action_baseline != action_no_history else 0
            contributions['rnn_history'].append({
                'prob_diff': history_diff,
                'action_changed': history_action_change
            })
            
            # Execute action and update
            action_idx = action_baseline
            obs, done, info = self.env.step(action_idx)
            obs = tokenize_instruction_in_observation(obs, self.vocab, max_length=None, output_format="numpy")
            
            prev_action = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
            not_done_mask = torch.tensor(
                [[0] if done else [1]],
                dtype=torch.uint8,
                device=self.device
            )
            
            if done:
                break
        
        # Print statistics
        print("\nFeature Contribution Statistics:")
        print("-" * 80)
        for feature, contribs in contributions.items():
            if contribs:
                avg_prob_diff = np.mean([c['prob_diff'] for c in contribs])
                action_change_rate = np.mean([c['action_changed'] for c in contribs])
                print(f"\n{feature}:")
                print(f"  Average probability difference: {avg_prob_diff:.4f}")
                print(f"  Action change rate: {action_change_rate:.2%}")
                print(f"  Interpretation: {'High' if avg_prob_diff > 0.1 else 'Low'} impact on predictions")
        
        return contributions


def main():
    """Run feature importance analysis."""
    config_path = 'configs/baselines/seq2seq.yaml'
    ckpt_path = 'output/checkpoints/seq2seq/best.pth'
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        print("Please train a model first or specify a valid checkpoint path.")
        return
    
    analyzer = FeatureImportanceAnalyzer(config_path, ckpt_path)
    
    # Run analyses
    print("=" * 80)
    print("Feature Importance Analysis")
    print("=" * 80)
    
    # 1. Ablation study
    ablation_results = analyzer.ablation_study()
    
    # 2. Feature contribution analysis
    contributions = analyzer.analyze_feature_contributions()
    
    # 3. Gradient analysis (if possible)
    try:
        gradient_stats = analyzer.analyze_gradient_importance(num_samples=5)
    except Exception as e:
        print(f"\nGradient analysis failed: {e}")
        print("(This is okay, gradient analysis requires training mode)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Which Features Does the Model Rely On?")
    print("=" * 80)
    
    print("\nBased on ablation study:")
    sorted_ablations = sorted(ablation_results.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, diff) in enumerate(sorted_ablations, 1):
        importance = "HIGH" if diff > 0.1 else "MEDIUM" if diff > 0.05 else "LOW"
        print(f"  {i}. {feature}: {diff:.4f} ({importance} importance)")
    
    print("\nConclusion:")
    if ablation_results.get('change_prev_action', 0) > ablation_results.get('no_instruction', 0) and \
       ablation_results.get('change_prev_action', 0) > ablation_results.get('no_rgb', 0):
        print("  ✓ Model relies MOST on prev_action")
        print("  ✗ Model relies LESS on RGB and instruction")
        print("  → This confirms: Model uses prev_action as shortcut!")
    elif ablation_results.get('no_rgb', 0) > ablation_results.get('no_instruction', 0):
        print("  ✓ Model relies MORE on RGB than instruction")
        print("  → Model uses visual cues but ignores language")
    else:
        print("  → Model uses multiple features")


if __name__ == "__main__":
    main()


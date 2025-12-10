#!/usr/bin/env python3
"""Evaluation module for SatNav VLN tasks.

This module provides evaluation functionality for trained models,
including checkpoint evaluation, metrics collection, and video generation.

Reference: VLN-CE vlnce_baselines/common/base_il_trainer.py (_eval_checkpoint)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from satnav.core.env import Env
from satnav.dataset.recollect_dataset import RecollectionDataset
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.utils.build_vocab import VocabDict, build_vocab_from_dataset


class Evaluator:
    """Evaluator for VLN models.
    
    This class handles evaluation of trained models, including:
    - Loading checkpoints and initializing policies
    - Running evaluation episodes
    - Collecting and aggregating metrics
    - Generating videos (optional)
    - Saving results to JSON
    
    Attributes:
        config: Evaluation configuration
        device: torch.device for computation
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        """Initialize the evaluator.
        
        Args:
            config: Configuration object
            device: Device for computation
        """
        self.config = config
        self.device = device
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        policy: torch.nn.Module,
        checkpoint_index: int = 0
    ) -> Dict[str, float]:
        """Evaluate a single checkpoint with full environment rollouts.
        
        This method:
        1. Loads checkpoint and initializes policy
        2. Creates evaluation environment
        3. Runs episodes and collects metrics
        4. Optionally generates videos
        5. Saves aggregated results to JSON
        
        Args:
            checkpoint_path: Path to checkpoint file
            policy: Policy network to evaluate (will load weights from checkpoint)
            checkpoint_index: Index for logging purposes
            
        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\nEvaluating checkpoint: {checkpoint_path}")
        
        # Get evaluation config
        split = OmegaConf.select(self.config, 'EVAL.SPLIT', default='val_seen')
        episode_count = OmegaConf.select(self.config, 'EVAL.EPISODE_COUNT', default=-1)
        save_results = OmegaConf.select(self.config, 'EVAL.SAVE_RESULTS', default=True)
        
        # Video generation: support both boolean (GENERATE_VIDEOS) and legacy (VIDEO_OPTION) formats
        generate_videos = OmegaConf.select(self.config, 'GENERATE_VIDEOS', default=False)
        video_option = OmegaConf.select(self.config, 'VIDEO_OPTION', default=None)
        
        # Handle legacy VIDEO_OPTION format (list) for backward compatibility
        if video_option is not None:
            # Legacy format: VIDEO_OPTION is a list
            video_enabled = isinstance(video_option, list) and len(video_option) > 0 and 'disk' in video_option
        else:
            # New format: GENERATE_VIDEOS is a boolean
            video_enabled = bool(generate_videos)
        
        video_dir = OmegaConf.select(self.config, 'VIDEO_DIR', default='data/videos/default')
        
        # Handle None values
        if split is None:
            split = 'val_seen'
        if episode_count is None:
            episode_count = -1
        if save_results is None:
            save_results = True
        
        print(f"Evaluation configuration:")
        print(f"  Split: {split}")
        print(f"  Episode count: {episode_count if episode_count > 0 else 'all'}")
        print(f"  Save results: {save_results}")
        print(f"  Video generation: {'Enabled' if video_enabled else 'Disabled'}")
        
        # ===================================================================
        # 1. Load checkpoint
        # ===================================================================
        print(f"\nLoading checkpoint...")
        try:
            ckpt = self._load_checkpoint(checkpoint_path)
            print(f"  Checkpoint info: Epoch {ckpt.get('epoch', 'N/A')}, "
                  f"Step {ckpt.get('step_id', 'N/A')}, "
                  f"Loss {ckpt.get('loss', 'N/A')}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Load checkpoint weights
        policy.load_state_dict(ckpt['state_dict'])
        policy.eval()
        print(f"  Policy loaded and set to eval mode")
        
        # ===================================================================
        # 2. Create evaluation environment
        # ===================================================================
        print(f"\nCreating evaluation environment...")
        
        # Create dataset with synced DATASET.SPLIT
        dataset = SatNavDataset(self.config.DATASET)
        num_episodes_total = len(dataset.episodes)
        
        # Determine number of episodes to evaluate
        if episode_count > 0:
            num_episodes = min(episode_count, num_episodes_total)
        else:
            num_episodes = num_episodes_total
        
        print(f"  Dataset: {num_episodes_total} episodes ({split} split)")
        print(f"  Evaluating: {num_episodes} episodes")
        
        # Create environment
        env = Env(self.config, dataset=dataset, cycle=False)
        max_steps = env.max_episode_steps
        print(f"  Max steps per episode: {max_steps}")
        
        # Create video directory if needed
        if video_enabled:
            video_dir = Path(video_dir)  # Convert string to Path object
            video_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Video directory: {video_dir}")
        else:
            video_dir = None
        
        # ===================================================================
        # 3. Load vocabulary for instruction tokenization
        # ===================================================================
        vocab = self._load_vocabulary()
        
        # ===================================================================
        # 4. Run evaluation episodes
        # ===================================================================
        print(f"\nEvaluating Episodes...")
        print("=" * 80)
        
        episode_metrics = {}
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm.tqdm(total=num_episodes, desc=f"[Eval {split}]", dynamic_ncols=True)
        
        for ep_idx in range(num_episodes):
            try:
                metrics = self._evaluate_single_episode(
                    env=env,
                    policy=policy,
                    vocab=vocab,
                    max_steps=max_steps,
                    video_enabled=video_enabled,
                    video_dir=video_dir,  # Already converted to Path or None
                    checkpoint_index=checkpoint_index
                )
                
                if metrics:
                    ep_id = metrics['episode_id']
                    episode_metrics[ep_id] = {
                        'spl': metrics['spl'],
                        'success': metrics['success'],
                        'distance_to_goal': metrics['distance_to_goal'],
                        'path_length': metrics['path_length'],
                        'steps_taken': metrics['steps_taken'],
                    }
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'SPL': f"{episode_metrics[ep_id]['spl']:.3f}",
                        'Success': f"{episode_metrics[ep_id]['success']:.0f}"
                    })
                else:
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n  Error during episode {ep_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next episode
                pbar.update(1)
                continue
        
        pbar.close()
        
        # ===================================================================
        # 5. Aggregate metrics
        # ===================================================================
        if len(episode_metrics) == 0:
            print("\nError: No episodes were successfully evaluated")
            return {}
        
        aggregated_metrics = self._aggregate_metrics(episode_metrics, split, checkpoint_index)
        
        # ===================================================================
        # 6. Save results to JSON
        # ===================================================================
        if save_results:
            results_file = self._save_results(aggregated_metrics, checkpoint_index, split)
            print(f"\nResults saved to: {results_file}")
        
        # ===================================================================
        # 7. Print summary
        # ===================================================================
        self._print_summary(aggregated_metrics, len(episode_metrics), start_time, checkpoint_index)
        
        return aggregated_metrics
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    
    def _load_vocabulary(self) -> VocabDict:
        """Load vocabulary for instruction tokenization.
        
        Returns:
            VocabDict instance
        """
        # Try to load vocab from file if specified
        vocab_file = OmegaConf.select(self.config, 'DATASET.vocab_file', default=None)
        if vocab_file and os.path.exists(vocab_file):
            return VocabDict.load(vocab_file)
        else:
            # Build from dataset
            dataset_path = self.config.DATASET.DATA_PATH.format(split=self.config.DATASET.SPLIT)
            return build_vocab_from_dataset(dataset_path)
    
    def _tokenize_instruction(self, obs: Dict[str, Any], vocab: VocabDict) -> Dict[str, Any]:
        """Tokenize instruction in observation.
        
        Args:
            obs: Observation dictionary with instruction
            vocab: Vocabulary object for tokenization
            
        Returns:
            Modified observation with tokenized instruction as numpy array
        """
        from satnav.utils.build_vocab import tokenize_instruction_in_observation
        
        # Use unified tokenization function (numpy format for evaluation)
        return tokenize_instruction_in_observation(obs, vocab, max_length=None, output_format="numpy")
    
    def _evaluate_single_episode(
        self,
        env: Env,
        policy: torch.nn.Module,
        vocab: VocabDict,
        max_steps: int,
        video_enabled: bool,
        video_dir: Optional[Path],
        checkpoint_index: int
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single episode.
        
        Args:
            env: Environment instance
            policy: Policy network
            vocab: Vocabulary for tokenization
            max_steps: Maximum steps per episode
            video_enabled: Whether to collect video frames
            video_dir: Directory to save videos (if enabled)
            checkpoint_index: Checkpoint index for video naming
            
        Returns:
            Dictionary with episode metrics, or None if evaluation failed
        """
        # Import video-related modules at function start (not in loop)
        if video_enabled:
            from satnav.utils.maps import annotate_topdown_map
            from satnav.utils.examples import prepare_waypoints
            from satnav.core.utils import geodesic_distance
        
        # Reset environment
        obs = env.reset()
        episode = env.current_episode
        done = False
        step_count = 0
        
        # Tokenize instruction in initial observation
        obs = self._tokenize_instruction(obs, vocab)
        
        # Collect frames for video
        rgb_frames = []
        topdown_frames = []
        
        # Prepare waypoints once (for video annotation)
        if video_enabled:
            waypoints = prepare_waypoints(episode)
            action_names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
        
        # Initialize RNN state for this episode
        state_encoder = policy.net.state_encoder
        rnn_num_layers = state_encoder.rnn.num_layers
        hidden_size = self.config.MODEL.STATE_ENCODER.hidden_size
        
        rnn_state = torch.zeros(
            rnn_num_layers, 1, hidden_size,
            device=self.device
        )
        prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        not_done_mask = torch.ones(1, 1, device=self.device, dtype=torch.uint8)
        
        # Episode rollout loop
        while not done and step_count < max_steps:
            # Prepare observation batch
            obs_batch = {}
            for key, value in obs.items():
                # Handle different observation types
                if isinstance(value, np.ndarray):
                    # Convert numpy to tensor and add batch dimension
                    tensor_value = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    obs_batch[key] = tensor_value
            
            # Get action from policy
            with torch.no_grad():
                actions, rnn_state = policy.act(
                    obs_batch,
                    rnn_state,
                    prev_action,
                    not_done_mask,
                    deterministic=True  # Use greedy decoding for evaluation
                )
            
            # Execute action
            action_idx = actions[0].item()
            obs, done, info = env.step(action_idx)
            
            # Tokenize instruction in new observation
            obs = self._tokenize_instruction(obs, vocab)
            
            # Collect frames for video (after step, matching satnav_path_follower_example.py)
            if video_enabled:
                # Collect RGB frame (after step, like example)
                if 'rgb' in obs:
                    rgb_frames.append(obs['rgb'].copy())
                
                # Collect topdown frame using annotate_topdown_map (after step, like example)
                try:
                    # Get agent state
                    agent_state = env._task._sim.get_agent_state()
                    
                    # Calculate distance to goal (use first waypoint as current target for simplicity)
                    # Note: In evaluation, we don't track waypoint progress, so we use goal distance
                    current_waypoint_idx = 0  # For visualization purposes only
                    if waypoints:
                        current_distance = geodesic_distance(
                            agent_state.position,
                            waypoints[-1]  # Use last waypoint (goal) for distance calculation
                        )
                    else:
                        current_distance = 0.0
                    
                    # Get action name for display
                    action_name = action_names.get(action_idx, "UNKNOWN")
                    
                    # Annotate top-down map (this function handles extracting map from info and annotating it)
                    # It will append to topdown_frames if successful
                    topdown_frame_count_before = len(topdown_frames)
                    annotated_map = annotate_topdown_map(
                        info=info,
                        agent_state=agent_state,
                        waypoints=waypoints,
                        current_waypoint_idx=current_waypoint_idx,
                        step_count=step_count,
                        action=action_name,
                        current_distance=current_distance,
                        goal_radius=self.config.TASK.SUCCESS_DISTANCE,
                        config=self.config,
                        topdown_frames=topdown_frames  # Function will append to this list
                    )
                    
                    # If annotate_topdown_map didn't append (returned None or failed), fallback to direct extraction
                    if len(topdown_frames) == topdown_frame_count_before:
                        # Fallback: extract map directly
                        if 'top_down_map' in info.get('metrics', {}):
                            topdown_map_data = info['metrics']['top_down_map']
                            if isinstance(topdown_map_data, dict) and 'map' in topdown_map_data:
                                map_array = topdown_map_data['map']
                                if isinstance(map_array, np.ndarray):
                                    topdown_frames.append(map_array.copy())
                except Exception as e:
                    # Fallback to simple extraction if annotation fails
                    if 'top_down_map' in info.get('metrics', {}):
                        topdown_map_data = info['metrics']['top_down_map']
                        if isinstance(topdown_map_data, dict) and 'map' in topdown_map_data:
                            map_array = topdown_map_data['map']
                            if isinstance(map_array, np.ndarray):
                                topdown_frames.append(map_array.copy())
            
            # Update state
            prev_action = actions
            not_done_mask = torch.tensor(
                [[0] if done else [1]],
                dtype=torch.uint8,
                device=self.device
            )
            step_count += 1
        
        # Get final metrics
        metrics = env.get_metrics()
        
        # Generate video if enabled
        if video_enabled and rgb_frames:
            self._generate_video(
                rgb_frames=rgb_frames,
                topdown_frames=topdown_frames,
                episode=episode,
                video_dir=video_dir,
                checkpoint_index=checkpoint_index
            )
        
        # Return episode metrics
        ep_id = episode.episode_id
        return {
            'episode_id': ep_id,
            'spl': float(metrics.get('spl', 0.0)),
            'success': float(metrics.get('success', 0.0)),
            'distance_to_goal': float(metrics.get('distance_to_goal', 0.0)),
            'path_length': float(metrics.get('path_length', 0.0)),
            'steps_taken': step_count,
        }
    
    def _generate_video(
        self,
        rgb_frames: list,
        topdown_frames: list,
        episode: Any,
        video_dir: Path,
        checkpoint_index: int
    ) -> None:
        """Generate video for an episode.
        
        Args:
            rgb_frames: List of RGB frames
            topdown_frames: List of topdown map frames
            episode: Episode object
            video_dir: Directory to save video
            checkpoint_index: Checkpoint index for naming
        """
        from satnav.utils.examples import generate_video
        
        ep_id = episode.episode_id
        video_path = video_dir / f"episode_{ep_id}_ckpt_{checkpoint_index}.mp4"
        
        try:
            # Only generate video if we have both RGB and topdown frames
            if topdown_frames and len(topdown_frames) == len(rgb_frames):
                generate_video(
                    rgb_frames=rgb_frames,
                    topdown_frames=topdown_frames,
                    instruction_text=episode.instruction.instruction_text,
                    output_path=video_path,
                    fps=5,
                    frame_width=2048
                )
            else:
                print(f"  ⚠ Skipping video for {ep_id}: topdown frames unavailable "
                      f"(enable TOP_DOWN_MAP in TASK.MEASUREMENTS)")
        except Exception as e:
            print(f"  ⚠ Video generation failed for {ep_id}: {e}")
    
    def _aggregate_metrics(
        self,
        episode_metrics: Dict[str, Dict[str, float]],
        split: str,
        checkpoint_index: int
    ) -> Dict[str, float]:
        """Aggregate metrics across episodes.
        
        Args:
            episode_metrics: Dictionary mapping episode_id to metrics
            split: Dataset split name
            checkpoint_index: Checkpoint index
            
        Returns:
            Dictionary of aggregated metrics
        """
        print(f"\n" + "=" * 80)
        print(f"Aggregating metrics from {len(episode_metrics)} episodes...")
        
        aggregated_metrics = {}
        
        # Get all metric names from first episode
        first_ep_metrics = next(iter(episode_metrics.values()))
        
        for metric_name in first_ep_metrics.keys():
            values = [ep_metrics[metric_name] for ep_metrics in episode_metrics.values()]
            aggregated_metrics[metric_name] = sum(values) / len(values)
        
        # Add metadata
        aggregated_metrics['num_episodes'] = len(episode_metrics)
        aggregated_metrics['split'] = split
        aggregated_metrics['checkpoint_index'] = checkpoint_index
        
        return aggregated_metrics
    
    def _save_results(
        self,
        aggregated_metrics: Dict[str, float],
        checkpoint_index: int,
        split: str
    ) -> Path:
        """Save evaluation results to JSON file.
        
        Args:
            aggregated_metrics: Aggregated metrics dictionary
            checkpoint_index: Checkpoint index
            split: Dataset split name
            
        Returns:
            Path to saved results file
        """
        results_dir = Path(self.config.get('RESULTS_DIR', 'data/results/default'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"eval_ckpt_{checkpoint_index}_{split}.json"
        
        with open(results_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=4)
        
        return results_file
    
    def _print_summary(
        self,
        aggregated_metrics: Dict[str, float],
        num_episodes: int,
        start_time: float,
        checkpoint_index: int
    ) -> None:
        """Print evaluation summary.
        
        Args:
            aggregated_metrics: Aggregated metrics dictionary
            num_episodes: Number of episodes evaluated
            start_time: Start time of evaluation
            checkpoint_index: Checkpoint index
        """
        print("\n" + "=" * 80)
        print(f"Evaluation Summary (Checkpoint {checkpoint_index})")
        print("=" * 80)
        print(f"Episodes evaluated: {num_episodes}")
        print(f"Time elapsed: {time.time() - start_time:.2f}s")
        print(f"\nMetrics:")
        for metric_name, value in aggregated_metrics.items():
            if metric_name not in ['num_episodes', 'split', 'checkpoint_index']:
                print(f"  {metric_name}: {value:.4f}")
        print("=" * 80)


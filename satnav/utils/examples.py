#!/usr/bin/env python3
"""Utility functions for example scripts.

This module provides helper functions to keep example scripts focused on
core logic rather than implementation details.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from satnav.core import Env
from satnav.core.config import load_config
from satnav.core.episode import VLNEpisode
from satnav.dataset.satnav_dataset import SatNavDataset
from omegaconf import DictConfig


def setup_example(
    config_path: Path,
    output_dir: Path
) -> Tuple[DictConfig, SatNavDataset, Env]:
    """Setup configuration, dataset, and environment for an example.
    
    Args:
        config_path: Path to configuration YAML file.
        output_dir: Output directory for saving results.
        
    Returns:
        Tuple of (config, dataset, env).
        
    Raises:
        FileNotFoundError: If config file not found.
        Exception: If dataset or environment creation fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Create dataset
    dataset = SatNavDataset(config.DATASET)
    
    # Create environment
    env = Env(config, dataset=dataset, cycle=False)
    
    return config, dataset, env


def print_episode_info(episode: VLNEpisode) -> None:
    """Print episode information.
    
    Args:
        episode: VLN episode to print information for.
    """
    print(f"Episode ID: {episode.episode_id}")
    print(f"Scene ID: {episode.scene_id}")
    print(f"Start Position: [{episode.start_position[0]:.6f}, {episode.start_position[1]:.6f}, {episode.start_position[2]:.1f}]")
    print(f"Start Rotation: {episode.start_rotation}°")
    print(f"Goal Position: [{episode.goals[0].position[0]:.6f}, {episode.goals[0].position[1]:.6f}, {episode.goals[0].position[2]:.1f}]")
    print(f"Instruction: {episode.instruction.instruction_text}")


def prepare_waypoints(episode: VLNEpisode) -> List[List[float]]:
    """Prepare waypoints from episode reference path.
    
    Extracts waypoints from reference path (excluding start) and ensures
    goal is included.
    
    Args:
        episode: VLN episode with reference path.
        
    Returns:
        List of waypoints [[lon, lat, alt], ...].
    """
    from satnav.core.utils import geodesic_distance
    
    waypoints = []
    if episode.reference_path and len(episode.reference_path) > 1:
        # Skip the first point (start position), include all intermediate waypoints
        waypoints = episode.reference_path[1:]
    
    # Ensure goal is included (if not already the last waypoint)
    goal_position = episode.goals[0].position
    if not waypoints or not np.allclose(waypoints[-1], goal_position, atol=1e-6, rtol=0):
        waypoints.append(goal_position)
    
    return waypoints


def print_waypoints_info(episode: VLNEpisode, waypoints: List[List[float]]) -> None:
    """Print waypoints information.
    
    Args:
        episode: VLN episode.
        waypoints: List of waypoints.
    """
    from satnav.core.utils import geodesic_distance
    
    print(f"Reference Path: {len(episode.reference_path) if episode.reference_path else 0} waypoints")
    print(f"Navigation Waypoints: {len(waypoints)} (excluding start)")
    for i, wp in enumerate(waypoints):
        wp_type = "Goal" if i == len(waypoints) - 1 else f"Waypoint {i+1}"
        print(f"  {wp_type}: [{wp[0]:.6f}, {wp[1]:.6f}, {wp[2]:.1f}]")
    
    # Calculate initial distance to first waypoint
    if waypoints:
        initial_distance = geodesic_distance(
            episode.start_position,
            waypoints[0]
        )
        print(f"Initial Distance to First Waypoint: {initial_distance:.2f}m")


def generate_video(
    rgb_frames: List[np.ndarray],
    topdown_frames: List[np.ndarray],
    instruction_text: str,
    output_path: Path,
    fps: int = 5,
    frame_width: int = 2048
) -> bool:
    """Generate video from frames.
    
    Args:
        rgb_frames: List of RGB observation frames.
        topdown_frames: List of top-down map frames.
        instruction_text: Instruction text to display.
        output_path: Path to save video file.
        fps: Frames per second.
        frame_width: Target frame width.
        
    Returns:
        True if video was generated successfully, False otherwise.
    """
    if not rgb_frames or not topdown_frames:
        return False
    
    try:
        from satnav.utils import make_video
        
        # Ensure we have the same number of frames
        min_frames = min(len(rgb_frames), len(topdown_frames))
        rgb_frames_trimmed = rgb_frames[:min_frames]
        topdown_frames_trimmed = topdown_frames[:min_frames]
        
        make_video(
            rgb_frames=rgb_frames_trimmed,
            topdown_frames=topdown_frames_trimmed,
            instruction_text=instruction_text,
            output_path=output_path,
            fps=fps,
            frame_width=frame_width
        )
        return True
    except ImportError as e:
        print(f"\n⚠ Video generation skipped: {e}")
        print("  Install imageio and imageio-ffmpeg to enable video generation:")
        print("    pip install imageio imageio-ffmpeg")
        return False
    except Exception as e:
        print(f"\n⚠ Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_visualizations(
    env: Env,
    obs: Dict[str, Any],
    output_dir: Path,
    prefix: str = "satnav_follower"
) -> None:
    """Save visualization images.
    
    Args:
        env: Environment instance.
        obs: Final observations.
        output_dir: Output directory.
        prefix: Prefix for output filenames.
    """
    try:
        metrics = env.get_metrics()
        
        if "top_down_map" in metrics:
            top_down_info = metrics["top_down_map"]
            map_image = top_down_info["map"]
            
            # Convert RGB to BGR for OpenCV
            map_bgr = cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR)
            
            # Save top-down map
            output_path = output_dir / f"topdown_map_{prefix}.png"
            cv2.imwrite(str(output_path), map_bgr)
            print(f"✓ Top-down map saved to: {output_path}")
            print(f"  Map size: {map_image.shape[1]}x{map_image.shape[0]}")
        else:
            print("  Top-down map not available (measure not enabled)")
        
        # Save RGB observation
        if "rgb" in obs:
            rgb_image = obs["rgb"]
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_path = output_dir / f"final_observation_{prefix}.png"
            cv2.imwrite(str(rgb_path), rgb_bgr)
            print(f"✓ Final observation saved to: {rgb_path}")
            
    except Exception as e:
        print(f"⚠ Error saving visualization: {e}")
        import traceback
        traceback.print_exc()


def print_metrics(
    env: Env,
    episode: VLNEpisode,
    action_history: List[str],
    step_count: int,
    note: Optional[str] = None
) -> None:
    """Print evaluation metrics.
    
    Args:
        env: Environment instance.
        episode: VLN episode.
        action_history: List of actions taken.
        step_count: Number of steps taken.
        note: Optional note to print at the end.
    """
    from satnav.core.utils import geodesic_distance
    
    try:
        metrics = env.get_metrics()
        
        success = metrics.get('success', 0.0)
        spl = metrics.get('spl', 0.0)
        distance = metrics.get('distance_to_goal', 0.0)
        path_length = metrics.get('path_length', 0.0)
        
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics")
        print("=" * 60)
        print(f"Success:           {success:.4f} {'✓' if success > 0 else '✗'}")
        print(f"SPL:               {spl:.4f}")
        print(f"Distance to Goal:  {distance:.2f} m")
        print(f"Path Length:       {path_length:.2f} m")
        
        # Compare with reference path if available
        if episode.reference_path:
            # Calculate reference path length
            ref_length = 0.0
            for i in range(len(episode.reference_path) - 1):
                ref_length += geodesic_distance(
                    episode.reference_path[i],
                    episode.reference_path[i + 1]
                )
            print(f"Reference Path:    {ref_length:.2f} m ({len(episode.reference_path)} waypoints)")
            
            # Calculate efficiency (path_length / reference_path_length)
            if ref_length > 0:
                efficiency = path_length / ref_length
                print(f"Path Efficiency:   {efficiency:.2f}x (lower is better)")
        
        # Action distribution
        action_counts = {}
        for a in action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        print(f"Action Distribution: {action_counts}")
        
        print("=" * 60)
        
        # Summary
        if success > 0:
            print("\n🎉 Navigation successful! Agent reached the goal.")
        else:
            print(f"\n⚠ Navigation incomplete. Final distance: {distance:.2f}m")
        
        # Optional note
        if note:
            print(f"\n📝 {note}")
            
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
        import traceback
        traceback.print_exc()

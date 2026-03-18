#!/usr/bin/env python3
"""Example script for running VLN episodes using SatNav with SatNavPathFollower.

This script demonstrates how to:
1. Load configuration from vln_task.yaml
2. Create a dataset from test data
3. Create an environment
4. Run all episodes with SatNavPathFollower (direct goal navigation)
5. Generate videos for each episode (optional)
6. Save evaluation metrics to JSON file

The key difference from ReferencePathFollower:
- SatNavPathFollower navigates directly to the goal position
- It doesn't follow a reference path, but uses a greedy strategy
- Useful for oracle action generation and shortest path navigation

Usage:
    python examples/satnav_path_follower_example.py [--no-video]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from satnav.core.config import get_success_distance_default
from satnav.core.utils import geodesic_distance
from satnav.navigation import SatNavPathFollower
from satnav.utils.maps import annotate_topdown_map
from satnav.utils.examples import (
    setup_example,
    prepare_waypoints,
    generate_video,
)


def get_success_distance(config, trajectory_type: str) -> float:
    """Get SUCCESS_DISTANCE based on trajectory type.
    
    Args:
        config: Configuration object.
        trajectory_type: Type of trajectory ('Boundary', 'LandmarkSet', or 'Road').
        
    Returns:
        SUCCESS_DISTANCE value.
    """
    sd_config = config.TASK.SUCCESS_DISTANCE
    
    # Old format: single value
    if isinstance(sd_config, (int, float)):
        return float(sd_config)
    
    # New format: dict with type-specific values
    if trajectory_type and hasattr(sd_config, trajectory_type):
        return float(getattr(sd_config, trajectory_type))
    elif hasattr(sd_config, "DEFAULT"):
        return float(sd_config.DEFAULT)
    else:
        return 10.0  # Fallback


def run_single_episode(
    env,
    path_follower,
    config,
    output_dir,
    save_video,
    episode_idx,
    total_episodes
):
    """Run a single episode and return metrics.
    
    Returns:
        dict: Episode metrics including episode_id, scene_id, instruction, and all metrics.
    """
    try:
        # Reset environment and get initial observations
        obs = env.reset()
        episode = env.current_episode
        
        # Update goal_radius based on trajectory_type
        trajectory_type = getattr(episode, 'trajectory_type', None)
        goal_radius = get_success_distance(config, trajectory_type)
        path_follower.goal_radius = goal_radius
        
        print(f"\n[{episode_idx+1}/{total_episodes}] Running episode: {episode.episode_id}")
        print(f"  Scene: {episode.scene_id}")
        print(f"  Type: {trajectory_type} (goal_radius={goal_radius}m)")
        print(f"  Instruction: {episode.instruction.instruction_text[:80]}...")
        
        # Prepare waypoints from reference path
        waypoints = prepare_waypoints(episode)
        
        # Run episode loop - navigate through waypoints sequentially
        done = False
        step_count = 0
        action_history = []
        current_waypoint_idx = 0
        
        # Store frames for video generation
        rgb_frames = []
        topdown_frames = []
        
        while not done and step_count < env.max_episode_steps and current_waypoint_idx < len(waypoints):
            # Get current waypoint target
            current_waypoint = waypoints[current_waypoint_idx]
            is_final_waypoint = (current_waypoint_idx == len(waypoints) - 1)
            
            # Get next action from SatNavPathFollower for current waypoint
            action = path_follower.get_next_action(current_waypoint, env._task._sim)
            
            # Key fix: Only execute STOP at the final waypoint (goal)
            # For intermediate waypoints, if path_follower returns STOP (meaning we've
            # reached that waypoint), we should move to the next waypoint instead of stopping
            if action == "STOP" and not is_final_waypoint:
                # Reached intermediate waypoint, move to next
                current_waypoint_idx += 1
                continue  # Skip executing STOP, continue with next waypoint
            
            action_history.append(action)
            
            # Execute action
            obs, done, info = env.step(action)
            
            # Store RGB frame for video
            if save_video and "rgb" in obs:
                rgb_frames.append(obs["rgb"].copy())
            
            # Get current agent state
            agent_state = env._task._sim.get_agent_state()
            
            # Check if we've reached the current waypoint
            current_distance = geodesic_distance(
                agent_state.position,
                current_waypoint
            )
            
            # Check if we've reached the current waypoint (within goal_radius)
            reached_waypoint = current_distance <= path_follower.goal_radius
            
            # Annotate top-down map visualization
            annotate_topdown_map(
                info=info,
                agent_state=agent_state,
                waypoints=waypoints,
                current_waypoint_idx=current_waypoint_idx,
                step_count=step_count,
                action=action,
                current_distance=current_distance,
                goal_radius=path_follower.goal_radius,
                config=config,
                topdown_frames=topdown_frames if save_video else None
            )
            
            step_count += 1
            
            # If reached current waypoint, move to next
            if reached_waypoint:
                if current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
                # else: at final waypoint and STOP was executed, loop will exit via done=True
            
            # At final waypoint, STOP has been executed, done=True, loop will exit
        
        # Generate video if requested
        if save_video and rgb_frames and topdown_frames:
            video_filename = f"episode_{episode.episode_id}_video.mp4"
            video_path = output_dir / video_filename
            generate_video(
                rgb_frames=rgb_frames,
                topdown_frames=topdown_frames,
                instruction_text=episode.instruction.instruction_text,
                output_path=video_path,
                fps=5,
                frame_width=2048
            )
            print(f"  ✓ Video saved: {video_filename}")
        
        # Get final metrics
        metrics = env.get_metrics()
        
        # Calculate reference path length if available
        ref_length = 0.0
        if episode.reference_path:
            for i in range(len(episode.reference_path) - 1):
                ref_length += geodesic_distance(
                    episode.reference_path[i],
                    episode.reference_path[i + 1]
                )
        
        # Calculate efficiency
        path_length = metrics.get('path_length', 0.0)
        efficiency = path_length / ref_length if ref_length > 0 else 0.0
        
        # Count actions
        action_counts = {}
        for a in action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        # Prepare episode result
        episode_result = {
            'episode_id': episode.episode_id,
            'scene_id': episode.scene_id,
            'instruction': episode.instruction.instruction_text,
            'success': float(metrics.get('success', 0.0)),
            'spl': float(metrics.get('spl', 0.0)),
            'distance_to_goal': float(metrics.get('distance_to_goal', 0.0)),
            'path_length': float(path_length),
            'reference_path_length': float(ref_length),
            'path_efficiency': float(efficiency),
            'num_steps': step_count,
            'action_distribution': action_counts,
        }
        
        print(f"  ✓ Completed: Success={episode_result['success']:.4f}, "
              f"SPL={episode_result['spl']:.4f}, Steps={step_count}")
        
        return episode_result
        
    except Exception as e:
        print(f"  ✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return {
            'episode_id': episode.episode_id if 'episode' in locals() else 'unknown',
            'scene_id': episode.scene_id if 'episode' in locals() else 'unknown',
            'instruction': episode.instruction.instruction_text if 'episode' in locals() else 'unknown',
            'error': str(e)
        }


def main():
    """Run all VLN episodes with SatNavPathFollower."""
    parser = argparse.ArgumentParser(description='Run SatNavPathFollower on all episodes')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SatNav VLN Example with SatNavPathFollower")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "satnav_task.yaml"
    output_dir = project_root / "output"
    
    # Setup: load config, create dataset and environment
    try:
        config, dataset, env = setup_example(config_path, output_dir)
        print(f"✓ Configuration loaded from: {config_path}")
        print(f"✓ Dataset loaded: {len(dataset.episodes)} episode(s)")
        print(f"✓ Environment created successfully")
        print(f"  Max episode steps: {env.max_episode_steps}")
        print(f"  Action space: {env.action_space['actions']}")
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get dataset filename for output YAML
    dataset_path = Path(config.DATASET.DATA_PATH)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_name = dataset_path.stem  # Remove .json or .json.gz extension
    
    # Create SatNavPathFollower
    print("\n[4] Creating SatNavPathFollower...")
    path_follower = SatNavPathFollower(
        goal_radius=get_success_distance_default(config),
        turn_angle=config.SIMULATOR.TURN_ANGLE,
        return_action_string=True
    )
    print(f"✓ SatNavPathFollower created")
    print(f"  Goal radius: {path_follower.goal_radius}m")
    print(f"  Turn angle: {path_follower.turn_angle}°")
    print(f"  Strategy: Direct navigation to goal (greedy)")
    print(f"  Video generation: {'Enabled' if not args.no_video else 'Disabled'}")
    
    # Run all episodes
    print("\n[5] Running all episodes...")
    print("=" * 60)
    
    all_results = []
    total_episodes = len(dataset.episodes)
    
    for episode_idx in range(total_episodes):
        result = run_single_episode(
            env=env,
            path_follower=path_follower,
            config=config,
            output_dir=output_dir,
            save_video=not args.no_video,
            episode_idx=episode_idx,
            total_episodes=total_episodes
        )
        all_results.append(result)
    
    print("\n" + "=" * 60)
    print(f"All {total_episodes} episodes completed")
    print("=" * 60)
    
    # Calculate summary statistics
    successful_episodes = [r for r in all_results if r.get('success', 0) > 0]
    avg_success = sum(r.get('success', 0) for r in all_results) / len(all_results) if all_results else 0
    avg_spl = sum(r.get('spl', 0) for r in all_results) / len(all_results) if all_results else 0
    avg_steps = sum(r.get('num_steps', 0) for r in all_results) / len(all_results) if all_results else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Successful: {len(successful_episodes)} ({len(successful_episodes)/total_episodes*100:.1f}%)")
    print(f"  Average Success: {avg_success:.4f}")
    print(f"  Average SPL: {avg_spl:.4f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    # Generate JSON output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{dataset_name}_{timestamp}.json"
    json_path = output_dir / json_filename
    
    json_data = {
        'dataset_name': dataset_name,
        'timestamp': timestamp,
        'total_episodes': total_episodes,
        'summary': {
            'successful_episodes': len(successful_episodes),
            'success_rate': len(successful_episodes) / total_episodes if total_episodes > 0 else 0,
            'average_success': avg_success,
            'average_spl': avg_spl,
            'average_steps': avg_steps,
        },
        'episodes': all_results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {json_path}")
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()


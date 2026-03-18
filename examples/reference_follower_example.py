#!/usr/bin/env python3
"""Example script for running a VLN episode using SatNav with ReferencePathFollower.

This script demonstrates how to:
1. Load configuration from vln_task.yaml
2. Create a dataset from test data
3. Create an environment
4. Run an episode with ReferencePathFollower (follows reference path waypoints)
5. Save visualization and print evaluation metrics

The key difference from SatNavPathFollower:
- ReferencePathFollower navigates along a reference path of waypoints sequentially
- It follows the exact reference path from the dataset
- Useful for generating teacher forcing data and evaluating path following accuracy

Usage:
    python examples/reference_follower_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from satnav.core.config import get_success_distance_default
from satnav.navigation import ReferencePathFollower
from satnav.utils.examples import (
    setup_example,
    print_episode_info,
    save_visualizations,
    print_metrics,
)


def get_success_distance(config, trajectory_type: str) -> float:
    """Get SUCCESS_DISTANCE based on trajectory type."""
    sd_config = config.TASK.SUCCESS_DISTANCE
    if isinstance(sd_config, (int, float)):
        return float(sd_config)
    if trajectory_type and hasattr(sd_config, trajectory_type):
        return float(getattr(sd_config, trajectory_type))
    elif hasattr(sd_config, "DEFAULT"):
        return float(sd_config.DEFAULT)
    return 10.0


def main():
    """Run a VLN episode example with path follower."""
    print("=" * 60)
    print("SatNav VLN Example with Path Follower")
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
    
    # 4. Create reference path follower
    print("\n[4] Creating reference path follower...")
    path_follower = ReferencePathFollower(
        goal_radius=get_success_distance_default(config),
        turn_angle=config.SIMULATOR.TURN_ANGLE
    )
    print(f"✓ Reference path follower created")
    print(f"  Goal radius: {path_follower.goal_radius}m (used for all waypoints)")
    print(f"  Turn angle: {path_follower.turn_angle}°")
    
    # 5. Run episode
    print("\n[5] Running episode with path follower...")
    print("-" * 60)
    
    try:
        # Reset environment and get initial observations
        obs = env.reset()
        episode = env.current_episode
        
        # Update goal_radius based on trajectory_type
        trajectory_type = getattr(episode, 'trajectory_type', None)
        goal_radius = get_success_distance(config, trajectory_type)
        path_follower.goal_radius = goal_radius
        
        # Print episode information
        print_episode_info(episode)
        print(f"Trajectory Type: {trajectory_type} (goal_radius={goal_radius}m)")
        print(f"Reference Path: {len(episode.reference_path)} waypoints")
        print("-" * 60)
        
        # Initialize reference path follower with the episode's reference path
        path_follower.reset(episode.reference_path)
        
        # Run episode loop - follow reference path waypoints
        done = False
        step_count = 0
        action_history = []
        
        while not done and step_count < env.max_episode_steps:
            # Get next action from reference path follower
            action = path_follower.get_next_action(env._task._sim)
            action_history.append(action)
            
            # Execute action
            obs, done, info = env.step(action)
            
            step_count += 1
            
            # Stop if STOP action was taken
            if action == "STOP":
                print("  → Reached final goal!")
                break
        
        print("-" * 60)
        print(f"Episode finished after {step_count} steps")
        
    except Exception as e:
        print(f"✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save visualization
    print("\n[6] Saving visualization...")
    save_visualizations(env, obs, output_dir, prefix="example")
    
    # Print final evaluation metrics
    print_metrics(
        env=env,
        episode=episode,
        action_history=action_history,
        step_count=step_count,
        note="ReferencePathFollower navigates along the exact reference path from the dataset. "
             "It follows waypoints sequentially, useful for generating teacher forcing data "
             "and evaluating path following accuracy."
    )
    
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()

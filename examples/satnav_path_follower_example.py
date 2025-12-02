#!/usr/bin/env python3
"""Example script for running a VLN episode using SatNav with SatNavPathFollower.

This script demonstrates how to:
1. Load configuration from vln_task.yaml
2. Create a dataset from test data
3. Create an environment
4. Run an episode with SatNavPathFollower (direct goal navigation)
5. Visualize the top-down map with agent trajectory
6. Print evaluation metrics

The key difference from ReferencePathFollower:
- SatNavPathFollower navigates directly to the goal position
- It doesn't follow a reference path, but uses a greedy strategy
- Useful for oracle action generation and shortest path navigation

Usage:
    python examples/satnav_path_follower_example.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from satnav.core import Env
from satnav.core.config import load_config
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.navigation import SatNavPathFollower


def main():
    """Run a VLN episode example with SatNavPathFollower."""
    print("=" * 60)
    print("SatNav VLN Example with SatNavPathFollower")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "vln_task.yaml"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load configuration
    print("\n[1] Loading configuration...")
    try:
        config = load_config(str(config_path))
        print(f"✓ Configuration loaded from: {config_path}")
        print(f"  Forward step size: {config.SIMULATOR.FORWARD_STEP_SIZE}m")
        print(f"  Turn angle: {config.SIMULATOR.TURN_ANGLE}°")
        print(f"  RGB size: {config.SIMULATOR.RGB_SENSOR.WIDTH}x{config.SIMULATOR.RGB_SENSOR.HEIGHT}")
        print(f"  Dataset: {config.DATASET.DATA_PATH}")
        print(f"  Scenes dir: {config.DATASET.SCENES_DIR}")
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {config_path}")
        return
    
    # 2. Create dataset
    print("\n[2] Creating dataset...")
    try:
        dataset = SatNavDataset(config.DATASET)
        print(f"✓ Dataset loaded: {len(dataset.episodes)} episode(s)")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Create environment
    print("\n[3] Creating environment...")
    try:
        env = Env(config, dataset=dataset, cycle=False)
        print("✓ Environment created successfully")
        print(f"  Max episode steps: {env.max_episode_steps}")
        print(f"  Action space: {env.action_space['actions']}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Create SatNavPathFollower
    print("\n[4] Creating SatNavPathFollower...")
    path_follower = SatNavPathFollower(
        goal_radius=config.TASK.SUCCESS_DISTANCE,
        turn_angle=config.SIMULATOR.TURN_ANGLE,
        return_action_string=True
    )
    print(f"✓ SatNavPathFollower created")
    print(f"  Goal radius: {path_follower.goal_radius}m")
    print(f"  Turn angle: {path_follower.turn_angle}°")
    print(f"  Strategy: Direct navigation to goal (greedy)")
    
    # 5. Run episode
    print("\n[5] Running episode with SatNavPathFollower...")
    print("-" * 60)
    
    try:
        # Reset environment and get initial observations
        obs = env.reset()
        episode = env.current_episode
        
        print(f"Episode ID: {episode.episode_id}")
        print(f"Scene ID: {episode.scene_id}")
        print(f"Start Position: [{episode.start_position[0]:.6f}, {episode.start_position[1]:.6f}, {episode.start_position[2]:.1f}]")
        print(f"Start Rotation: {episode.start_rotation}°")
        print(f"Goal Position: [{episode.goals[0].position[0]:.6f}, {episode.goals[0].position[1]:.6f}, {episode.goals[0].position[2]:.1f}]")
        print(f"Instruction: {episode.instruction.instruction_text}")
        
        # Prepare waypoints from reference path (excluding start point)
        from satnav.core.utils import geodesic_distance
        
        # Collect all waypoints: reference_path waypoints + goal
        waypoints = []
        if episode.reference_path and len(episode.reference_path) > 1:
            # Skip the first point (start position), include all intermediate waypoints
            waypoints = episode.reference_path[1:]
        
        # Ensure goal is included (if not already the last waypoint)
        goal_position = episode.goals[0].position
        if not waypoints or waypoints[-1] != goal_position:
            waypoints.append(goal_position)
        
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
        print("-" * 60)
        
        # Run episode loop - navigate through waypoints sequentially
        done = False
        step_count = 0
        action_history = []
        current_waypoint_idx = 0
        
        while not done and step_count < env.max_episode_steps and current_waypoint_idx < len(waypoints):
            # Get current waypoint target
            current_waypoint = waypoints[current_waypoint_idx]
            
            # Get next action from SatNavPathFollower for current waypoint
            action = path_follower.get_next_action(current_waypoint, env._task._sim)
            action_history.append(action)
            
            # Execute action
            obs, done, info = env.step(action)
            
            # Get current agent state
            agent_state = env._task._sim.get_agent_state()
            
            # Check if we've reached the current waypoint
            current_distance = geodesic_distance(
                agent_state.position,
                current_waypoint
            )
            
            # Check if we've reached the current waypoint (within goal_radius)
            reached_waypoint = current_distance <= path_follower.goal_radius
            
            # Print step information (every step)
            metrics = info.get("metrics", {})
            distance_to_final_goal = metrics.get("distance_to_goal", geodesic_distance(
                agent_state.position,
                waypoints[-1]
            ))
            
            wp_info = f"WP{current_waypoint_idx+1}/{len(waypoints)}"
            status = "✓ REACHED" if reached_waypoint else ""
            print(f"Step {step_count:3d}: Action={action:15s} | "
                  f"Pos=[{agent_state.position[0]:.6f}, {agent_state.position[1]:.6f}] | "
                  f"Heading={agent_state.rotation:5.1f}° | "
                  f"{wp_info} | Dist={current_distance:.1f}m | Goal={distance_to_final_goal:.1f}m {status}")
            
            # Save top-down map visualization to file
            try:
                # Use metrics from info dict (already updated after step)
                env_metrics = info.get("metrics", {})
                if "top_down_map" in env_metrics:
                    top_down_info = env_metrics["top_down_map"]
                    map_image = top_down_info["map"]
                    bounds = top_down_info["bounds"]
                    
                    # Convert RGB to BGR for OpenCV
                    map_bgr = cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR)
                    
                    # Draw waypoint threshold circles
                    from satnav.utils.maps import draw_circle_outline
                    for i, wp in enumerate(waypoints):
                        # Use different colors for current waypoint vs others
                        if i == current_waypoint_idx:
                            # Current waypoint: yellow circle
                            circle_color = (0, 255, 255)  # Yellow (BGR)
                        elif i < current_waypoint_idx:
                            # Reached waypoints: gray circle
                            circle_color = (128, 128, 128)  # Gray (BGR)
                        else:
                            # Future waypoints: green circle
                            circle_color = (0, 255, 0)  # Green (BGR)
                        
                        # Draw threshold circle for each waypoint
                        draw_circle_outline(
                            map_bgr,
                            wp,
                            radius_meters=path_follower.goal_radius,
                            bounds=bounds,
                            color=circle_color,
                            thickness=2
                        )
                    
                    # Add text overlay with step info
                    cv2.putText(map_bgr, f"Step {step_count}: {action}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(map_bgr, f"Waypoint {current_waypoint_idx+1}/{len(waypoints)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(map_bgr, f"Dist: {current_distance:.1f}m", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(map_bgr, f"Threshold: {path_follower.goal_radius:.1f}m", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save to file (overwrite same file each step)
                    output_path = output_dir / "topdown_map_satnav_follower.png"
                    cv2.imwrite(str(output_path), map_bgr)
            except Exception as e:
                # Silently ignore errors
                pass
            
            step_count += 1
            
            # If reached current waypoint, move to next
            if reached_waypoint:
                if current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
                    print(f"  → Switching to waypoint {current_waypoint_idx+1}/{len(waypoints)}")
                else:
                    # Reached final goal - need to execute STOP action for Success measure
                    print("  → Reached final goal!")
                    # Execute STOP action to mark success
                    # This is required for Success measure (needs both distance < threshold AND STOP called)
                    if action != "STOP":
                        obs, done, info = env.step("STOP")
                        action_history.append("STOP")
                        step_count += 1
                    break
            
            # Stop if STOP action was taken (SatNavPathFollower returns STOP when at goal)
            if action == "STOP":
                # If we're at the last waypoint, we're done
                if current_waypoint_idx >= len(waypoints) - 1:
                    break
                # Otherwise, switch to next waypoint
                elif not reached_waypoint:
                    # This shouldn't happen, but handle it gracefully
                    current_waypoint_idx += 1
                    if current_waypoint_idx >= len(waypoints):
                        break
        
        print("-" * 60)
        print(f"Episode finished after {step_count} steps")
        
        # Count actions
        action_counts = {}
        for a in action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        print(f"Action distribution: {action_counts}")
        
        # Final map has been saved to output directory
        output_path = output_dir / "topdown_map_satnav_follower.png"
        print(f"\n✓ Top-down map saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Save visualization
    print("\n[6] Saving visualization...")
    try:
        metrics = env.get_metrics()
        
        if "top_down_map" in metrics:
            top_down_info = metrics["top_down_map"]
            map_image = top_down_info["map"]
            
            # Convert RGB to BGR for OpenCV
            map_bgr = cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR)
            
            # Save top-down map
            output_path = output_dir / "topdown_map_satnav_follower.png"
            cv2.imwrite(str(output_path), map_bgr)
            print(f"✓ Top-down map saved to: {output_path}")
            print(f"  Map size: {map_image.shape[1]}x{map_image.shape[0]}")
        else:
            print("  Top-down map not available (measure not enabled)")
        
        # Save RGB observation
        if "rgb" in obs:
            rgb_image = obs["rgb"]
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_path = output_dir / "final_observation_satnav_follower.png"
            cv2.imwrite(str(rgb_path), rgb_bgr)
            print(f"✓ Final observation saved to: {rgb_path}")
        
    except Exception as e:
        print(f"⚠ Error saving visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Print final evaluation metrics
    print("\n[7] Final Evaluation Metrics")
    print("=" * 60)
    
    try:
        metrics = env.get_metrics()
        
        success = metrics.get('success', 0.0)
        spl = metrics.get('spl', 0.0)
        distance = metrics.get('distance_to_goal', 0.0)
        path_length = metrics.get('path_length', 0.0)
        
        print(f"Success:           {success:.4f} {'✓' if success > 0 else '✗'}")
        print(f"SPL:               {spl:.4f}")
        print(f"Distance to Goal:  {distance:.2f} m")
        print(f"Path Length:       {path_length:.2f} m")
        
        # Compare with reference path if available
        if episode.reference_path:
            from satnav.core.utils import geodesic_distance
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
        
        print("=" * 60)
        
        # Summary
        if success > 0:
            print("\n🎉 Navigation successful! Agent reached the goal.")
        else:
            print(f"\n⚠ Navigation incomplete. Final distance: {distance:.2f}m")
        
        # Note about SatNavPathFollower
        print("\n📝 Note: SatNavPathFollower uses a greedy strategy to navigate")
        print("   directly to the goal. It may not follow the reference path")
        print("   exactly, but provides oracle actions for shortest path navigation.")
        
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()


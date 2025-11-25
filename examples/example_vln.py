#!/usr/bin/env python3
"""Example script for running a VLN episode using SatNav.

This script demonstrates how to:
1. Load configuration from YAML file
2. Create a dataset
3. Create an environment
4. Run a single episode with random actions
5. Print evaluation metrics

Usage:
    python examples/example_vln.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from satnav.core import Env
from satnav.core.config import load_config
from satnav.dataset.satnav_dataset import SatNavDataset


def main():
    """Run a VLN episode example."""
    print("=" * 60)
    print("SatNav VLN Example")
    print("=" * 60)
    
    # 1. Load configuration
    print("\n[1] Loading configuration...")
    config_path = Path(__file__).parent.parent / "configs" / "vln_task.yaml"
    
    # Use test dataset for example (since actual dataset may not exist)
    # In production, use the dataset path from config
    test_dataset_path = Path(__file__).parent.parent / "tests" / "test_data" / "satnav_dataset_example.json"
    
    try:
        config = load_config(str(config_path))
        print(f"✓ Configuration loaded from: {config_path}")
    except FileNotFoundError:
        print(f"⚠ Configuration file not found: {config_path}")
        print("  Using default configuration...")
        # Create minimal config for example
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": 500},
            "SIMULATOR": {
                "FORWARD_STEP_SIZE": 0.25,
                "TURN_ANGLE": 15,
                "RGB_SENSOR": {"WIDTH": 224, "HEIGHT": 224, "HFOV": 90}
            },
            "TASK": {
                "TYPE": "VLN",
                "SUCCESS_DISTANCE": 3.0,
                "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "PATH_LENGTH"]
            },
            "DATASET": {
                "TYPE": "SatNav",
                "SPLIT": "test",
                "DATA_PATH": str(test_dataset_path),
                "SCENES_DIR": "data/scene_datasets/"
            }
        })
    
    # 2. Create dataset
    print("\n[2] Creating dataset...")
    try:
        # Try to use dataset from config first
        if hasattr(config, "DATASET") and config.DATASET:
            dataset_config = config.DATASET
            # Override with test dataset if config path doesn't exist
            if not Path(dataset_config.DATA_PATH).exists() and test_dataset_path.exists():
                print(f"  Using test dataset: {test_dataset_path}")
                dataset_config = {
                    "DATA_PATH": str(test_dataset_path),
                    "SPLIT": "test"
                }
        else:
            dataset_config = {
                "DATA_PATH": str(test_dataset_path),
                "SPLIT": "test"
            }
        
        dataset = SatNavDataset(dataset_config)
        print(f"✓ Dataset loaded: {len(dataset.episodes)} episodes")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("  Please ensure the dataset file exists.")
        return
    
    # 3. Create environment
    print("\n[3] Creating environment...")
    try:
        # Use cycle=False for evaluation mode (default)
        env = Env(config, dataset=dataset, cycle=False)
        print("✓ Environment created successfully")
        print(f"  Max episode steps: {env.max_episode_steps}")
        print(f"  Action space: {env.action_space['actions']}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Run a single episode
    print("\n[4] Running episode...")
    print("-" * 60)
    
    try:
        # Reset environment and get initial observations
        obs = env.reset()
        episode = env.current_episode
        
        print(f"Episode ID: {episode.episode_id}")
        print(f"Scene ID: {episode.scene_id}")
        print(f"Start Position: {episode.start_position}")
        print(f"Start Rotation: {episode.start_rotation}°")
        print(f"Instruction: {episode.instruction.instruction_text}")
        print(f"Number of goals: {len(episode.goals)}")
        if len(episode.goals) > 0:
            print(f"Goal Position: {episode.goals[0].position}")
        print("-" * 60)
        
        # Run episode loop
        done = False
        step_count = 0
        actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
        
        while not done and step_count < env.max_episode_steps:
            # Simple action selection: alternate between actions
            # In a real scenario, this would be replaced by a policy/agent
            action_idx = step_count % len(actions)
            action = actions[action_idx]
            
            # Execute action
            obs, done, info = env.step(action)
            
            # Print step information
            if step_count % 10 == 0 or done:  # Print every 10 steps or at the end
                metrics = info.get("metrics", {})
                distance_to_goal = metrics.get("distance_to_goal", 0.0)
                path_length = metrics.get("path_length", 0.0)
                
                print(f"Step {step_count:3d}: Action={action:15s} | "
                      f"Distance={distance_to_goal:7.2f}m | "
                      f"Path={path_length:7.2f}m | "
                      f"Done={done}")
            
            step_count += 1
            
            # Early stop if agent calls STOP and is close to goal
            if action == "STOP" and done:
                break
        
        print("-" * 60)
        print(f"Episode finished after {step_count} steps")
        
    except RuntimeError as e:
        if "All episodes exhausted" in str(e):
            print("✓ All episodes have been evaluated")
            return
        else:
            print(f"✗ Error during episode: {e}")
            import traceback
            traceback.print_exc()
            return
    except Exception as e:
        print(f"✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Print final evaluation metrics
    print("\n[5] Final Evaluation Metrics")
    print("=" * 60)
    
    try:
        metrics = env.get_metrics()
        
        print(f"Success:           {metrics.get('success', 0.0):.4f}")
        print(f"SPL:               {metrics.get('spl', 0.0):.4f}")
        print(f"Distance to Goal:  {metrics.get('distance_to_goal', 0.0):.2f} m")
        print(f"Path Length:       {metrics.get('path_length', 0.0):.2f} m")
        
        # Additional info
        if episode.reference_path:
            print(f"\nReference Path Length: {len(episode.reference_path)} waypoints")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()


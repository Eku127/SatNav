#!/usr/bin/env python3
"""Parallel trajectory data generation from SatNav episodes.

This script uses multiprocessing to generate trajectory data in parallel,
significantly improving performance on multi-core CPUs.

Usage:
    python -m applications.trajectory_generation.generate_parallel \
        --config configs/satnav_task.yaml \
        --output_dir /path/to/output \
        --num_workers 64

Example:
    python -m applications.trajectory_generation.generate_parallel \
        --config configs/satnav_task.yaml \
        --output_dir output/trajectory_data \
        --num_workers 128
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from satnav.core import Env
from satnav.core.config import load_config, get_success_distance_default
from satnav.core.utils import geodesic_distance
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.navigation import SatNavPathFollower

from .utils import (
    INITIAL_ACTION,
    format_episode_dirname,
    satnav_action_to_streamvln,
)

# Global worker-local state (initialized once per worker process)
_worker_env = None
_worker_path_follower = None
_worker_dataset = None
_worker_output_path = None
_worker_dataset_name = "satnav"
_worker_config = None


def init_worker(config_path: str, output_path: str):
    """Initialize worker process with its own environment instance.
    
    This function is called once when each worker process starts.
    It initializes process-local global state.
    
    Args:
        config_path: Path to SatNav task configuration YAML file.
        output_path: Output directory for trajectory data.
    """
    global _worker_env, _worker_path_follower, _worker_dataset, _worker_output_path, _worker_config
    
    config = load_config(config_path)
    _worker_config = config
    _worker_dataset = SatNavDataset(config.DATASET)
    _worker_env = Env(config, dataset=_worker_dataset, cycle=False)
    _worker_output_path = output_path
    
    _worker_path_follower = SatNavPathFollower(
        goal_radius=get_success_distance_default(config),
        turn_angle=config.SIMULATOR.TURN_ANGLE,
        return_action_string=True
    )


def get_success_distance(trajectory_type: str) -> float:
    """Get SUCCESS_DISTANCE based on trajectory type.
    
    Args:
        trajectory_type: Type of trajectory ('Boundary' or 'LandmarkSet').
        
    Returns:
        SUCCESS_DISTANCE value.
    """
    global _worker_config
    sd_config = _worker_config.TASK.SUCCESS_DISTANCE
    
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


def prepare_waypoints(episode) -> List[List[float]]:
    """Prepare waypoints from episode reference path."""
    waypoints = []
    if episode.reference_path and len(episode.reference_path) > 1:
        waypoints = episode.reference_path[1:]
    
    goal_position = episode.goals[0].position
    if not waypoints or not np.allclose(waypoints[-1], goal_position, atol=1e-6, rtol=0):
        waypoints.append(goal_position)
    
    return waypoints


def check_episode_completed(output_path: str, episode_idx: int, scene_id: str, dataset_name: str) -> bool:
    """Check if episode has already been generated."""
    if isinstance(scene_id, str) and '/' in scene_id:
        scene_id = scene_id.split('/')[-1]
    
    episode_dirname = format_episode_dirname(scene_id, dataset_name, episode_idx)
    rgb_dir = os.path.join(output_path, "images", episode_dirname, "rgb")
    
    if os.path.exists(rgb_dir):
        images = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
        if len(images) > 0:
            return True
    return False


def process_single_episode(episode_idx: int) -> Optional[Dict]:
    """Process a single episode in a worker process.
    
    Uses worker-local global state initialized by init_worker.
    
    Args:
        episode_idx: Index of the episode to process.
        
    Returns:
        Annotation dictionary if successful, None otherwise.
        Returns dict with "_skipped" key if skipped due to already completed.
        Returns dict with "_max_steps" key if discarded due to reaching max steps.
    """
    global _worker_env, _worker_path_follower, _worker_dataset, _worker_output_path, _worker_dataset_name
    
    env = _worker_env
    path_follower = _worker_path_follower
    dataset = _worker_dataset
    output_path = _worker_output_path
    dataset_name = _worker_dataset_name
    
    episode = dataset.episodes[episode_idx]
    
    # Extract scene name
    scene_id = episode.scene_id
    if isinstance(scene_id, str) and '/' in scene_id:
        scene_id = scene_id.split('/')[-1]
    
    # Check if already completed (resume support)
    if check_episode_completed(output_path, episode_idx, scene_id, dataset_name):
        return {"_skipped": True, "id": episode_idx}  # Mark as skipped for counting
    
    try:
        # Set goal_radius based on trajectory_type
        trajectory_type = getattr(episode, 'trajectory_type', None)
        goal_radius = get_success_distance(trajectory_type)
        path_follower.goal_radius = goal_radius
        
        # Reset environment to this specific episode
        env._current_episode = episode
        obs = env.reset_to_episode(episode)
        
        # Prepare waypoints from reference path
        waypoints = prepare_waypoints(episode)
        
        if len(waypoints) == 0:
            print(f"[WARN] Episode {episode_idx}: no waypoints")
            return {"_failed": True, "id": episode_idx, "_reason": "no_waypoints"}
        
        # Prepare output directory
        episode_dirname = format_episode_dirname(scene_id, dataset_name, episode_idx)
        rgb_dir = os.path.join(output_path, "images", episode_dirname, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        
        # Initialize episode data
        rgb_list = []
        actions = [INITIAL_ACTION]
        current_waypoint_idx = 0
        step_count = 0
        done = False
        
        # Save initial observation
        rgb = obs["rgb"]
        rgb_list.append(rgb)
        Image.fromarray(rgb).convert("RGB").save(
            os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg")
        )
        
        # Run episode loop
        while not done and step_count < env.max_episode_steps and current_waypoint_idx < len(waypoints):
            current_waypoint = waypoints[current_waypoint_idx]
            is_final_waypoint = (current_waypoint_idx == len(waypoints) - 1)
            
            action = path_follower.get_next_action(current_waypoint, env._task._sim)
            
            if action == "STOP" and not is_final_waypoint:
                current_waypoint_idx += 1
                continue
            
            action_encoded = satnav_action_to_streamvln(action)
            actions.append(action_encoded)
            
            obs, done, info = env.step(action)
            step_count += 1
            
            rgb = obs["rgb"]
            rgb_list.append(rgb)
            Image.fromarray(rgb).convert("RGB").save(
                os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg")
            )
            
            agent_state = env._task._sim.get_agent_state()
            current_distance = geodesic_distance(
                agent_state.position,
                current_waypoint
            )
            reached_waypoint = current_distance <= path_follower.goal_radius
            
            if reached_waypoint and current_waypoint_idx < len(waypoints) - 1:
                current_waypoint_idx += 1
        
        # Check if reached max steps (discard this episode)
        if step_count >= env.max_episode_steps:
            # Clean up created directory
            import shutil
            episode_dirname = format_episode_dirname(scene_id, dataset_name, episode_idx)
            episode_dir = os.path.join(output_path, "images", episode_dirname)
            if os.path.exists(episode_dir):
                shutil.rmtree(episode_dir)
            return {"_max_steps": True, "id": episode_idx}
        
        # Validate data
        if len(actions) != len(rgb_list):
            print(f"[WARN] Episode {episode_idx}: actions/images mismatch ({len(actions)} vs {len(rgb_list)})")
            return {"_failed": True, "id": episode_idx, "_reason": "mismatch"}
        
        # Get instruction
        instruction_text = episode.instruction.instruction_text
        instructions = [instruction_text] if isinstance(instruction_text, str) else instruction_text
        
        # Create annotation
        annotation = {
            "id": episode_idx,
            "trajectory_id": episode.trajectory_id,
            "steps": len(actions) - 1,
            "video": os.path.join("images", episode_dirname),
            "instructions": instructions,
            "actions": actions,
            # Additional metadata for summary.json
            "_scene_id": episode.scene_id,
            "_episode_id": episode.episode_id,
        }
        
        return annotation
        
    except Exception as e:
        # Return failure info without printing (to avoid flooding output)
        return {"_failed": True, "id": episode_idx, "_reason": str(e)}


def main():
    """Main entry point for parallel trajectory generation."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory data from SatNav episodes (parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SatNav task configuration YAML file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for trajectory data"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(CPU count, 64))"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Determine number of workers (cap at reasonable default to avoid memory issues)
    num_workers = args.num_workers or min(cpu_count(), 64)
    
    print("=" * 60)
    print("SatNav Trajectory Generation (Parallel)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {num_workers}")
    print(f"CPU cores available: {cpu_count()}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration to get episode count
    print("Loading configuration...")
    config = load_config(args.config)
    dataset = SatNavDataset(config.DATASET)
    num_episodes = len(dataset.episodes)
    print(f"Total episodes: {num_episodes}")
    
    # Load existing annotations for resume support
    existing_annotations = {}
    summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        existing_annotations[entry["id"]] = entry
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(existing_annotations)} existing annotations")
    
    # Prepare episode indices
    episode_indices = list(range(num_episodes))
    
    print("\n" + "=" * 60)
    print("Starting parallel trajectory generation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run parallel processing with per-episode progress
    all_results = []
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(args.config), args.output_dir)
    ) as pool:
        # Use imap_unordered for real-time progress on each episode
        all_results = list(tqdm(
            pool.imap_unordered(process_single_episode, episode_indices, chunksize=1),
            total=num_episodes,
            desc="Generating trajectories",
            unit="episode"
        ))
    
    # Count and categorize results
    all_annotations = []
    skipped_count = 0
    max_steps_count = 0
    failed_reasons = {}
    none_count = 0
    
    for r in all_results:
        if r is None:
            none_count += 1
        elif r.get("_skipped"):
            skipped_count += 1
        elif r.get("_max_steps"):
            max_steps_count += 1
        elif r.get("_failed"):
            reason = r.get("_reason", "unknown")
            # Simplify reason for grouping
            if "Camera view bounds" in reason:
                reason = "Camera view bounds exceed image bounds"
            elif "exception" in reason.lower():
                reason = reason[:80]  # Truncate long messages
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
        else:
            all_annotations.append(r)
    
    print(f"\nProcessing statistics:")
    print(f"  Success: {len(all_annotations)}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Discarded (max steps): {max_steps_count}")
    print(f"  Failed: {sum(failed_reasons.values())}")
    if failed_reasons:
        print(f"  Failure reasons:")
        for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    print(f"  Unknown None: {none_count}")
    
    elapsed_time = time.time() - start_time
    
    # Combine with existing annotations (for resume support)
    final_annotations = dict(existing_annotations)
    for ann in all_annotations:
        final_annotations[ann["id"]] = ann
    
    # Sort by episode ID
    sorted_annotations = sorted(final_annotations.values(), key=lambda x: x["id"])
    
    # Save summary.json (JSONL format)
    print(f"\nSaving summary to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        for ann in sorted_annotations:
            # Create summary entry with all metadata
            summary_entry = {
                "id": ann["id"],
                "trajectory_id": ann.get("trajectory_id", ""),
                "steps": ann.get("steps", len(ann["actions"]) - 1),
                "video": ann["video"],
                "instructions": ann["instructions"],
                "actions": ann["actions"],
            }
            if "_scene_id" in ann:
                summary_entry["scene_id"] = ann["_scene_id"]
            if "_episode_id" in ann:
                summary_entry["episode_id"] = ann["_episode_id"]
            f.write(json.dumps(summary_entry) + "\n")
    
    # Save annotations.json (compact format)
    annotations_path = os.path.join(args.output_dir, "annotations.json")
    print(f"Saving annotations to: {annotations_path}")
    
    # Clean annotations (remove internal metadata)
    clean_annotations = []
    for ann in sorted_annotations:
        clean_ann = {
            "id": ann["id"],
            "trajectory_id": ann.get("trajectory_id", ""),
            "steps": ann.get("steps", len(ann["actions"]) - 1),
            "video": ann["video"],
            "instructions": ann["instructions"],
            "actions": ann["actions"],
        }
        clean_annotations.append(clean_ann)
    
    with open(annotations_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, ann in enumerate(clean_annotations):
            line = json.dumps(ann, ensure_ascii=False)
            if i < len(clean_annotations) - 1:
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")
        f.write("]\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Trajectory generation completed")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Generated annotations: {len(sorted_annotations)}")
    print(f"New in this run: {len(all_annotations)}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    if all_annotations:
        print(f"Average speed: {len(all_annotations) / elapsed_time:.2f} episodes/s")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - annotations.json: {len(sorted_annotations)} episodes")
    print(f"  - summary.json: JSONL format")
    print(f"  - images/: RGB frames")


if __name__ == "__main__":
    main()

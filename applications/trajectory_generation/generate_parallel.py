#!/usr/bin/env python3
"""Parallel trajectory data generation from SatNav episodes.

This script uses multiprocessing to generate trajectory data in parallel,
significantly improving performance on multi-core CPUs.

Usage:
    python -m applications.trajectory_generation.generate_parallel \
        --config configs/satnav_task.yaml \
        --output_dir /path/to/output \
        --num_workers 72

Example:
    python -m applications.trajectory_generation.generate_parallel \
        --config configs/satnav_task.yaml \
        --output_dir output/trajectory_data \
        --num_workers 72

Scene affinity is enabled by default: episodes are sorted by scene so each
worker processes episodes from a small number of TIF files, dramatically
reducing per-worker memory usage. Use --no_scene_affinity to disable.

Resume support: each completed episode writes a .done marker file. On resume,
episodes with a .done marker are skipped; episodes with partial images but no
.done marker (killed mid-way) are cleaned up and re-processed.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        trajectory_type: Type of trajectory ('Boundary', 'LandmarkSet', or 'Road').
        
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


def _episode_paths(output_path: str, episode_idx: int, scene_id: str, dataset_name: str):
    """Return (episode_dir, done_marker, annotation_file) paths for an episode."""
    if isinstance(scene_id, str) and '/' in scene_id:
        scene_id = scene_id.split('/')[-1]
    episode_dirname = format_episode_dirname(scene_id, dataset_name, episode_idx)
    episode_dir = os.path.join(output_path, "images", episode_dirname)
    done_marker = os.path.join(episode_dir, ".done")
    annotation_file = os.path.join(episode_dir, ".annotation.json")
    return episode_dir, done_marker, annotation_file, episode_dirname


def process_single_episode(episode_idx: int) -> Optional[Dict]:
    """Process a single episode in a worker process.

    Execution paths:

    1. ``.done`` + ``.annotation.json`` both exist  →  return cached annotation
       instantly (no simulation, no I/O).

    2. ``.done`` exists but ``.annotation.json`` missing (legacy episodes)  →
       re-simulate WITHOUT saving images to recover the action sequence.
       After simulation, compare ``len(actions)`` with the actual jpg count on
       disk.  If they match, write ``.annotation.json`` and return.
       If they **don't** match (images were corrupted / partial), log a warning,
       delete the episode directory, and fall through to a full re-run (path 3).

    3. No ``.done``  →  full run: simulate + save images + write ``.done`` and
       ``.annotation.json``.  Partial directories are cleaned up first.
    """
    global _worker_env, _worker_path_follower, _worker_dataset, _worker_output_path, _worker_dataset_name

    env = _worker_env
    path_follower = _worker_path_follower
    dataset = _worker_dataset
    output_path = _worker_output_path
    dataset_name = _worker_dataset_name

    episode = dataset.episodes[episode_idx]

    scene_id = episode.scene_id
    if isinstance(scene_id, str) and '/' in scene_id:
        scene_id = scene_id.split('/')[-1]

    episode_dir, done_marker, annotation_file, episode_dirname = _episode_paths(
        output_path, episode_idx, scene_id, dataset_name
    )

    # --- Path 1: fully cached ---
    if os.path.exists(done_marker) and os.path.exists(annotation_file):
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # annotation file corrupt → fall through to re-simulate

    # save_images=False  →  legacy episode (.done exists, re-simulate only)
    # save_images=True   →  full run (no .done, or mismatch detected below)
    save_images = not os.path.exists(done_marker)

    # Clean up partial directory before a full run.
    if save_images and os.path.exists(episode_dir):
        shutil.rmtree(episode_dir)

    try:
        trajectory_type = getattr(episode, 'trajectory_type', None)
        goal_radius = get_success_distance(trajectory_type)
        path_follower.goal_radius = goal_radius

        env._current_episode = episode
        obs = env.reset_to_episode(episode)
        path_follower.reset()

        waypoints = prepare_waypoints(episode)
        if len(waypoints) == 0:
            print(f"[WARN] Episode {episode_idx}: no waypoints")
            return {"_failed": True, "id": episode_idx, "_reason": "no_waypoints"}

        rgb_dir = os.path.join(episode_dir, "rgb")
        if save_images:
            os.makedirs(rgb_dir, exist_ok=True)

        frame_count = 0
        actions = [INITIAL_ACTION]
        current_waypoint_idx = 0
        step_count = 0
        done = False

        # Initial observation
        frame_count += 1
        if save_images:
            Image.fromarray(obs["rgb"]).convert("RGB").save(
                os.path.join(rgb_dir, f"{frame_count:03d}.jpg")
            )

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
            frame_count += 1

            if save_images:
                Image.fromarray(obs["rgb"]).convert("RGB").save(
                    os.path.join(rgb_dir, f"{frame_count:03d}.jpg")
                )

            agent_state = env._task._sim.get_agent_state()
            current_distance = geodesic_distance(
                agent_state.position,
                current_waypoint
            )
            if current_distance <= path_follower.goal_radius and current_waypoint_idx < len(waypoints) - 1:
                current_waypoint_idx += 1

        # Discard episode that hit the step cap
        if step_count >= env.max_episode_steps:
            if save_images and os.path.exists(episode_dir):
                shutil.rmtree(episode_dir)
            return {"_max_steps": True, "id": episode_idx}

        # Validate: simulation frame count must match len(actions)
        if len(actions) != frame_count:
            print(f"[WARN] Episode {episode_idx}: simulation actions/frames mismatch "
                  f"({len(actions)} vs {frame_count})")
            return {"_failed": True, "id": episode_idx, "_reason": "sim_mismatch"}

        # --- Path 2 extra check: simulation frames must match on-disk jpg count ---
        if not save_images:
            actual_jpg_count = (
                len([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
                if os.path.isdir(rgb_dir) else 0
            )
            if actual_jpg_count != frame_count:
                print(
                    f"[WARN] Episode {episode_idx}: on-disk jpg count ({actual_jpg_count}) "
                    f"!= simulation frames ({frame_count}). "
                    f"Images are corrupted/partial — re-generating."
                )
                # Delete the corrupt directory and redo as a full run.
                if os.path.exists(episode_dir):
                    shutil.rmtree(episode_dir)
                # Re-run env from start (reset already happened; need a new reset)
                obs = env.reset_to_episode(episode)
                path_follower.reset()
                os.makedirs(rgb_dir, exist_ok=True)
                save_images = True
                frame_count = 1
                actions = [INITIAL_ACTION]
                current_waypoint_idx = 0
                step_count = 0
                done = False
                Image.fromarray(obs["rgb"]).convert("RGB").save(
                    os.path.join(rgb_dir, f"{frame_count:03d}.jpg")
                )
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
                    frame_count += 1
                    Image.fromarray(obs["rgb"]).convert("RGB").save(
                        os.path.join(rgb_dir, f"{frame_count:03d}.jpg")
                    )
                    agent_state = env._task._sim.get_agent_state()
                    current_distance = geodesic_distance(agent_state.position, current_waypoint)
                    if current_distance <= path_follower.goal_radius and current_waypoint_idx < len(waypoints) - 1:
                        current_waypoint_idx += 1
                if step_count >= env.max_episode_steps:
                    if os.path.exists(episode_dir):
                        shutil.rmtree(episode_dir)
                    return {"_max_steps": True, "id": episode_idx}
                if len(actions) != frame_count:
                    return {"_failed": True, "id": episode_idx, "_reason": "rerun_mismatch"}

        instruction_text = episode.instruction.instruction_text
        instructions = [instruction_text] if isinstance(instruction_text, str) else instruction_text

        annotation = {
            "id": episode_idx,
            "trajectory_id": episode.trajectory_id,
            "steps": len(actions) - 1,
            "video": os.path.join("images", episode_dirname),
            "instructions": instructions,
            "actions": actions,
            "_scene_id": episode.scene_id,
            "_episode_id": episode.episode_id,
        }

        # Persist annotation cache so future resumes are instant.
        os.makedirs(episode_dir, exist_ok=True)
        with open(annotation_file, "w", encoding="utf-8") as _f:
            json.dump(annotation, _f, ensure_ascii=False)

        # Write .done only for full runs (legacy episodes already have it).
        if save_images:
            with open(done_marker, "w") as _f:
                pass

        return annotation

    except Exception as e:
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
        help="Number of worker processes (default: min(num_scenes, CPU//4, 72))"
    )

    parser.add_argument(
        "--no_scene_affinity",
        action="store_true",
        default=False,
        help="Disable scene-affinity grouping (not recommended: each worker may "
             "load all scene TIF files, causing high memory usage)"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration to get episode count
    print("Loading configuration...")
    config = load_config(args.config)
    dataset = SatNavDataset(config.DATASET)
    num_episodes = len(dataset.episodes)
    print(f"Total episodes: {num_episodes}")

    # Annotations are now persisted per-episode in .annotation.json files.
    # summary.json loading is no longer required for resume support.

    # Build episode index list, optionally sorted by scene for scene affinity.
    #
    # Scene affinity ensures consecutive episodes in each worker's chunk come
    # from the same scene, so each worker only needs to keep 1-3 TIF files in
    # its scene cache rather than potentially all scenes.
    episode_indices = list(range(num_episodes))
    chunksize = 1

    if not args.no_scene_affinity:
        scene_to_episodes: Dict[str, List[int]] = defaultdict(list)
        for idx in episode_indices:
            scene_id = dataset.episodes[idx].scene_id
            if isinstance(scene_id, str) and '/' in scene_id:
                scene_id = scene_id.split('/')[-1]
            scene_to_episodes[scene_id].append(idx)

        num_scenes = len(scene_to_episodes)
        # Sort episodes: all episodes from the same scene are consecutive.
        episode_indices = []
        for sid in sorted(scene_to_episodes.keys()):
            episode_indices.extend(scene_to_episodes[sid])

        # Default workers: one worker per scene up to CPU//4, capped at 72.
        # This ensures each worker "owns" roughly 1-3 scenes in its chunk.
        default_workers = min(num_scenes, max(1, cpu_count() // 4), 72)
        num_workers = args.num_workers or default_workers

        # chunksize: divide sorted list into num_workers contiguous blocks.
        # Each block spans 1-3 scenes → worker scene cache stays small.
        chunksize = max(1, math.ceil(len(episode_indices) / num_workers))

        print(f"Scene affinity: {num_scenes} scenes, "
              f"~{len(episode_indices) // num_scenes} episodes/scene, "
              f"chunksize={chunksize}")
    else:
        num_workers = args.num_workers or min(cpu_count() // 4, 72)
        print("Scene affinity: disabled")

    print("=" * 60)
    print("SatNav Trajectory Generation (Parallel)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {num_workers}")
    print(f"CPU cores available: {cpu_count()}")
    print()

    print("\n" + "=" * 60)
    print("Starting parallel trajectory generation")
    print("=" * 60)

    start_time = time.time()

    # Run parallel processing.
    # pool.imap (ordered) with large chunksize keeps scene-sorted episodes
    # together in each worker, minimising TIF file cache thrashing.
    all_results = []
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(args.config), args.output_dir)
    ) as pool:
        all_results = list(tqdm(
            pool.imap(process_single_episode, episode_indices, chunksize=chunksize),
            total=num_episodes,
            desc="Generating trajectories",
            unit="episode"
        ))
    
    # Count and categorize results.
    # Note: there is no longer a "_skipped" path — episodes with a cached
    # .annotation.json return the full annotation directly, so every valid
    # episode contributes to all_annotations.
    all_annotations = []
    max_steps_count = 0
    failed_reasons = {}
    none_count = 0

    for r in all_results:
        if r is None:
            none_count += 1
        elif r.get("_max_steps"):
            max_steps_count += 1
        elif r.get("_failed"):
            reason = r.get("_reason", "unknown")
            if "Camera view bounds" in reason:
                reason = "Camera view bounds exceed image bounds"
            elif "exception" in reason.lower():
                reason = reason[:80]
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
        else:
            all_annotations.append(r)

    print(f"\nProcessing statistics:")
    print(f"  Success (incl. cached): {len(all_annotations)}")
    print(f"  Discarded (max steps): {max_steps_count}")
    print(f"  Failed: {sum(failed_reasons.values())}")
    if failed_reasons:
        print(f"  Failure reasons:")
        for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    print(f"  Unknown None: {none_count}")

    elapsed_time = time.time() - start_time

    # Deduplicate by episode id (in case of rare duplicate results) and sort.
    final_annotations: Dict[int, Dict] = {}
    for ann in all_annotations:
        final_annotations[ann["id"]] = ann

    # Sort by episode ID
    sorted_annotations = sorted(final_annotations.values(), key=lambda x: x["id"])
    
    summary_path = os.path.join(args.output_dir, "summary.json")

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
    print(f"Generated annotations: {len(sorted_annotations)} / {num_episodes} episodes")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    if all_annotations:
        print(f"Average speed: {len(all_annotations) / elapsed_time:.2f} episodes/s")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - annotations.json: {len(sorted_annotations)} episodes")
    print(f"  - summary.json: JSONL format")
    print(f"  - images/: RGB frames")


if __name__ == "__main__":
    main()

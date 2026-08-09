from __future__ import annotations

"""SatNav Trajectory Generation Runner."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class SatNavTrajectoryRunner:
    """Runner for generating trajectory data from SatNav episodes.
    
    This class loads SatNav episodes, runs them using a path follower,
    and saves RGB images and action sequences in StreamVLN-compatible format.
    """
    
    def __init__(
        self,
        config_path: str,
        output_path: str,
        landmark_success: Optional[float] = None,
    ):
        """Initialize the trajectory generator.
        
        Args:
            config_path: Path to SatNav task configuration YAML file.
            output_path: Output directory for trajectory data.
            landmark_success: Optional override for LandmarkSet
                SUCCESS_DISTANCE, kept for backward-compatible CLI usage.
        """
        self.config_path = config_path
        self.output_path = output_path
        self.dataset_name = "satnav"
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        self.config = load_config(config_path)
        self._apply_success_distance_overrides(landmark_success)
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create dataset
        print(f"Loading dataset from: {self.config.DATASET.DATA_PATH}")
        self.dataset = SatNavDataset(self.config.DATASET)
        print(f"Total episodes: {len(self.dataset.episodes)}")
        
        # Create environment
        print("Creating environment...")
        self.env = Env(self.config, dataset=self.dataset, cycle=False)
        
        # Create path follower
        self.path_follower = SatNavPathFollower(
            goal_radius=get_success_distance_default(self.config),
            turn_angle=self.config.SIMULATOR.TURN_ANGLE,
            return_action_string=True
        )
        
        print(f"✓ Initialization complete")
        print(f"  Episodes: {len(self.dataset.episodes)}")
        print(f"  Goal radius: {self.path_follower.goal_radius}m")
        print(f"  Turn angle: {self.path_follower.turn_angle}°")
        print(f"  Output directory: {self.output_path}")
        self._completed_episode_ids = set()

    def _apply_success_distance_overrides(
        self, landmark_success: Optional[float]
    ) -> None:
        """Apply optional CLI overrides to SUCCESS_DISTANCE settings."""
        if landmark_success is None:
            return

        sd_config = self.config.TASK.SUCCESS_DISTANCE

        if isinstance(sd_config, (int, float)):
            self.config.TASK.SUCCESS_DISTANCE = {
                "DEFAULT": float(sd_config),
                "Boundary": float(sd_config),
                "LandmarkSet": float(landmark_success),
                "Road": float(sd_config),
            }
            return

        self.config.TASK.SUCCESS_DISTANCE.LandmarkSet = float(landmark_success)
    
    def _prepare_waypoints(self, episode) -> List[List[float]]:
        """Prepare waypoints from episode reference path.
        
        Skip the first point (start position) and ensure goal is included.
        
        Args:
            episode: VLN episode with reference path.
            
        Returns:
            List of waypoints [[lon, lat, alt], ...].
        """
        waypoints = []
        if episode.reference_path and len(episode.reference_path) > 1:
            # Skip the first point (start position), include all intermediate waypoints
            waypoints = episode.reference_path[1:]
        
        # Ensure goal is included (if not already the last waypoint)
        goal_position = episode.goals[0].position
        if not waypoints or any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(waypoints[-1], goal_position)):
            waypoints.append(goal_position)
        
        return waypoints

    def _get_episode_scene_name(self, scene_id: str) -> str:
        """Normalize scene id to a compact scene name."""
        if isinstance(scene_id, str) and '/' in scene_id:
            return scene_id.split('/')[-1]
        return scene_id
    
    def _check_episode_completed(self, episode_idx: int, scene_id: str) -> bool:
        """Check if episode has already been generated.
        
        Args:
            episode_idx: Episode index.
            scene_id: Scene ID (might be a full path).
            
        Returns:
            True if episode has a verified successful record in summary.json.
        """
        return episode_idx in self._completed_episode_ids

    def _cleanup_episode_output(self, scene_id: str, episode_idx: int) -> None:
        """Remove partial outputs for one episode."""
        scene_id = self._get_episode_scene_name(scene_id)
        episode_dirname = format_episode_dirname(scene_id, self.dataset_name, episode_idx)
        episode_dir = os.path.join(self.output_path, "images", episode_dirname)
        if os.path.exists(episode_dir):
            shutil.rmtree(episode_dir)

    def _is_summary_entry_complete(self, entry: Dict) -> bool:
        """Validate summary entry and corresponding on-disk artifacts."""
        try:
            int(entry["id"])
            steps = int(entry["steps"])
            video_rel = entry["video"]
        except (KeyError, TypeError, ValueError):
            return False

        if steps < 0 or not isinstance(video_rel, str):
            return False

        rgb_dir = os.path.join(self.output_path, video_rel, "rgb")
        if not os.path.isdir(rgb_dir):
            return False

        jpgs = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg"))
        expected = steps + 1  # includes initial frame before first action
        if len(jpgs) != expected:
            return False

        for idx, filename in enumerate(jpgs, start=1):
            if filename != f"{idx:03d}.jpg":
                return False

        return True
    
    def _get_success_distance(self, trajectory_type: str) -> float:
        """Get SUCCESS_DISTANCE based on trajectory type.
        
        Args:
            trajectory_type: Type of trajectory ('Boundary', 'LandmarkSet', or 'Road').
            
        Returns:
            SUCCESS_DISTANCE value.
        """
        sd_config = self.config.TASK.SUCCESS_DISTANCE
        
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
    
    def _run_episode(
        self,
        episode_idx: int,
        episode
    ) -> Optional[Dict]:
        """Run a single episode and save trajectory data.
        
        Args:
            episode_idx: Episode index (used as episode ID).
            episode: VLN episode object.
            
        Returns:
            Episode annotation dictionary, or None if failed.
            Returns dict with "_max_steps" key if discarded due to reaching max steps.
        """
        # Extract scene name from scene_id (might be a full path)
        scene_id = self._get_episode_scene_name(episode.scene_id)
        
        # Check if already completed (resume support)
        if self._check_episode_completed(episode_idx, scene_id):
            return None  # Skip already completed episodes
        
        try:
            # Set goal_radius based on trajectory_type
            trajectory_type = getattr(episode, 'trajectory_type', None)
            goal_radius = self._get_success_distance(trajectory_type)
            self.path_follower.goal_radius = goal_radius
            
            # Reset environment to this specific episode
            # Use reset_to_episode to ensure correct episode is loaded
            # (avoids sync issues with dataset iterator)
            obs = self.env.reset_to_episode(episode)
            self.path_follower.reset()
            
            # Prepare waypoints from reference path
            waypoints = self._prepare_waypoints(episode)
            
            if len(waypoints) == 0:
                print(f"  Warning: Episode {episode_idx} has no waypoints, skipping")
                return None
            
            # Prepare output directory
            episode_dirname = format_episode_dirname(scene_id, self.dataset_name, episode_idx)
            rgb_dir = os.path.join(self.output_path, "images", episode_dirname, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            
            # Initialize episode data
            rgb_list = []
            actions = [INITIAL_ACTION]  # Start with -1 placeholder
            current_waypoint_idx = 0
            step_count = 0
            done = False
            
            # Save initial observation (before any action)
            rgb = obs["rgb"]
            rgb_list.append(rgb)
            Image.fromarray(rgb).convert("RGB").save(
                os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg")
            )
            
            # Run episode loop
            while not done and step_count < self.env.max_episode_steps and current_waypoint_idx < len(waypoints):
                # Get current waypoint
                current_waypoint = waypoints[current_waypoint_idx]
                is_final_waypoint = (current_waypoint_idx == len(waypoints) - 1)
                
                # Get next action
                action = self.path_follower.get_next_action(current_waypoint, self.env._task._sim)
                
                # Handle intermediate waypoints: skip STOP and move to next waypoint
                if action == "STOP" and not is_final_waypoint:
                    current_waypoint_idx += 1
                    continue
                
                # Convert action to StreamVLN encoding
                action_encoded = satnav_action_to_streamvln(action)
                actions.append(action_encoded)
                
                # Execute action
                obs, done, info = self.env.step(action)
                step_count += 1
                
                # Save RGB frame (observation after executing action)
                rgb = obs["rgb"]
                rgb_list.append(rgb)
                Image.fromarray(rgb).convert("RGB").save(
                    os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg")
                )
                
                # Check if reached current waypoint
                agent_state = self.env._task._sim.get_agent_state()
                current_distance = geodesic_distance(
                    agent_state.position,
                    current_waypoint
                )
                reached_waypoint = current_distance <= self.path_follower.goal_radius
                
                # Move to next waypoint if reached
                if reached_waypoint and current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
            
            # Check if reached max steps (discard this episode)
            if step_count >= self.env.max_episode_steps:
                print(f"  Warning: Episode {episode_idx} reached max steps ({step_count}), discarding")
                # Clean up created directory
                self._cleanup_episode_output(scene_id, episode_idx)
                return {"_max_steps": True, "id": episode_idx}
            
            # Validate data
            if len(actions) != len(rgb_list):
                print(f"  Warning: Episode {episode_idx} actions/images mismatch "
                      f"({len(actions)} vs {len(rgb_list)}), skipping")
                self._cleanup_episode_output(scene_id, episode_idx)
                return None
            
            # Get instruction
            instruction_text = episode.instruction.instruction_text
            instructions = [instruction_text] if isinstance(instruction_text, str) else instruction_text
            
            # Create annotation
            # steps = len(actions) - 1 (excluding initial -1 placeholder)
            annotation = {
                "id": episode_idx,
                "trajectory_id": episode.trajectory_id,
                "steps": len(actions) - 1,
                "video": os.path.join("images", episode_dirname),
                "instructions": instructions,
                "actions": actions,
            }
            
            return annotation
            
        except Exception as e:
            print(f"  Error processing episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_episode_output(scene_id, episode_idx)
            return None
    
    def generate(self, episode_indices: Optional[List[int]] = None) -> None:
        """Generate trajectory data for all episodes.
        
        Iterates through all episodes, runs them with the path follower,
        and saves RGB images and action sequences.
        """
        print("\n" + "=" * 60)
        print("Starting trajectory generation")
        print("=" * 60)
        
        # Load existing annotations from summary.json for resume support
        existing_annotations = {}
        invalid_summary_entries = 0
        summary_path = os.path.join(self.output_path, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            invalid_summary_entries += 1
                            continue
                        if not self._is_summary_entry_complete(entry):
                            invalid_summary_entries += 1
                            continue
                        try:
                            episode_id = int(entry["id"])
                            actions = entry["actions"]
                        except (KeyError, TypeError, ValueError):
                            invalid_summary_entries += 1
                            continue
                        existing_annotations[episode_id] = {
                            "id": episode_id,
                            "trajectory_id": entry.get("trajectory_id", ""),
                            "steps": entry.get("steps", len(actions) - 1),
                            "video": entry["video"],
                            "instructions": entry["instructions"],
                            "actions": actions,
                        }
            self._completed_episode_ids = set(existing_annotations.keys())
            print(f"Loaded {len(existing_annotations)} existing annotations from summary.json")
            if invalid_summary_entries > 0:
                print(
                    f"Ignored {invalid_summary_entries} invalid/incomplete summary entries "
                    "(will regenerate these episodes)"
                )
        else:
            self._completed_episode_ids = set()
        
        annotations = []
        completed_count = 0
        skipped_count = 0
        failed_count = 0
        max_steps_count = 0
        
        if episode_indices is None:
            episodes_to_process = list(range(len(self.dataset.episodes)))
        else:
            episodes_to_process = []
            for idx in episode_indices:
                if 0 <= idx < len(self.dataset.episodes):
                    episodes_to_process.append(int(idx))
                else:
                    print(f"Warning: episode index out of range, skipping: {idx}")

        # Process requested episodes with progress bar
        for episode_idx in tqdm(episodes_to_process, desc="Generating trajectories", unit="episode"):
            episode = self.dataset.episodes[episode_idx]
            # Check if already completed - load from existing annotations
            if self._check_episode_completed(episode_idx, episode.scene_id):
                skipped_count += 1
                # Load existing annotation
                if episode_idx in existing_annotations:
                    annotations.append(existing_annotations[episode_idx])
                continue
            
            # Run episode
            annotation = self._run_episode(episode_idx, episode)
            
            if annotation is None:
                failed_count += 1
            elif annotation.get("_max_steps"):
                # Episode reached max steps, discarded
                max_steps_count += 1
            else:
                annotations.append(annotation)
                completed_count += 1
                
                # Append to summary.json (JSONL format)
                summary_path = os.path.join(self.output_path, "summary.json")
                with open(summary_path, "a", encoding="utf-8") as f:
                    summary_entry = {
                        **annotation,
                        "trajectory_id": episode.trajectory_id,
                        "scene_id": episode.scene_id,
                        "episode_id": episode.episode_id,
                    }
                    f.write(json.dumps(summary_entry) + "\n")
        
        # Save final annotations.json (compact format to save space)
        annotations_path = os.path.join(self.output_path, "annotations.json")
        print(f"\nSaving annotations to: {annotations_path}")
        with open(annotations_path, "w", encoding="utf-8") as f:
            # Use compact format: one episode per line
            f.write("[\n")
            for i, ann in enumerate(annotations):
                line = json.dumps(ann, ensure_ascii=False)
                if i < len(annotations) - 1:
                    f.write(f"  {line},\n")
                else:
                    f.write(f"  {line}\n")
            f.write("]\n")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Trajectory generation completed")
        print("=" * 60)
        print(f"Total episodes: {len(episodes_to_process)}")
        print(f"Newly generated: {completed_count}")
        print(f"Skipped (already exists): {skipped_count}")
        print(f"Discarded (max steps): {max_steps_count}")
        print(f"Failed: {failed_count}")
        print(f"\nOutput directory: {self.output_path}")
        print(f"  - annotations.json: {len(annotations)} episodes")
        print(f"  - summary.json: JSONL format")
        print(f"  - images/: RGB frames")

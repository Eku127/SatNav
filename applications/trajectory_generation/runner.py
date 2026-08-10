"""SatNav Trajectory Generation Runner."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from satnav.core import Env
from satnav.core.config import load_config, get_success_distance_default
from satnav.core.utils import geodesic_distance
from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.navigation import SatNavPathFollower
from satnav.task.actions import INITIAL_ACTION_INDEX, encode_action
from satnav.task.config import get_episode_success_distance

from .utils import (
    annotation_artifacts_complete,
    build_annotation,
    build_summary_entry,
    episode_generation_digest,
    episode_source_digest,
    format_episode_dirname,
    normalize_scene_id,
    public_annotation,
    strict_json_loads,
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
        self._scene_identity_cache = {}

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
            return_action_string=True,
        )

        print("✓ Initialization complete")
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
        if not waypoints or any(
            abs(float(a) - float(b)) > 1e-6
            for a, b in zip(waypoints[-1], goal_position)
        ):
            waypoints.append(goal_position)

        return waypoints

    def _get_episode_scene_name(self, scene_id: str) -> str:
        """Normalize scene id to a compact scene name."""
        return normalize_scene_id(scene_id)

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
        episode_dirname = format_episode_dirname(
            scene_id, self.dataset_name, episode_idx
        )
        episode_dir = os.path.join(self.output_path, "images", episode_dirname)
        if os.path.exists(episode_dir):
            shutil.rmtree(episode_dir)

    def _is_summary_entry_complete(
        self,
        entry: Dict,
        *,
        expected_source_digest: str,
        expected_generation_digest: str,
        expected_video: str,
        expected_scene_id: str,
        expected_episode_id,
    ) -> bool:
        """Validate summary entry and corresponding on-disk artifacts."""
        return annotation_artifacts_complete(
            self.output_path,
            entry,
            expected_source_digest=expected_source_digest,
            expected_generation_digest=expected_generation_digest,
            expected_video=expected_video,
            expected_scene_id=expected_scene_id,
            expected_episode_id=expected_episode_id,
        )

    def _generation_digest_for_episode(self, episode) -> str:
        return episode_generation_digest(
            self.config,
            episode,
            scene_cache=self._scene_identity_cache,
        )

    def _write_summary(self, annotations: List[Dict]) -> None:
        """Atomically replace summary.json with unique, source-bound rows."""

        summary_path = Path(self.output_path) / "summary.json"
        temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for annotation in sorted(annotations, key=lambda item: int(item["id"])):
                source_episode = self.dataset.episodes[int(annotation["id"])]
                summary_entry = build_summary_entry(
                    annotation,
                    scene_id=source_episode.scene_id,
                    episode_id=source_episode.episode_id,
                )
                handle.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, summary_path)

    def _get_success_distance(self, trajectory_type: str) -> float:
        """Get SUCCESS_DISTANCE based on trajectory type.

        Args:
            trajectory_type: Type of trajectory ('Boundary', 'LandmarkSet', or 'Road').

        Returns:
            SUCCESS_DISTANCE value.
        """
        return get_episode_success_distance(self.config, trajectory_type)

    def _run_episode(self, episode_idx: int, episode) -> Optional[Dict]:
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

        # A non-complete summary record does not own reusable frames.  Remove
        # any partial/extra files before regenerating the episode so the next
        # integrity check sees exactly the new contiguous sequence.
        self._cleanup_episode_output(scene_id, episode_idx)

        try:
            # Set goal_radius based on trajectory_type
            trajectory_type = getattr(episode, "trajectory_type", None)
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
            episode_dirname = format_episode_dirname(
                scene_id, self.dataset_name, episode_idx
            )
            rgb_dir = os.path.join(self.output_path, "images", episode_dirname, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)

            # Initialize episode data
            rgb_list = []
            actions = [INITIAL_ACTION_INDEX]
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
            while (
                not done
                and step_count < self.env.max_episode_steps
                and current_waypoint_idx < len(waypoints)
            ):
                # Get current waypoint
                current_waypoint = waypoints[current_waypoint_idx]
                is_final_waypoint = current_waypoint_idx == len(waypoints) - 1

                # Get next action
                action = self.path_follower.get_next_action(
                    current_waypoint,
                    self.env.simulator,
                )

                # Handle intermediate waypoints: skip STOP and move to next waypoint
                if action == "STOP" and not is_final_waypoint:
                    current_waypoint_idx += 1
                    continue

                # Convert the action to SatNav's canonical integer encoding.
                action_encoded = encode_action(action)
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
                agent_state = self.env.agent_state
                current_distance = geodesic_distance(
                    agent_state.position, current_waypoint
                )
                reached_waypoint = current_distance <= self.path_follower.goal_radius

                # Move to next waypoint if reached
                if reached_waypoint and current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1

            # Check if reached max steps (discard this episode)
            if step_count >= self.env.max_episode_steps:
                print(
                    f"  Warning: Episode {episode_idx} reached max steps ({step_count}), discarding"
                )
                # Clean up created directory
                self._cleanup_episode_output(scene_id, episode_idx)
                return {"_max_steps": True, "id": episode_idx}

            # Validate data
            if len(actions) != len(rgb_list):
                print(
                    f"  Warning: Episode {episode_idx} actions/images mismatch "
                    f"({len(actions)} vs {len(rgb_list)}), skipping"
                )
                self._cleanup_episode_output(scene_id, episode_idx)
                return None

            annotation = build_annotation(
                episode_idx=episode_idx,
                trajectory_id=episode.trajectory_id,
                episode_dirname=episode_dirname,
                instruction_text=episode.instruction.instruction_text,
                actions=actions,
            )
            annotation["_scene_id"] = normalize_scene_id(episode.scene_id)
            annotation["_episode_id"] = episode.episode_id
            annotation["_source_digest"] = episode_source_digest(episode_idx, episode)
            annotation["_generation_digest"] = self._generation_digest_for_episode(
                episode
            )

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
        conflicted_summary_ids = set()
        invalid_summary_entries = 0
        summary_path = os.path.join(self.output_path, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = strict_json_loads(line)
                        except (json.JSONDecodeError, ValueError):
                            invalid_summary_entries += 1
                            continue
                        try:
                            episode_id = int(entry["id"])
                            actions = entry["actions"]
                        except (KeyError, TypeError, ValueError):
                            invalid_summary_entries += 1
                            continue
                        if not 0 <= episode_id < len(self.dataset.episodes):
                            invalid_summary_entries += 1
                            continue
                        expected_source_digest = episode_source_digest(
                            episode_id, self.dataset.episodes[episode_id]
                        )
                        expected_generation_digest = (
                            self._generation_digest_for_episode(
                                self.dataset.episodes[episode_id]
                            )
                        )
                        source_episode = self.dataset.episodes[episode_id]
                        expected_scene_id = normalize_scene_id(source_episode.scene_id)
                        expected_video = os.path.join(
                            "images",
                            format_episode_dirname(
                                expected_scene_id,
                                self.dataset_name,
                                episode_id,
                            ),
                        )
                        if not self._is_summary_entry_complete(
                            entry,
                            expected_source_digest=expected_source_digest,
                            expected_generation_digest=expected_generation_digest,
                            expected_video=expected_video,
                            expected_scene_id=expected_scene_id,
                            expected_episode_id=source_episode.episode_id,
                        ):
                            invalid_summary_entries += 1
                            continue
                        if (
                            episode_id in existing_annotations
                            or episode_id in conflicted_summary_ids
                        ):
                            existing_annotations.pop(episode_id, None)
                            conflicted_summary_ids.add(episode_id)
                            invalid_summary_entries += 1
                            continue
                        existing_annotations[episode_id] = {
                            "id": episode_id,
                            "trajectory_id": entry.get("trajectory_id", ""),
                            "steps": entry.get("steps", len(actions) - 1),
                            "video": entry["video"],
                            "instructions": entry["instructions"],
                            "actions": actions,
                            "_scene_id": entry.get("scene_id"),
                            "_episode_id": entry.get("episode_id"),
                            "_source_digest": entry.get("source_digest"),
                            "_generation_digest": entry.get("generation_digest"),
                        }
            self._completed_episode_ids = set(existing_annotations.keys())
            self._write_summary(list(existing_annotations.values()))
            print(
                f"Loaded {len(existing_annotations)} existing annotations from summary.json"
            )
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
        for episode_idx in tqdm(
            episodes_to_process, desc="Generating trajectories", unit="episode"
        ):
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
                    summary_entry = build_summary_entry(
                        annotation,
                        scene_id=episode.scene_id,
                        episode_id=episode.episode_id,
                    )
                    f.write(json.dumps(summary_entry) + "\n")

        # Save final annotations.json (compact format to save space)
        annotations.sort(key=lambda annotation: int(annotation["id"]))
        self._write_summary(annotations)

        annotations_path = os.path.join(self.output_path, "annotations.json")
        print(f"\nSaving annotations to: {annotations_path}")
        with open(annotations_path, "w", encoding="utf-8") as f:
            # Use compact format: one episode per line
            f.write("[\n")
            for i, ann in enumerate(annotations):
                line = json.dumps(public_annotation(ann), ensure_ascii=False)
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
        print("  - summary.json: JSONL format")
        print("  - images/: RGB frames")

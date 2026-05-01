#!/usr/bin/env python3
"""Generate trajectory data from SatNav episodes.

This script loads SatNav episodes and generates trajectory data in
StreamVLN-compatible format, including RGB images and action sequences.

Usage:
    python -m applications.trajectory_generation.generate \\
        --config configs/satnav_task.yaml \\
        --output_dir /path/to/output

Example:
    python -m applications.trajectory_generation.generate \\
        --config configs/satnav_task.yaml \\
        --output_dir output/trajectory_data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from applications.trajectory_generation import SatNavTrajectoryRunner


def _load_episode_indices(path: Path):
    """Load episode indices from a JSON/JSONL/text file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if "episode_indices" in payload:
            payload = payload["episode_indices"]
        elif "episodes" in payload:
            payload = payload["episodes"]

    if isinstance(payload, list):
        indices = []
        for item in payload:
            if isinstance(item, dict):
                if "episode_index" in item:
                    indices.append(int(item["episode_index"]))
                elif "id" in item:
                    indices.append(int(item["id"]))
            else:
                indices.append(int(item))
        return indices

    indices = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for chunk in line.split(','):
            chunk = chunk.strip()
            if chunk:
                indices.append(int(chunk))
    return indices


def main():
    """Main entry point for trajectory generation."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory data from SatNav episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate trajectories using test data
  python -m applications.trajectory_generation.generate \\
      --config configs/satnav_task.yaml \\
      --output_dir output/trajectory_data

  # Use custom configuration
  python -m applications.trajectory_generation.generate \\
      --config path/to/custom_config.yaml \\
      --output_dir /path/to/output
        """
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
        "--landmark_success",
        type=float,
        default=2.0,
        help="SUCCESS_DISTANCE for LandmarkSet episodes in meters (default: 2.0)"
    )

    parser.add_argument(
        "--episode_indices_file",
        type=str,
        default=None,
        help="Optional file containing episode indices to generate (JSON array/object or newline/comma-separated text)"
    )

    parser.add_argument(
        "--preselect_aerial",
        action="store_true",
        help="Preselect 3D-rich AerialSim episodes before trajectory generation"
    )
    parser.add_argument(
        "--preselect_candidates",
        type=int,
        default=200,
        help="Number of candidate episodes to preselect from (default: 200)"
    )
    parser.add_argument(
        "--preselect_samples",
        type=int,
        default=4,
        help="Number of sampled frames per episode during preselection (default: 4)"
    )
    parser.add_argument(
        "--preselect_topk",
        type=int,
        default=50,
        help="Keep top-K episodes after thresholding (default: 50)"
    )
    parser.add_argument(
        "--preselect_min_score",
        type=float,
        default=None,
        help="Optional minimum preselection score"
    )
    parser.add_argument(
        "--preselect_min_avg_std",
        type=float,
        default=35.0,
        help="Minimum average frame std for preselection (default: 35.0)"
    )
    parser.add_argument(
        "--preselect_max_black_frac",
        type=float,
        default=0.01,
        help="Maximum average black pixel fraction for preselection (default: 0.01)"
    )
    parser.add_argument(
        "--preselect_min_height_range",
        type=float,
        default=3.0,
        help="Minimum average local height range for preselection (default: 3.0)"
    )
    parser.add_argument(
        "--preselect_only",
        action="store_true",
        help="Run preselection only and skip trajectory generation"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SatNav Trajectory Generation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Landmark success distance: {args.landmark_success}m")
    selected_indices = None
    if args.episode_indices_file:
        episode_indices_path = Path(args.episode_indices_file)
        if not episode_indices_path.exists():
            print(f"Error: episode indices file not found: {episode_indices_path}")
            sys.exit(1)
        selected_indices = _load_episode_indices(episode_indices_path)
        print(f"Episode subset file: {episode_indices_path}")
        print(f"Requested episodes: {len(selected_indices)}")
    print()
    
    # Create runner and generate trajectories
    try:
        runner = SatNavTrajectoryRunner(
            config_path=args.config,
            output_path=args.output_dir,
            landmark_success=args.landmark_success
        )
        if args.preselect_aerial:
            os.makedirs(args.output_dir, exist_ok=True)
            report_path = os.path.join(args.output_dir, "preselect_ranking.json")
            selected = runner.preselect_aerial_episodes(
                max_candidates=args.preselect_candidates,
                samples_per_episode=args.preselect_samples,
                top_k=args.preselect_topk,
                min_score=args.preselect_min_score,
                min_avg_std=args.preselect_min_avg_std,
                max_avg_black_frac=args.preselect_max_black_frac,
                min_avg_height_range=args.preselect_min_height_range,
                save_path=report_path,
            )
            selected_indices = [item["episode_index"] for item in selected]
            print(f"Preselected episodes: {len(selected_indices)}")
            if args.preselect_only:
                print("\n✓ Preselection completed successfully!")
                return
            if len(selected_indices) == 0:
                print("\nNo episodes passed preselection thresholds; skip generation.")
                return

        runner.generate(episode_indices=selected_indices)
        print("\n✓ Trajectory generation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during trajectory generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

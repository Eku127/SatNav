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
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from applications.trajectory_generation import SatNavTrajectoryRunner


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
    print()
    
    # Create runner and generate trajectories
    try:
        runner = SatNavTrajectoryRunner(
            config_path=args.config,
            output_path=args.output_dir
        )
        runner.generate()
        print("\n✓ Trajectory generation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during trajectory generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
一键处理数据集：生成 episodes

Usage:
    python -m applications.episode_processing.run_all SatNav-v0.1
"""

import argparse
import sys

try:
    from .config import get_data_dir, get_dataset_path, TRAIN_CITIES, EVAL_CITIES
    from .process_episodes import process_episodes
except ImportError:
    from config import get_data_dir, get_dataset_path, TRAIN_CITIES, EVAL_CITIES
    from process_episodes import process_episodes


def run_all(version: str):
    """
    运行所有数据处理步骤

    Args:
        version: 数据集版本名称
    """
    dataset_path = get_dataset_path(version)
    data_dir = get_data_dir(version)

    print("=" * 70)
    print("SatNav Data Processing Pipeline")
    print("=" * 70)
    print(f"Version: {version}")
    print(f"Dataset path: {dataset_path}")
    print(f"Data directory: {data_dir}")
    print(f"Train cities: {', '.join(TRAIN_CITIES)}")
    print(f"Eval cities: {', '.join(EVAL_CITIES)}")
    print("=" * 70)

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    results = {}

    print("\n" + "=" * 70)
    print("Processing Episodes")
    print("=" * 70)
    try:
        results["episodes"] = process_episodes(version)
    except Exception as e:
        print(f"Error processing episodes: {e}")
        results["episodes"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    if "episodes" in results and isinstance(results["episodes"], dict):
        if "error" not in results["episodes"]:
            print("\nEpisodes:")
            for split, stats in results["episodes"].items():
                if stats:
                    print(f"  {split}: {stats['total']} total "
                          f"({stats['boundary']} boundary, {stats['landmark']} landmark, {stats['road']} road)")

    print("\n" + "=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="一键处理数据集：生成 episodes"
    )
    parser.add_argument(
        "version",
        type=str,
        help="数据集版本名称 (e.g., SatNav-v0.1)"
    )
    args = parser.parse_args()

    run_all(version=args.version)


if __name__ == "__main__":
    main()

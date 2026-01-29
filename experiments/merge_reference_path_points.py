#!/usr/bin/env python3
"""
Merge close points in reference_path of SatNav dataset.

This script merges points in reference_path that are closer than a given threshold.
It uses an iterative approach: if consecutive points are within the threshold distance,
they are merged into their midpoint, and the process continues until no more merges
are possible.

Usage:
    python merge_reference_path_points.py input.json output.json --threshold 10.0

Example:
    python merge_reference_path_points.py \
        ../tests/test_data/satnav_dataset.json \
        ../tests/test_data/satnav_dataset_merged.json \
        --threshold 10.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satnav.core.utils import geodesic_distance


def compute_midpoint(
    point_a: List[float], 
    point_b: List[float]
) -> List[float]:
    """Compute the midpoint of two GPS coordinates.
    
    Args:
        point_a: First point as [longitude, latitude, altitude].
        point_b: Second point as [longitude, latitude, altitude].
        
    Returns:
        Midpoint as [longitude, latitude, altitude].
    """
    mid_lon = (point_a[0] + point_b[0]) / 2.0
    mid_lat = (point_a[1] + point_b[1]) / 2.0
    mid_alt = (point_a[2] + point_b[2]) / 2.0
    return [mid_lon, mid_lat, mid_alt]


def merge_close_points(
    reference_path: List[List[float]], 
    threshold_meters: float
) -> Tuple[List[List[float]], int]:
    """Merge close points in reference path iteratively.
    
    Uses iterative approach (方案A): if points A and B are close, merge them into
    midpoint AB. Then if AB and C are still close, merge them, and so on.
    
    Args:
        reference_path: List of points, each as [longitude, latitude, altitude].
        threshold_meters: Distance threshold in meters. Points closer than this
            will be merged.
            
    Returns:
        Tuple of (merged_path, merge_count) where merged_path is the optimized
        reference path and merge_count is the number of merges performed.
    """
    if len(reference_path) <= 1:
        return reference_path, 0
    
    # Work with a copy
    path = [point.copy() for point in reference_path]
    total_merges = 0
    
    # Keep merging until no more merges are possible
    merged = True
    while merged:
        merged = False
        new_path = []
        i = 0
        
        while i < len(path):
            if i == len(path) - 1:
                # Last point, just add it
                new_path.append(path[i])
                i += 1
            else:
                # Check distance to next point
                dist = geodesic_distance(path[i], path[i + 1])
                
                if dist < threshold_meters:
                    # Merge these two points into midpoint
                    midpoint = compute_midpoint(path[i], path[i + 1])
                    new_path.append(midpoint)
                    total_merges += 1
                    merged = True
                    i += 2  # Skip both points
                else:
                    # Keep current point
                    new_path.append(path[i])
                    i += 1
        
        path = new_path
    
    return path, total_merges


def process_dataset(
    input_path: str, 
    output_path: str, 
    threshold_meters: float
) -> dict:
    """Process a SatNav dataset JSON file and merge close reference path points.
    
    Args:
        input_path: Path to input JSON file.
        output_path: Path to output JSON file.
        threshold_meters: Distance threshold in meters for merging.
        
    Returns:
        Statistics dictionary with processing results.
    """
    # Load input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_episodes': 0,
        'total_original_points': 0,
        'total_merged_points': 0,
        'total_merges': 0,
        'episodes_modified': 0,
    }
    
    # Process each episode
    if 'episodes' in data:
        stats['total_episodes'] = len(data['episodes'])
        
        for episode in data['episodes']:
            if 'reference_path' in episode:
                original_path = episode['reference_path']
                original_count = len(original_path)
                
                merged_path, merge_count = merge_close_points(
                    original_path, 
                    threshold_meters
                )
                
                episode['reference_path'] = merged_path
                
                stats['total_original_points'] += original_count
                stats['total_merged_points'] += len(merged_path)
                stats['total_merges'] += merge_count
                
                if merge_count > 0:
                    stats['episodes_modified'] += 1
                    
                    # Print per-episode info
                    episode_id = episode.get('episode_id', 'unknown')
                    print(f"  Episode {episode_id}: "
                          f"{original_count} -> {len(merged_path)} points "
                          f"({merge_count} merges)")
    
    # Save output JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Merge close points in reference_path of SatNav dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_reference_path_points.py input.json output.json
  python merge_reference_path_points.py input.json output.json --threshold 5.0
        """
    )
    
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to input SatNav dataset JSON file'
    )
    
    parser.add_argument(
        'output_json',
        type=str,
        help='Path to output JSON file with merged reference paths'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=10.0,
        help='Distance threshold in meters for merging points (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {args.input_json}")
    print(f"Threshold: {args.threshold} meters")
    print("-" * 50)
    
    # Process the dataset
    stats = process_dataset(args.input_json, args.output_json, args.threshold)
    
    # Print summary
    print("-" * 50)
    print("Summary:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Episodes modified: {stats['episodes_modified']}")
    print(f"  Original points: {stats['total_original_points']}")
    print(f"  Merged points: {stats['total_merged_points']}")
    print(f"  Points reduced: {stats['total_original_points'] - stats['total_merged_points']}")
    print(f"  Total merges performed: {stats['total_merges']}")
    print(f"\nOutput saved to: {args.output_json}")


if __name__ == '__main__':
    main()

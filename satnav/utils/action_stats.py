#!/usr/bin/env python3
"""Action distribution statistics from VLN datasets.

This module provides utilities to compute action distributions from training
datasets, which can be used for random baseline agents.

Reference:
    - VLN-CE: vlnce_baselines/nonlearning_agents.py (RandomAgent uses fixed probs)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union


def compute_action_distribution(dataset_path: Union[str, Path]) -> Dict:
    """Compute action distribution from a SatNav dataset.
    
    This function loads a SatNav dataset JSON file and computes the frequency
    of each action from the reference paths. The distribution can be used to
    create a random agent that samples actions according to the dataset's
    natural distribution.
    
    Args:
        dataset_path: Path to the dataset JSON file (e.g., train.json).
    
    Returns:
        Dictionary containing:
            - 'probs': List of probabilities [p_stop, p_forward, p_left, p_right]
            - 'counts': Dictionary mapping action index to count
            - 'total': Total number of actions in the dataset
    
    Example:
        >>> dist = compute_action_distribution('data/debug_data/train/train.json')
        >>> print(f"Action probabilities: {dist['probs']}")
        Action probabilities: [0.02, 0.68, 0.15, 0.15]
        >>> print(f"MOVE_FORWARD appears {dist['counts'][1]} times")
    
    Note:
        Action indices follow the convention in satnav/task/actions.py:
        - 0: STOP
        - 1: MOVE_FORWARD
        - 2: TURN_LEFT
        - 3: TURN_RIGHT
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract actions from all episodes
    action_list = []
    
    for episode in data.get('episodes', []):
        # Get reference path waypoints
        ref_path = episode.get('reference_path', [])
        
        # Each waypoint may have an 'action' field (single action)
        # This is the action to take AT this waypoint to reach the next one
        for waypoint in ref_path:
            if isinstance(waypoint, dict) and 'action' in waypoint:
                action_list.append(waypoint['action'])
    
    if not action_list:
        raise ValueError(
            f"No actions found in dataset {dataset_path}. "
            "Make sure the dataset has 'action' fields in reference_path waypoints."
        )
    
    # Count action frequencies
    action_counts = Counter(action_list)
    total_actions = len(action_list)
    
    # Compute probabilities for each action (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
    probs = []
    counts_dict = {}
    
    for action_idx in range(4):  # Assuming 4 actions
        count = action_counts.get(action_idx, 0)
        counts_dict[action_idx] = count
        probs.append(count / total_actions if total_actions > 0 else 0.0)
    
    return {
        'probs': probs,
        'counts': counts_dict,
        'total': total_actions,
    }


def print_action_statistics(dataset_path: Union[str, Path]) -> None:
    """Print action statistics from a dataset in a readable format.
    
    Args:
        dataset_path: Path to the dataset JSON file.
    
    Example:
        >>> print_action_statistics('data/debug_data/train/train.json')
        Action Statistics for data/debug_data/train/train.json
        ========================================
        Total actions: 150
        
        Action Distribution:
          STOP         (0):   3 (  2.0%)
          MOVE_FORWARD (1): 102 ( 68.0%)
          TURN_LEFT    (2):  22 ( 14.7%)
          TURN_RIGHT   (3):  23 ( 15.3%)
    """
    dist = compute_action_distribution(dataset_path)
    
    action_names = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    
    print(f"\nAction Statistics for {dataset_path}")
    print("=" * 60)
    print(f"Total actions: {dist['total']}")
    print("\nAction Distribution:")
    
    for idx, name in enumerate(action_names):
        count = dist['counts'][idx]
        prob = dist['probs'][idx]
        print(f"  {name:12s} ({idx}): {count:4d} ({prob*100:5.1f}%)")
    
    print()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Default to debug training data
        dataset_path = "data/debug_data/train/train.json"
    
    try:
        print_action_statistics(dataset_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


"""Utility functions for trajectory generation."""

from typing import Union


# Action mapping: SatNav -> StreamVLN encoding
ACTION_MAPPING = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}

# Initial action placeholder
INITIAL_ACTION = -1


def satnav_action_to_streamvln(action: Union[str, int]) -> int:
    """Convert SatNav action to StreamVLN encoding.
    
    Args:
        action: SatNav action string (e.g., "MOVE_FORWARD") or action index.
        
    Returns:
        StreamVLN action encoding:
            -1: Initial state (placeholder)
            0: STOP
            1: MOVE_FORWARD
            2: TURN_LEFT
            3: TURN_RIGHT
            
    Raises:
        ValueError: If action is invalid.
    """
    if isinstance(action, int):
        # Already encoded, validate
        if action not in [-1, 0, 1, 2, 3]:
            raise ValueError(f"Invalid action index: {action}")
        return action
    
    if action not in ACTION_MAPPING:
        raise ValueError(
            f"Invalid action: {action}. "
            f"Valid actions: {list(ACTION_MAPPING.keys())}"
        )
    
    return ACTION_MAPPING[action]


def format_episode_dirname(scene_id: str, dataset_name: str, episode_idx: int) -> str:
    """Format episode directory name.
    
    Args:
        scene_id: Scene ID (e.g., "MN2").
        dataset_name: Dataset name (e.g., "satnav").
        episode_idx: Episode index (integer).
        
    Returns:
        Directory name in format: "{scene_id}_{dataset_name}_{episode_idx:06d}"
        
    Example:
        >>> format_episode_dirname("MN2", "satnav", 5)
        'MN2_satnav_000005'
    """
    return f"{scene_id}_{dataset_name}_{episode_idx:06d}"

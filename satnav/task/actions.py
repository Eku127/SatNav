#!/usr/bin/env python3
"""Action definitions for SatNav VLN tasks."""


class Action:
    """Action constants for VLN navigation tasks.
    
    This class defines the available actions that an agent can take in the
    SatNav environment. Actions are represented as string constants.
    
    Attributes:
        STOP: Stop action - agent stops moving and ends the episode.
        MOVE_FORWARD: Move forward action - agent moves forward by a fixed step size.
        TURN_LEFT: Turn left action - agent rotates left by a fixed angle.
        TURN_RIGHT: Turn right action - agent rotates right by a fixed angle.
    """
    
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    
    # List of all available actions
    ALL_ACTIONS = [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
    
    @classmethod
    def is_valid_action(cls, action: str) -> bool:
        """Check if an action string is valid.
        
        Args:
            action: Action string to validate.
            
        Returns:
            True if the action is valid, False otherwise.
        """
        return action in cls.ALL_ACTIONS
    
    @classmethod
    def get_action_index(cls, action: str) -> int:
        """Get the index of an action in the action list.
        
        Args:
            action: Action string.
            
        Returns:
            Index of the action in ALL_ACTIONS list.
            
        Raises:
            ValueError: If the action is not valid.
        """
        if not cls.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}. Valid actions are: {cls.ALL_ACTIONS}")
        return cls.ALL_ACTIONS.index(action)
    
    @classmethod
    def get_action_from_index(cls, index: int) -> str:
        """Get action string from index.
        
        Args:
            index: Index in ALL_ACTIONS list.
            
        Returns:
            Action string.
            
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(cls.ALL_ACTIONS):
            raise IndexError(f"Action index {index} out of range [0, {len(cls.ALL_ACTIONS)})")
        return cls.ALL_ACTIONS[index]


#!/usr/bin/env python3
"""Trainer registry for SatNav.

This module provides a registry system for managing different trainer classes,
similar to VLN-CE's baseline_registry system.
"""

from typing import Type, Dict


# Global registry dictionary
_TRAINERS: Dict[str, Type] = {}


def register_trainer(name: str):
    """Decorator to register a trainer class.
    
    Usage:
        @register_trainer("my_trainer")
        class MyTrainer(BaseILTrainer):
            ...
    
    Args:
        name: The name to register the trainer under
        
    Returns:
        Decorator function that registers the class
    """
    def decorator(cls):
        if name in _TRAINERS:
            raise ValueError(
                f"Trainer '{name}' is already registered. "
                f"Existing: {_TRAINERS[name]}, New: {cls}"
            )
        _TRAINERS[name] = cls
        return cls
    return decorator


def get_trainer(name: str) -> Type:
    """Get a trainer class by name.
    
    Args:
        name: The name of the trainer to retrieve
        
    Returns:
        The trainer class
        
    Raises:
        ValueError: If the trainer name is not registered
    """
    if name not in _TRAINERS:
        available = ', '.join(sorted(_TRAINERS.keys()))
        raise ValueError(
            f"Trainer '{name}' is not registered.\n"
            f"Available trainers: {available if available else 'None'}\n"
            f"Make sure to import the trainer module to trigger registration."
        )
    return _TRAINERS[name]


def list_trainers() -> list:
    """List all registered trainer names.
    
    Returns:
        List of registered trainer names
    """
    return sorted(_TRAINERS.keys())


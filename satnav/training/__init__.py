#!/usr/bin/env python3
"""Training module for SatNav VLN tasks.

This module provides trainers and utilities for training navigation models
using imitation learning.

The trainer registry system allows trainers to be registered and retrieved
by name, similar to VLN-CE's baseline_registry.
"""

from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.registry import register_trainer, get_trainer
from satnav.training.utils import collate_fn, pad_helper

__all__ = [
    # Base classes
    "BaseILTrainer",
    
    # Trainers
    "OfflineTrainer",
    "RecollectTrainer",
    
    # Evaluator
    "Evaluator",
    
    # Registry functions
    "register_trainer",
    "get_trainer",
    
    # Utilities
    "collate_fn",
    "pad_helper",
]


def __getattr__(name):
    if name == "Evaluator":
        from satnav.training.evaluator import Evaluator

        return Evaluator
    if name == "OfflineTrainer":
        from satnav.training.offline_trainer import OfflineTrainer

        return OfflineTrainer
    if name == "RecollectTrainer":
        from satnav.training.recollect_trainer import RecollectTrainer

        return RecollectTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

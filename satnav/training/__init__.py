#!/usr/bin/env python3
"""Training module for SatNav VLN tasks.

This module provides trainers and utilities for training navigation models
using imitation learning.

The trainer registry system allows trainers to be registered and retrieved
by name, similar to VLN-CE's baseline_registry.
"""

from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.utils import collate_fn, pad_helper, tokenize_instruction
from satnav.training.registry import register_trainer, get_trainer

# Import trainers to trigger registration via @register_trainer decorator
from satnav.training.recollect_trainer import RecollectTrainer

__all__ = [
    # Base classes
    "BaseILTrainer",
    
    # Trainers
    "RecollectTrainer",
    
    # Registry functions
    "register_trainer",
    "get_trainer",
    
    # Utilities
    "collate_fn",
    "pad_helper",
    "tokenize_instruction",
]

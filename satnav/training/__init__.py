#!/usr/bin/env python3
"""Training module for SatNav VLN tasks.

This module provides trainers and utilities for training navigation models
using imitation learning.
"""

from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.utils import collate_fn, pad_helper, tokenize_instruction

# Lazy imports to avoid RuntimeWarning when running as module
# These will be imported on first access
def _lazy_import_recollect_trainer():
    from satnav.training.recollect_trainer import RecollectTrainer
    return RecollectTrainer

def _lazy_import_recollection_dataset():
    from satnav.dataset.recollect_dataset import RecollectionDataset
    return RecollectionDataset

# Use __getattr__ for lazy imports (Python 3.7+)
def __getattr__(name):
    if name == "RecollectTrainer":
        return _lazy_import_recollect_trainer()
    elif name == "RecollectionDataset":
        return _lazy_import_recollection_dataset()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseILTrainer",
    "RecollectTrainer",
    "RecollectionDataset",
    "collate_fn",
    "pad_helper",
    "tokenize_instruction",
]


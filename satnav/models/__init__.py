"""SatNav models package.

This package contains baseline models, encoders, and utilities for
Vision-and-Language Navigation in continuous environments.

Organization:
    - base.py: Base classes for policies and networks
    - registry.py: Model registration system
    - baselines/: Baseline model implementations (Seq2Seq, CMA, etc.)
    - encoders/: Shared encoder modules (instruction, visual, state)
    - utils.py: Model utilities

Reference:
    - MISSING_MODULES.md Section 7
"""

from satnav.models.base import ILPolicy, Net, CategoricalNet, FixedCategorical
from satnav.models.registry import ModelRegistry

# Register baseline models
from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy
from satnav.models.baselines.cma_policy import CMAPolicy
from satnav.models.baselines.random_agent import RandomAgent
from satnav.models.baselines.greedy_agent import GreedyAgent

ModelRegistry.register_baseline("seq2seq", Seq2SeqPolicy)
ModelRegistry.register_baseline("cma", CMAPolicy)
ModelRegistry.register_baseline("random", RandomAgent)
ModelRegistry.register_baseline("greedy", GreedyAgent)

__all__ = [
    "ILPolicy",
    "Net",
    "CategoricalNet",
    "FixedCategorical",
    "ModelRegistry",
    "Seq2SeqPolicy",
    "CMAPolicy",
    "RandomAgent",
    "GreedyAgent",
]

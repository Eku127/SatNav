"""Dependency-light classic policy adapters."""

from baselines.classic.agents.random import RandomAdapter
from baselines.classic.agents.reference_follower import ReferenceFollowerAdapter

__all__ = ["RandomAdapter", "ReferenceFollowerAdapter"]

"""SatNav Trajectory Generation Application.

This module generates trajectory data from SatNav episodes for training
StreamVLN models. It follows reference paths and saves RGB images along
with action sequences in StreamVLN-compatible format.
"""

from .runner import SatNavTrajectoryRunner

__all__ = ['SatNavTrajectoryRunner']

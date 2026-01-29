#!/usr/bin/env python3
"""Simulator module for SatNav."""

from typing import Optional, Union

from omegaconf import DictConfig

from satnav.core.simulator import Simulator
from satnav.sims.satsim_wrapper import SatSimWrapper


def create_simulator(
    config: Union[DictConfig, dict],
    scenes_dir: Optional[str] = None
) -> Simulator:
    """Factory function to create a simulator based on configuration.
    
    Currently only supports SatSim (satellite imagery based simulator).
    
    Args:
        config: Configuration dictionary or DictConfig containing:
            - SIMULATOR: Simulator configuration
        scenes_dir: Optional path to scenes directory.
        
    Returns:
        Simulator instance (SatSimWrapper).
    """
    return SatSimWrapper(config, scenes_dir)


__all__ = ["create_simulator", "SatSimWrapper", "Simulator"]

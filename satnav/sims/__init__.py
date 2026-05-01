#!/usr/bin/env python3
"""Simulator module for SatNav.

This module provides a factory function to create simulators based on configuration.
Supported simulator types:
    - satsim: 2D satellite imagery based simulator (SatSim)
"""

from typing import Optional, Union

from omegaconf import DictConfig

from satnav.core.simulator import Simulator
from satnav.sims.satsim_wrapper import SatSimWrapper


def create_simulator(
    config: Union[DictConfig, dict],
    scenes_dir: Optional[str] = None
) -> Simulator:
    """Factory function to create a simulator based on configuration.
    
    The simulator type is determined by the SIMULATOR.TYPE field in config.
    Only "satsim" is supported.
    
    Args:
        config: Configuration dictionary or DictConfig containing:
            - SIMULATOR: Simulator configuration
                - TYPE: Simulator type ("satsim")
                - FORWARD_STEP_SIZE: Step size for forward movement
                - TURN_ANGLE: Angle for turning
                - RGB_SENSOR: Camera configuration
        scenes_dir: Optional path to scenes directory.
        
    Returns:
        Simulator instance (SatSimWrapper).
        
    Raises:
        ValueError: If simulator type is unknown.
    """
    # Get simulator type from config
    if isinstance(config, DictConfig):
        sim_config = getattr(config, "SIMULATOR", config)
        sim_type = getattr(sim_config, "TYPE", "satsim").lower()
    else:
        sim_config = config.get("SIMULATOR", config)
        sim_type = sim_config.get("TYPE", "satsim").lower()
    
    if sim_type == "satsim":
        return SatSimWrapper(config, scenes_dir)

    raise ValueError(
        f"Unknown simulator type: '{sim_type}'. Supported type: 'satsim'"
    )


__all__ = ["create_simulator", "SatSimWrapper", "Simulator"]

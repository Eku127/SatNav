#!/usr/bin/env python3
"""Simulator module for SatNav.

This module provides a factory function to create simulators based on configuration.
Supported simulator types:
    - satsim: 2D satellite imagery based simulator (SatSim)
    - aerialsim: 3D aerial view simulator using CesiumJS and Google 3D Tiles
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
    
    The simulator type is determined by the SIMULATOR.TYPE field in config:
        - "satsim" (default): 2D satellite imagery simulator
        - "aerialsim": 3D aerial view simulator using CesiumJS
    
    Args:
        config: Configuration dictionary or DictConfig containing:
            - SIMULATOR: Simulator configuration
                - TYPE: Simulator type ("satsim" or "aerialsim")
                - FORWARD_STEP_SIZE: Step size for forward movement
                - TURN_ANGLE: Angle for turning
                - RGB_SENSOR: Camera configuration
                - AERIAL: AerialSim-specific settings (for aerialsim only)
        scenes_dir: Optional path to scenes directory.
        
    Returns:
        Simulator instance (SatSimWrapper or AerialSimWrapper).
        
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
    
    # Create simulator based on type
    if sim_type == "satsim":
        return SatSimWrapper(config, scenes_dir)
    
    elif sim_type == "aerialsim":
        # Import AerialSimWrapper only when needed (avoids selenium dependency for satsim)
        from satnav.sims.aerialsim_wrapper import AerialSimWrapper
        return AerialSimWrapper(config, scenes_dir)
    
    else:
        raise ValueError(
            f"Unknown simulator type: '{sim_type}'. "
            f"Supported types: 'satsim', 'aerialsim'"
        )


__all__ = ["create_simulator", "SatSimWrapper", "Simulator"]

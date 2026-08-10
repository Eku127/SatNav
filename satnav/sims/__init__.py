#!/usr/bin/env python3
"""Simulator factory for built-in and external SatNav backends."""

import importlib
from typing import Optional, Type, Union

from omegaconf import DictConfig

from satnav.core.simulator import Simulator


def _get_satsim_wrapper():
    """Import SatSim lazily so external backends do not load raster deps."""
    from satnav.sims.satsim_wrapper import SatSimWrapper

    return SatSimWrapper


def _load_simulator_class(class_path: str) -> Type[Simulator]:
    """Load and validate a fully-qualified external simulator class."""
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator or not module_name or not class_name:
        raise ValueError(
            "SIMULATOR.CLASS must be a fully-qualified class name such as "
            "'package.module.SimulatorClass'"
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Could not import simulator module '{module_name}'"
        ) from error

    try:
        simulator_class = getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(
            f"Simulator class '{class_name}' was not found in '{module_name}'"
        ) from error

    if not isinstance(simulator_class, type) or not issubclass(
        simulator_class, Simulator
    ):
        raise TypeError(
            f"SIMULATOR.CLASS '{class_path}' must inherit "
            "satnav.core.simulator.Simulator"
        )
    return simulator_class


def create_simulator(
    config: Union[DictConfig, dict],
    scenes_dir: Optional[str] = None
) -> Simulator:
    """Factory function to create a simulator based on configuration.
    
    ``TYPE=satsim`` selects the built-in 2D renderer.  Other types must provide
    ``SIMULATOR.CLASS=package.module.SimulatorClass``; the class must inherit
    :class:`satnav.core.simulator.Simulator` and accept ``(config, scenes_dir)``.
    
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
        sim_type = str(getattr(sim_config, "TYPE", "satsim")).lower()
    else:
        sim_config = config.get("SIMULATOR", config)
        sim_type = str(sim_config.get("TYPE", "satsim")).lower()
    
    if sim_type == "satsim":
        return _get_satsim_wrapper()(config, scenes_dir)

    if isinstance(sim_config, DictConfig):
        class_path = getattr(sim_config, "CLASS", None)
    else:
        class_path = sim_config.get("CLASS")
    if not class_path:
        raise ValueError(
            f"Unknown simulator type '{sim_type}'. Configure "
            "SIMULATOR.CLASS with a fully-qualified Simulator subclass."
        )

    simulator_class = _load_simulator_class(str(class_path))
    simulator = simulator_class(config, scenes_dir)
    if not isinstance(simulator, Simulator):
        raise TypeError(
            f"SIMULATOR.CLASS '{class_path}' returned a non-Simulator instance"
        )
    return simulator


def __getattr__(name):
    if name == "SatSimWrapper":
        return _get_satsim_wrapper()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_simulator", "SatSimWrapper", "Simulator"]

"""Core components for SatNav."""

from satnav.core.config import load_config, save_config
from satnav.core.episode import (
    InstructionData,
    NavigationGoal,
    VLNEpisode,
)
from satnav.core.simulator import (
    AgentState,
    Simulator,
    Observations,
)
from satnav.core.utils import (
    geodesic_distance,
    geodesic_distance_with_altitude,
    EARTH_RADIUS_METERS,
)

# Import Env lazily to avoid circular import with dataset
def _get_env():
    """Lazy import of Env to avoid circular import."""
    from satnav.core.env import Env
    return Env

# Make Env available via __getattr__ for lazy loading
def __getattr__(name):
    if name == "Env":
        return _get_env()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "Env",
    "InstructionData",
    "NavigationGoal",
    "VLNEpisode",
    "AgentState",
    "Simulator",
    "Observations",
    "geodesic_distance",
    "geodesic_distance_with_altitude",
    "EARTH_RADIUS_METERS",
    "load_config",
    "save_config",
]


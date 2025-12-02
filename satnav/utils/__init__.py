#!/usr/bin/env python3
"""Utility modules for SatNav."""

# Use lazy imports to avoid import errors when optional dependencies are missing
__all__ = [
    "TOP_DOWN_MAP_COLORS",
    "draw_agent",
    "draw_path",
    "draw_point",
    "draw_circle_outline",
    "draw_reference_path",
    "draw_source_and_target",
    "geo_to_pixel",
    "crop_satellite_map",
]


def __getattr__(name):
    """Lazy import of map utilities."""
    if name in __all__:
        from satnav.utils import maps
        return getattr(maps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


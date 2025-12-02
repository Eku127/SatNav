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
    "draw_camera_view_bounds",
    "geo_to_pixel",
    "crop_satellite_map",
    "make_video",
    "create_video_frame",
    "add_instruction_text",
    "annotate_topdown_map",
    # Example utilities
    "setup_example",
    "print_episode_info",
    "prepare_waypoints",
    "print_waypoints_info",
    "generate_video",
    "save_visualizations",
    "print_metrics",
]


def __getattr__(name):
    """Lazy import of utilities."""
    if name in __all__:
        # Try maps first
        try:
            from satnav.utils import maps
            if hasattr(maps, name):
                return getattr(maps, name)
        except ImportError:
            pass
        
        # Try examples
        try:
            from satnav.utils import examples
            if hasattr(examples, name):
                return getattr(examples, name)
        except ImportError:
            pass
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


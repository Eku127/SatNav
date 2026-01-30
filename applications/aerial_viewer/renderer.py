#!/usr/bin/env python3
"""Aerial view renderer using CesiumJS and Google 3D Tiles.

This module provides a high-level interface for rendering 3D aerial views.
The actual rendering is performed by the AerialSim engine in satnav.sims.aerialsim.

For standalone usage (CLI), use this module directly.
For simulator integration, use satnav.sims.aerialsim_wrapper.AerialSimWrapper.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

# Import AerialSim engine from core
from satnav.sims.aerialsim import AerialSim


class AerialRenderer:
    """Non-interactive aerial view renderer.
    
    This class provides a high-level interface for rendering 3D aerial views
    using the AerialSim engine. It supports:
    - Single frame rendering with SatSim-compatible parameters
    - Sequence rendering for multiple waypoints
    - Configuration via YAML files
    """
    
    def __init__(self, config_path: str):
        """Initialize the renderer.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = OmegaConf.load(config_path)
        self._aerialsim: Optional[AerialSim] = None
        
        # Build simulator config from viewer config
        self._sim_config = self._build_sim_config()
    
    def _build_sim_config(self) -> dict:
        """Build simulator config from viewer config.
        
        Returns:
            Simulator configuration dictionary.
        """
        config = {
            "FORWARD_STEP_SIZE": 10.0,
            "TURN_ANGLE": 15.0,
            "RGB_SENSOR": {
                "WIDTH": self.config.CAMERA.WIDTH,
                "HEIGHT": self.config.CAMERA.HEIGHT,
                "HFOV": self.config.CAMERA.HFOV,
            },
            "AERIAL": {
                "API_KEY": self.config.API.API_KEY,
                "BROWSER": self.config.BROWSER.DRIVER,
                "HEADLESS": self.config.BROWSER.HEADLESS,
            }
        }
        return config
    
    def _ensure_aerialsim(self) -> AerialSim:
        """Ensure AerialSim engine is initialized.
        
        Returns:
            AerialSim instance.
        """
        if self._aerialsim is None:
            self._aerialsim = AerialSim(self._sim_config)
        return self._aerialsim
    
    def render(
        self,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        height: Optional[float] = None,
        heading: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        hfov: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> str:
        """Render aerial view and save as image.
        
        Args:
            lat: Latitude (degrees). If None, uses config value.
            lng: Longitude (degrees). If None, uses config value.
            height: Height above ground (meters). If None, uses config value.
            heading: Heading angle (degrees). If None, uses config value.
            pitch: Pitch angle (degrees). Ignored, always -90 (vertical down).
            roll: Roll angle (degrees). Ignored, always 0.
            hfov: Horizontal field of view (degrees). If None, uses config value.
            output_path: Output image path. If None, uses config value.
            
        Returns:
            Path to the saved image.
        """
        # Use config values if not provided
        agent = self.config.AGENT
        camera = self.config.CAMERA
        
        lat = lat if lat is not None else agent.LATITUDE
        lng = lng if lng is not None else agent.LONGITUDE
        height = height if height is not None else agent.ALTITUDE
        heading = heading if heading is not None else agent.ROTATION
        hfov = hfov if hfov is not None else camera.HFOV
        output_path = output_path or self.config.RENDERING.OUTPUT_PATH
        
        print("Starting aerial view rendering...")
        print(f"  Position: lat={lat:.6f}, lng={lng:.6f}, height={height:.1f}m")
        print(f"  Heading: {heading:.1f}°, HFOV: {hfov:.1f}°")
        
        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {output_file}")
        
        # Ensure engine is initialized
        aerialsim = self._ensure_aerialsim()
        
        # Set agent state
        position = [lng, lat, height]
        aerialsim.set_agent_state(position, heading)
        
        # Get observation (renders the view)
        obs = aerialsim.get_observations()
        rgb = obs["rgb"]
        
        # Save image
        from PIL import Image
        img = Image.fromarray(rgb)
        img.save(str(output_file))
        
        print(f"✓ Saved: {output_file}")
        print(f"  Size: {rgb.shape[1]}x{rgb.shape[0]}")
        
        return str(output_file)
    
    def render_satsim_compatible(
        self,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        altitude: Optional[float] = None,
        rotation: Optional[float] = None,
        hfov: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> str:
        """Render with SatSim-compatible parameters.
        
        This method provides a direct interface compatible with SatSim's camera model:
        - Always renders vertical down view (pitch = -90°)
        - Uses rotation as heading (0 = North)
        
        Args:
            longitude: Agent longitude (degrees).
            latitude: Agent latitude (degrees).
            altitude: Agent altitude above ground (meters).
            rotation: Agent heading (degrees, 0=North).
            hfov: Horizontal field of view (degrees).
            width: Output image width (pixels). Not supported, uses config.
            height: Output image height (pixels). Not supported, uses config.
            output_path: Output image path.
            
        Returns:
            Path to the saved image.
        """
        # Get values from config if not provided
        agent = self.config.get("AGENT", None)
        camera = self.config.get("CAMERA", None)
        
        longitude = longitude if longitude is not None else (agent.LONGITUDE if agent else None)
        latitude = latitude if latitude is not None else (agent.LATITUDE if agent else None)
        altitude = altitude if altitude is not None else (agent.ALTITUDE if agent else None)
        rotation = rotation if rotation is not None else (agent.ROTATION if agent else 0.0)
        hfov = hfov if hfov is not None else (camera.HFOV if camera else 90.0)
        
        if longitude is None or latitude is None or altitude is None:
            raise ValueError("longitude, latitude, and altitude must be provided or in config")
        
        return self.render(
            lat=latitude,
            lng=longitude,
            height=altitude,
            heading=rotation,
            hfov=hfov,
            output_path=output_path
        )
    
    def render_sequence(
        self,
        waypoints: List[list],
        rotations: Optional[List[float]] = None,
        altitude: float = 50.0,
        hfov: float = 90.0,
        output_dir: str = "output",
        output_prefix: str = "frame",
        reuse_ground_height: bool = True,
        ground_height_threshold: float = 200.0,
    ) -> List[str]:
        """Render a sequence of waypoints efficiently.
        
        This method optimizes rendering by reusing the browser instance
        across all frames.
        
        Args:
            waypoints: List of [longitude, latitude] or [longitude, latitude, altitude] points.
            rotations: List of rotation angles (degrees). If None, uses 0 for all.
            altitude: Default altitude if not specified in waypoints.
            hfov: Horizontal field of view (degrees).
            output_dir: Output directory for rendered images.
            output_prefix: Prefix for output filenames.
            reuse_ground_height: If True, reuse ground height for nearby points.
            ground_height_threshold: Distance threshold (meters) for reusing.
            
        Returns:
            List of output file paths.
        """
        from PIL import Image
        
        if not waypoints:
            return []
        
        # Default rotations
        if rotations is None:
            rotations = [0.0] * len(waypoints)
        elif len(rotations) < len(waypoints):
            rotations = rotations + [rotations[-1]] * (len(waypoints) - len(rotations))
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"=" * 60)
        print(f"Rendering sequence: {len(waypoints)} waypoints")
        print(f"Settings: altitude={altitude}m, HFOV={hfov}°")
        print(f"=" * 60)
        
        # Ensure engine is initialized
        aerialsim = self._ensure_aerialsim()
        
        # Update ground height threshold
        aerialsim._ground_height_threshold = ground_height_threshold
        
        output_files = []
        
        for i, wp in enumerate(waypoints):
            print(f"\n[Frame {i+1}/{len(waypoints)}]")
            
            # Parse waypoint
            lng = wp[0]
            lat = wp[1]
            wp_altitude = wp[2] if len(wp) > 2 else altitude
            rotation = rotations[i]
            
            print(f"  Position: ({lng:.6f}, {lat:.6f}), alt={wp_altitude}m, rot={rotation}°")
            
            # Set agent state
            position = [lng, lat, wp_altitude]
            aerialsim.set_agent_state(position, rotation)
            
            # Get observation
            obs = aerialsim.get_observations()
            rgb = obs["rgb"]
            
            # Save image
            final_path = str(output_path / f"{output_prefix}_{i:04d}.png")
            img = Image.fromarray(rgb)
            img.save(final_path)
            
            output_files.append(final_path)
            print(f"  ✓ Saved: {final_path}")
        
        print(f"\n{'=' * 60}")
        print(f"✓ Rendered {len(output_files)} frames to {output_dir}/")
        
        return output_files
    
    def close(self):
        """Close browser and cleanup."""
        if self._aerialsim is not None:
            self._aerialsim.close()
            self._aerialsim = None


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Render aerial 3D view using CesiumJS and Google 3D Tiles"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path (overrides config)"
    )
    parser.add_argument("--longitude", type=float, default=None, help="Longitude (degrees)")
    parser.add_argument("--latitude", type=float, default=None, help="Latitude (degrees)")
    parser.add_argument("--altitude", type=float, default=None, help="Altitude (meters)")
    parser.add_argument("--rotation", type=float, default=None, help="Rotation (degrees, 0=North)")
    parser.add_argument("--hfov", type=float, default=None, help="Horizontal FOV (degrees)")
    # Legacy arguments
    parser.add_argument("--lat", type=float, default=None, help="Latitude (legacy)")
    parser.add_argument("--lng", type=float, default=None, help="Longitude (legacy)")
    parser.add_argument("--height", type=float, default=None, help="Height (legacy)")
    parser.add_argument("--heading", type=float, default=None, help="Heading (legacy)")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return
    
    renderer = AerialRenderer(str(config_path))
    
    try:
        # Map legacy arguments
        longitude = args.longitude if args.longitude is not None else args.lng
        latitude = args.latitude if args.latitude is not None else args.lat
        altitude = args.altitude if args.altitude is not None else args.height
        rotation = args.rotation if args.rotation is not None else args.heading
        
        renderer.render_satsim_compatible(
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            rotation=rotation,
            hfov=args.hfov,
            output_path=args.output
        )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()

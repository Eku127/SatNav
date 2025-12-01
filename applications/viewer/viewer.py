#!/usr/bin/env python3
"""Interactive viewer for satellite maps using SatSim.

This application allows users to navigate through satellite maps using keyboard controls:
- 'w': Move forward
- 's': Move backward
- 'a': Turn left
- 'd': Turn right
- 'q': Move up (increase altitude)
- 'e': Move down (decrease altitude)
- 'ESC': Quit

The viewer displays:
- Current RGB observation from the satellite map
- Agent state information (WGS84 and Mercator coordinates)
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from omegaconf import OmegaConf

from satnav.sims.satsim import SatSim
from satnav.sims.satsim.geoutils import GeoUtils


class InteractiveViewer:
    """Interactive viewer for satellite maps."""
    
    def __init__(self, config_path: str):
        """Initialize the interactive viewer.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize simulator
        sim_config = self.config.SIMULATOR
        self.sim = SatSim(sim_config)
        
        # Load scene
        tif_path_str = self.config.TIF_PATH
        tif_path = Path(tif_path_str)
        
        # Find project root by looking for satnav directory
        def find_project_root(start_path: Path) -> Path:
            """Find project root by looking for satnav directory."""
            current = start_path.resolve()
            while current != current.parent:
                if (current / "satnav").exists() and (current / "satnav" / "__init__.py").exists():
                    return current
                current = current.parent
            # Fallback: use start_path's parent
            return start_path.parent.parent.parent if start_path.parts else Path.cwd()
        
        if not tif_path.is_absolute():
            config_dir = Path(config_path).parent.resolve()
            project_root = find_project_root(config_dir)
            
            # Try multiple paths
            candidates = [
                config_dir / tif_path,  # Relative to config file
                project_root / tif_path,  # Relative to project root
                Path.cwd() / tif_path,  # Relative to current working directory
            ]
            
            # Also try going up from config dir
            for i in range(3):
                candidates.append(config_dir.parents[i] / tif_path if i < len(config_dir.parents) else None)
            
            candidates = [c for c in candidates if c is not None]
            
            found = False
            for candidate in candidates:
                if candidate.exists():
                    tif_path = candidate.resolve()
                    found = True
                    break
            
            if not found:
                # Provide helpful error message
                error_msg = (
                    f"TIF file not found: {tif_path_str}\n"
                    f"Tried the following paths:\n"
                )
                for i, candidate in enumerate(candidates, 1):
                    error_msg += f"  {i}. {candidate}\n"
                error_msg += f"\nCurrent working directory: {Path.cwd()}\n"
                error_msg += f"Config file location: {config_dir}\n"
                error_msg += f"Project root: {project_root}"
                raise FileNotFoundError(error_msg)
        
        print(f"Loading scene: {tif_path}")
        self.sim.load_scene(str(tif_path))
        
        # Get scene bounds and calculate center
        bounds_wgs84 = self.sim.get_scene_bounds()
        left_lon, right_lon, bottom_lat, top_lat = bounds_wgs84
        
        # Calculate center in WGS84
        center_lon = (left_lon + right_lon) / 2.0
        center_lat = (bottom_lat + top_lat) / 2.0
        center_alt = self.config.AGENT.ALTITUDE
        
        print(f"Scene bounds: lon=[{left_lon:.6f}, {right_lon:.6f}], "
              f"lat=[{bottom_lat:.6f}, {top_lat:.6f}]")
        print(f"Initial position: [{center_lon:.6f}, {center_lat:.6f}, {center_alt:.2f}]")
        
        # Set initial agent state
        initial_rotation = self.config.AGENT.ROTATION
        initial_position = [center_lon, center_lat, center_alt]
        
        # Verify initial position is navigable (within safe bounds)
        # If center is not safe, this means the map is too small for the given altitude
        # In this case, we still set the position but user will see boundary warnings
        if not self.sim.is_navigable(initial_position):
            print(f"Warning: Initial center position is not within safe bounds.")
            print(f"  This means the map is too small for altitude {center_alt}m.")
            print(f"  You may not be able to move. Try reducing altitude (press 'e').")
            self.at_boundary = True
        else:
            self.at_boundary = False
        
        self.sim.set_agent_state(initial_position, initial_rotation)
        
        # Get altitude step size from config
        self.altitude_step_size = self.config.AGENT.get("ALTITUDE_STEP_SIZE", 10.0)
        
        # Display configuration
        self.window_name = self.config.DISPLAY.WINDOW_NAME
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # State tracking
        self.running = True
        self.at_boundary = False  # Track if agent is at boundary
        
    def get_agent_state_info(self) -> dict:
        """Get current agent state information.
        
        Returns:
            Dictionary containing state information in both WGS84 and Mercator.
        """
        # SatSim.get_agent_state() returns a tuple: (position_wgs84, rotation)
        position_wgs84, rotation = self.sim.get_agent_state()
        
        # Convert to Mercator
        position_mercator = GeoUtils.position_wgs84_to_mercator(
            np.array(position_wgs84, dtype=np.float32)
        )
        
        return {
            "wgs84": {
                "longitude": float(position_wgs84[0]),
                "latitude": float(position_wgs84[1]),
                "altitude": float(position_wgs84[2]),
            },
            "mercator": {
                "x": float(position_mercator[0]),
                "y": float(position_mercator[1]),
                "altitude": float(position_mercator[2]),
            },
            "rotation": float(rotation),
        }
    
    def draw_state_info(self, image: np.ndarray) -> np.ndarray:
        """Draw state information on the image.
        
        Args:
            image: Input RGB image.
            
        Returns:
            Image with state information overlaid.
        """
        state_info = self.get_agent_state_info()
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White text
        thickness = 2  # Thicker for better visibility without background
        line_height = 25
        y_start = 30
        
        # Add text shadow for better visibility (black outline)
        shadow_color = (0, 0, 0)
        shadow_thickness = 3
        
        # WGS84 coordinates
        text = "WGS84 Coordinates:"
        cv2.putText(image, text, (20, y_start), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start), font, font_scale, color, thickness, cv2.LINE_AA)
        
        text = f"  Longitude: {state_info['wgs84']['longitude']:.6f}"
        cv2.putText(image, text, (20, y_start + line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        text = f"  Latitude:  {state_info['wgs84']['latitude']:.6f}"
        cv2.putText(image, text, (20, y_start + 2 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 2 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        text = f"  Altitude:  {state_info['wgs84']['altitude']:.2f} m"
        cv2.putText(image, text, (20, y_start + 3 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 3 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Mercator coordinates
        text = "Mercator Coordinates:"
        cv2.putText(image, text, (20, y_start + 5 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 5 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        text = f"  X: {state_info['mercator']['x']:.2f} m"
        cv2.putText(image, text, (20, y_start + 6 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 6 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        text = f"  Y: {state_info['mercator']['y']:.2f} m"
        cv2.putText(image, text, (20, y_start + 7 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 7 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Rotation
        text = f"Rotation (Roll): {state_info['rotation']:.1f}"
        cv2.putText(image, text, (20, y_start + 8 * line_height), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_start + 8 * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Draw boundary warning if at boundary
        # Note: With the updated is_navigable() logic, this warning appears when
        # agent reaches the safe boundary (which accounts for camera view size)
        if self.at_boundary:
            warning_text = "WARNING: At safe boundary! Cannot move further."
            hint_text = "Try moving in a different direction or changing altitude"
            warning_y = y_start + 10 * line_height
            hint_y = y_start + 11 * line_height
            # Red text for warning
            warning_color = (0, 0, 255)  # Red in BGR
            warning_shadow = (0, 0, 0)  # Black shadow
            cv2.putText(image, warning_text, (20, warning_y), font, font_scale, warning_shadow, shadow_thickness + 1, cv2.LINE_AA)
            cv2.putText(image, warning_text, (20, warning_y), font, font_scale, warning_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(image, hint_text, (20, hint_y), font, font_scale * 0.8, warning_shadow, shadow_thickness, cv2.LINE_AA)
            cv2.putText(image, hint_text, (20, hint_y), font, font_scale * 0.8, (200, 200, 200), thickness, cv2.LINE_AA)
        
        # Controls hint
        h, w = image.shape[:2]
        text = "Controls: w=forward, s=backward, a=left, d=right, q=up, e=down, ESC=quit"
        cv2.putText(image, text, (20, h - 20), font, 0.5, shadow_color, 2, cv2.LINE_AA)
        cv2.putText(image, text, (20, h - 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return image
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey().
            
        Returns:
            True if should continue, False if should quit.
        """
        # Convert to lowercase
        key_char = chr(key & 0xFF).lower()
        
        if key_char == 'q':
            # Move up (increase altitude)
            # Note: Higher altitude means larger camera view, so we need to check
            # if the new position with increased altitude is still navigable
            # using the updated is_navigable() which uses safe bounds
            position_wgs84, rotation = self.sim.get_agent_state()
            new_altitude = position_wgs84[2] + self.altitude_step_size
            new_position = [position_wgs84[0], position_wgs84[1], new_altitude]
            # Check if new position with increased altitude is navigable
            # Higher altitude means larger camera view, so we need to check bounds
            if self.sim.is_navigable(new_position):
                self.sim.set_agent_state(new_position, rotation)
                self.at_boundary = False  # Reset boundary flag on successful move
            else:
                print("Warning: Cannot move up - position would exceed safe bounds at this altitude")
                self.at_boundary = True
        elif key_char == 'e':
            # Move down (decrease altitude)
            # Note: Lower altitude means smaller camera view, so this should usually succeed
            # but we still check using the updated is_navigable() for consistency
            position_wgs84, rotation = self.sim.get_agent_state()
            new_altitude = max(0.0, position_wgs84[2] - self.altitude_step_size)  # Don't go below 0
            new_position = [position_wgs84[0], position_wgs84[1], new_altitude]
            # Check if new position is navigable (should always pass for lower altitude, but check anyway)
            if self.sim.is_navigable(new_position):
                self.sim.set_agent_state(new_position, rotation)
                self.at_boundary = False  # Reset boundary flag on successful move
            else:
                print("Warning: Cannot move down - position would exceed safe bounds")
                self.at_boundary = True
        elif key_char == 'w':
            # Move forward
            # Note: step("MOVE_FORWARD") includes boundary checking internally using is_navigable().
            # If the new position is not within safe bounds, agent stays at current position.
            # We detect boundary hits by checking if position actually changed after step().
            position_before, rotation_before = self.sim.get_agent_state()
            try:
                self.sim.step("MOVE_FORWARD")
                position_after, rotation_after = self.sim.get_agent_state()
                
                # Check if position actually changed (movement was successful)
                # Compare longitude and latitude separately with appropriate tolerance
                # FORWARD_STEP_SIZE is typically 0.25m or 10m
                # 1e-6 degrees ≈ 0.11 meters, which should detect any meaningful movement
                lon_diff = abs(position_before[0] - position_after[0])
                lat_diff = abs(position_before[1] - position_after[1])
                position_changed = (lon_diff > 1e-6) or (lat_diff > 1e-6)
                
                if position_changed:
                    # Movement was successful - reset boundary flag
                    self.at_boundary = False
                else:
                    # Position didn't change - this means step() tried to move but the new position
                    # was not navigable (outside safe bounds), so agent stayed at current position.
                    # This indicates we've reached the safe boundary limit.
                    self.at_boundary = True
                    print("Warning: Cannot move forward - reached safe boundary limit")
            
            except (ValueError, RuntimeError) as e:
                # Handle other errors (e.g., agent state not initialized, camera view exceeds bounds)
                # These errors are rare with the new safe boundary checking, but handle them anyway
                error_msg = str(e)
                if "Camera view bounds exceed" in error_msg:
                    self.at_boundary = True
                    print("Warning: Cannot move forward - camera view would exceed map bounds")
                else:
                    print(f"Warning: Cannot move forward - {e}")
                    self.at_boundary = True
        elif key_char == 'a':
            # Turn left
            self.sim.step("TURN_LEFT")
        elif key_char == 'd':
            # Turn right
            self.sim.step("TURN_RIGHT")
        elif key_char == 's':
            # Move backward (opposite direction of current rotation)
            # Note: We manually calculate backward movement and check navigability
            # using the updated is_navigable() which uses safe bounds
            try:
                position_wgs84, rotation = self.sim.get_agent_state()
                
                # Convert to Mercator for movement calculation
                position_mercator = GeoUtils.position_wgs84_to_mercator(
                    np.array(position_wgs84, dtype=np.float32)
                )
                x, y, alt = position_mercator
                
                # Backward direction is opposite to current rotation
                backward_heading = (rotation + 180.0) % 360.0
                
                # Get forward step size from config
                forward_step_size = self.config.SIMULATOR.get("FORWARD_STEP_SIZE", 0.25)
                
                # Calculate new position
                x_new, y_new = GeoUtils.move_in_mercator(
                    x, y, forward_step_size, backward_heading
                )
                
                # Convert back to WGS84
                position_new_wgs84 = GeoUtils.position_mercator_to_wgs84(
                    (x_new, y_new, alt)
                )
                
                # Check if new position is navigable (uses safe bounds based on altitude)
                # The updated is_navigable() ensures camera view won't exceed map bounds
                if self.sim.is_navigable(position_new_wgs84):
                    # Update agent state
                    self.sim.set_agent_state(
                        position_new_wgs84.tolist(),
                        rotation
                    )
                    self.at_boundary = False  # Reset boundary flag on successful move
                else:
                    print("Warning: Cannot move backward - reached safe boundary limit")
                    self.at_boundary = True
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Cannot move backward - {e}")
                self.at_boundary = True
        
        return True
    
    def _render_and_display(self):
        """Render current observation and display it.
        
        This is called whenever the agent state changes or initially.
        """
        try:
            # Get observations
            observations = self.sim.get_observations()
            rgb_image = observations["rgb"]
            
            # Convert RGB to BGR for OpenCV (OpenCV uses BGR)
            rgb_image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Draw state information
            display_image = self.draw_state_info(rgb_image_bgr.copy())
            
            # Display image
            cv2.imshow(self.window_name, display_image)
        except ValueError as e:
            # Handle boundary errors or invalid camera view
            error_msg = str(e)
            if "Camera view bounds exceed" in error_msg:
                print(f"Warning: Camera view exceeds map bounds. Try moving away from the edge.")
                self.at_boundary = True
            else:
                print(f"Warning: {error_msg}")
            
            # Create error image to display boundary warning
            # Get image dimensions from config or use default
            img_height = self.config.SIMULATOR.RGB_SENSOR.get("HEIGHT", 480)
            img_width = self.config.SIMULATOR.RGB_SENSOR.get("WIDTH", 640)
            error_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            error_image_bgr = cv2.cvtColor(error_image, cv2.COLOR_RGB2BGR)
            
            # Draw warning message on error image
            # Note: With the updated is_navigable() logic, this error should rarely occur
            # as we prevent moving to unsafe positions. However, it might still happen
            # if altitude changes or other edge cases.
            font = cv2.FONT_HERSHEY_SIMPLEX
            warning_text = "Camera view exceeds map bounds!"
            instruction_text = "Use w/a/d/s to move, or q/e to change altitude"
            cv2.putText(error_image_bgr, warning_text, 
                       (50, img_height // 2 - 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(error_image_bgr, instruction_text, 
                       (50, img_height // 2 + 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw state info on error image
            display_image = self.draw_state_info(error_image_bgr.copy())
            cv2.imshow(self.window_name, display_image)
    
    def run(self):
        """Run the interactive viewer main loop.
        
        Uses event-driven rendering: only renders when agent state changes.
        No continuous loop rendering - waits for keyboard input.
        """
        print("\n" + "=" * 60)
        print("SatNav Interactive Viewer")
        print("=" * 60)
        print("Controls:")
        print("  'w' - Move forward")
        print("  's' - Move backward")
        print("  'a' - Turn left")
        print("  'd' - Turn right")
        print("  'q' - Move up (increase altitude)")
        print("  'e' - Move down (decrease altitude)")
        print("  'ESC' - Quit")
        print("=" * 60 + "\n")
        
        # Initial render
        self._render_and_display()
        
        # Event-driven loop: wait for keyboard input, then render
        while self.running:
            try:
                # Wait for keyboard input (no timeout - event-driven)
                key = cv2.waitKey(0) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key != 255:  # Some key was pressed
                    # Handle keyboard input (this may change agent state)
                    self.running = self.handle_keyboard(key)
                    # Re-render after state change
                    self._render_and_display()
                
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        self.sim.close()
        print("\nViewer closed.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive viewer for satellite maps"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create and run viewer
    viewer = InteractiveViewer(str(config_path))
    viewer.run()


if __name__ == "__main__":
    main()


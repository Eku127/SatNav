#!/usr/bin/env python3
"""Interactive task viewer for VLN tasks using SatNav.

This application allows users to navigate through VLN tasks using keyboard controls:
- 'w': Move forward
- 'a': Turn left
- 'd': Turn right
- 't': Toggle topdown view
- 'SPACE': Stop and show metrics
- 'ESC': Quit

The viewer displays:
- Left: Current RGB observation from the satellite map
- Right: Top-down map visualization with agent path and waypoints (optional)
- Bottom: Instruction text and distance information

Input: Task YAML configuration file (e.g., configs/vln_task.yaml)
"""

import sys
import textwrap
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from omegaconf import OmegaConf

from satnav.core.env import Env
from satnav.core.utils import geodesic_distance
from satnav.utils.maps import annotate_topdown_map
from satnav.utils.examples import prepare_waypoints, print_metrics


# Constants
SPACE_KEY = 32
ESC_KEY = 27
POSITION_TOLERANCE = 1e-6


class TaskViewer:
    """Interactive viewer for VLN tasks."""
    
    def __init__(self, config_path: str):
        """Initialize the task viewer.
        
        Args:
            config_path: Path to the task configuration YAML file.
        """
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Ensure TOP_DOWN_MAP is enabled in measurements
        self._ensure_topdown_map_enabled()
        
        # Create environment
        self.env = Env(self.config)
        
        # Reset environment to get first episode
        self.obs = self.env.reset()
        self.episode = self.env.current_episode
        
        print(f"\nLoaded episode: {self.episode.episode_id}")
        print(f"Scene: {self.episode.scene_id}")
        print(f"Instruction: {self.episode.instruction.instruction_text}")
        
        # Display configuration
        self.window_name = "SatNav Task Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Set initial window size (width, height) - larger for task viewer with multiple panels
        cv2.resizeWindow(self.window_name, 1600, 900)
        
        # State tracking
        self.running = True
        self.at_boundary = False
        self.step_count = 0
        self.action_history = []
        self.last_action = None
        self.show_topdown = True
        
        # Navigation state
        self.waypoints = prepare_waypoints(self.episode)
        self.current_waypoint_idx = 0
        self.goal_radius = self._get_config_value("TASK", "SUCCESS_DISTANCE", 10.0)
        
        print(f"  Waypoints: {len(self.waypoints)}")
        print(f"  Goal radius: {self.goal_radius}m")
    
    def _ensure_topdown_map_enabled(self):
        """Ensure TOP_DOWN_MAP is enabled in measurements."""
        if isinstance(self.config.TASK, dict):
            measurements = self.config.TASK.get("MEASUREMENTS", [])
        else:
            measurements = getattr(self.config.TASK, "MEASUREMENTS", [])
        
        if "TOP_DOWN_MAP" not in measurements:
            print("Warning: TOP_DOWN_MAP not in MEASUREMENTS. Adding it...")
            if isinstance(self.config.TASK, dict):
                self.config.TASK["MEASUREMENTS"] = list(measurements) + ["TOP_DOWN_MAP"]
            else:
                measurements_list = list(measurements)
                measurements_list.append("TOP_DOWN_MAP")
                self.config.TASK.MEASUREMENTS = measurements_list
    
    def _get_config_value(self, section: str, key: str, default):
        """Get configuration value with fallback."""
        if isinstance(self.config, dict):
            section_config = self.config.get(section, {})
            if isinstance(section_config, dict):
                return section_config.get(key, default)
            return getattr(section_config, key, default)
        else:
            section_config = getattr(self.config, section, {})
            return getattr(section_config, key, default)
    
    def _check_waypoint_reached(self):
        """Check if current waypoint has been reached and update index."""
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return
        
        agent_state = self.env._task._sim.get_agent_state()
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        
        current_distance = geodesic_distance(
            agent_state.position,
            current_waypoint
        )
        
        if current_distance <= self.goal_radius:
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
                print(f"  ✓ Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}")
            else:
                print(f"  ✓ Reached final goal!")
    
    def _handle_move_forward(self):
        """Handle move forward action with boundary checking."""
        action = "MOVE_FORWARD"
        agent_state_before = self.env._task._sim.get_agent_state()
        position_before = agent_state_before.position
        
        try:
            self.obs, done, info = self.env.step(action)
            agent_state_after = self.env._task._sim.get_agent_state()
            position_after = agent_state_after.position
            
            # Check if position actually changed (movement was successful)
            lon_diff = abs(position_before[0] - position_after[0])
            lat_diff = abs(position_before[1] - position_after[1])
            position_changed = (lon_diff > POSITION_TOLERANCE) or (lat_diff > POSITION_TOLERANCE)
            
            if position_changed:
                self.at_boundary = False
                self.step_count += 1
                self.action_history.append(action)
                self.last_action = action
                self._check_waypoint_reached()
            else:
                self.at_boundary = True
                print("Warning: Cannot move forward - reached safe boundary limit")
        
        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            self.at_boundary = True
            if "Camera view bounds exceed" in error_msg:
                print("Warning: Cannot move forward - camera view would exceed map bounds")
            else:
                print(f"Warning: Cannot move forward - {e}")
    
    def _handle_turn_action(self, action: str):
        """Handle turn left or right action."""
        self.obs, done, info = self.env.step(action)
        self.step_count += 1
        self.action_history.append(action)
        self.last_action = action
    
    def _handle_stop_action(self):
        """Handle STOP action: show metrics and load next episode."""
        action = "STOP"
        self.obs, done, info = self.env.step(action)
        self.step_count += 1
        self.action_history.append(action)
        self.last_action = action
        
        # Print metrics to console
        print_metrics(
            env=self.env,
            episode=self.episode,
            action_history=self.action_history,
            step_count=self.step_count,
            note=None
        )
        
        # Show metrics on screen and wait for key press
        self._show_metrics_and_wait()
        
        # Try to load next episode
        return self._load_next_episode()
    
    def _load_next_episode(self) -> bool:
        """Load next episode and reset state.
        
        Returns:
            True if next episode loaded successfully, False if no more episodes.
        """
        try:
            self.obs = self.env.reset()
            self.episode = self.env.current_episode
            
            # Reset state for new episode
            self.step_count = 0
            self.action_history = []
            self.last_action = None
            self.at_boundary = False
            self.waypoints = prepare_waypoints(self.episode)
            self.current_waypoint_idx = 0
            
            print(f"\nLoaded next episode: {self.episode.episode_id}")
            print(f"Scene: {self.episode.scene_id}")
            print(f"Instruction: {self.episode.instruction.instruction_text}")
            print(f"  Waypoints: {len(self.waypoints)}")
            print(f"  Goal radius: {self.goal_radius}m")
            
            return True
        
        except RuntimeError as e:
            if "All episodes exhausted" in str(e):
                print("\n" + "=" * 60)
                print("All episodes completed!")
                print("=" * 60)
                return False
            raise
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey().
            
        Returns:
            True if should continue, False if should quit.
        """
        key_char = chr(key & 0xFF).lower()
        
        if key_char == 'w':
            self._handle_move_forward()
        elif key_char == 'a':
            self._handle_turn_action("TURN_LEFT")
        elif key_char == 'd':
            self._handle_turn_action("TURN_RIGHT")
        elif key_char == 't':
            self.show_topdown = not self.show_topdown
            print(f"  Topdown view: {'ON' if self.show_topdown else 'OFF'}")
        elif key == SPACE_KEY:
            return self._handle_stop_action()
        
        return True
    
    def _show_metrics_and_wait(self):
        """Display metrics summary on screen and wait for key press."""
        metrics = self.env.get_metrics()
        success = metrics.get('success', 0.0)
        spl = metrics.get('spl', 0.0)
        distance = metrics.get('distance_to_goal', 0.0)
        path_length = metrics.get('path_length', 0.0)
        
        # Create display frame
        display_width = 800
        display_height = 600
        metrics_frame = np.ones((display_height, display_width, 3), dtype=np.uint8) * 240
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 0, 0)
        
        y_start = 50
        line_height = 40
        
        # Title
        title = "Evaluation Metrics"
        title_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        title_x = (display_width - title_size[0]) // 2
        cv2.putText(metrics_frame, title, (title_x, y_start), font, 1.2, color, 3, cv2.LINE_AA)
        
        y_pos = y_start + 60
        
        # Metrics
        metrics_text = [
            f"Success:           {success:.4f}",
            f"SPL:               {spl:.4f}",
            f"Distance to Goal:  {distance:.2f} m",
            f"Path Length:       {path_length:.2f} m",
            f"Steps:             {self.step_count}",
        ]
        
        for text in metrics_text:
            cv2.putText(metrics_frame, text, (50, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
            y_pos += line_height
        
        # Action distribution
        action_counts = {}
        for a in self.action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        action_text = f"Action Distribution: {action_counts}"
        cv2.putText(metrics_frame, action_text, (50, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
        y_pos += line_height * 2
        
        # Continue message
        continue_text = "Press any key for next episode..."
        continue_size = cv2.getTextSize(continue_text, font, 0.9, 2)[0]
        continue_x = (display_width - continue_size[0]) // 2
        cv2.putText(metrics_frame, continue_text, (continue_x, y_pos), font, 0.9, (0, 100, 200), 2, cv2.LINE_AA)
        
        # Display and wait
        cv2.imshow(self.window_name, metrics_frame)
        cv2.waitKey(0)
    
    def _add_episode_info_to_image(self, image: np.ndarray) -> np.ndarray:
        """Add episode ID and scene ID to top-left of image.
        
        Args:
            image: RGB image (H, W, 3).
            
        Returns:
            BGR image with episode info overlay.
        """
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)  # White
        shadow_color = (0, 0, 0)  # Black
        
        episode_text = f"Episode: {self.episode.episode_id}"
        scene_text = f"Scene: {self.episode.scene_id}"
        
        # Draw with shadow for visibility
        for text, y_pos in [(episode_text, 25), (scene_text, 50)]:
            cv2.putText(bgr_image, text, (10, y_pos), font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(bgr_image, text, (10, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return bgr_image
    
    def _get_topdown_map(self, agent_state) -> np.ndarray:
        """Get annotated top-down map.
        
        Args:
            agent_state: Current agent state.
            
        Returns:
            Annotated top-down map image (RGB format).
        """
        metrics = self.env.get_metrics()
        topdown_info = metrics.get("top_down_map", {})
        
        if not topdown_info or "map" not in topdown_info:
            # Fallback placeholder
            return np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        topdown_map = topdown_info["map"]
        
        # Calculate distance to current waypoint
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            current_distance = geodesic_distance(
                agent_state.position,
                self.waypoints[self.current_waypoint_idx]
            )
        else:
            current_distance = 0.0
        
        action = self.last_action if self.last_action else "MOVE_FORWARD"
        info = {"metrics": metrics}
        
        # Annotate top-down map
        topdown_image = annotate_topdown_map(
            info=info,
            agent_state=agent_state,
            waypoints=self.waypoints,
            current_waypoint_idx=self.current_waypoint_idx,
            step_count=self.step_count,
            action=action,
            current_distance=current_distance,
            goal_radius=self.goal_radius,
            config=self.config
        )
        
        return topdown_image if topdown_image is not None else topdown_map
    
    def _create_info_panel(self, panel_width: int, instruction_text: str, 
                          dist_to_next_waypoint: float, dist_to_goal: float) -> np.ndarray:
        """Create info panel with instruction and distances.
        
        Args:
            panel_width: Width of the panel in pixels.
            instruction_text: Instruction text to display.
            dist_to_next_waypoint: Distance to next waypoint in meters.
            dist_to_goal: Distance to goal in meters.
            
        Returns:
            Info panel image (BGR format).
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (0, 0, 0)
        shadow_color = (255, 255, 255)
        
        # Estimate character width and wrap instruction
        char_width = cv2.getTextSize("M", font, font_scale, thickness)[0][0]
        max_chars_per_line = int((panel_width - 40) / char_width)
        wrapped_instruction = textwrap.wrap(instruction_text, width=max_chars_per_line)
        
        # Calculate required height
        line_height = 26
        instruction_lines = len(wrapped_instruction)
        instruction_height = instruction_lines * line_height + 20
        distance_height = 2 * line_height + 10
        panel_height = instruction_height + distance_height + 20
        
        # Create panel
        info_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        y_start = 20
        
        # Display instruction
        instruction_y = y_start
        for i, line in enumerate(wrapped_instruction):
            prefix = "Instruction: " if i == 0 else "  "
            instruction_line = prefix + line
            cv2.putText(info_panel, instruction_line, (20, instruction_y), 
                       font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(info_panel, instruction_line, (20, instruction_y), 
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
            instruction_y += line_height
        
        # Display distances
        y_dist_start = instruction_y + 10
        distance_texts = [
            f"Distance to Next Waypoint: {dist_to_next_waypoint:.2f} m",
            f"Distance to Goal: {dist_to_goal:.2f} m"
        ]
        
        for i, text in enumerate(distance_texts):
            y_pos = y_dist_start + i * line_height
            cv2.putText(info_panel, text, (20, y_pos), 
                       font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(info_panel, text, (20, y_pos), 
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Add boundary warning if needed
        if self.at_boundary:
            warning_text = "WARNING: At safe boundary! Cannot move further."
            warning_color = (0, 0, 255)
            warning_y = y_dist_start - 5
            cv2.putText(info_panel, warning_text, (20, warning_y), 
                       font, font_scale * 0.8, shadow_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(info_panel, warning_text, (20, warning_y), 
                       font, font_scale * 0.8, warning_color, thickness, cv2.LINE_AA)
        
        return info_panel
    
    def _render_and_display(self):
        """Render current observation and display it."""
        try:
            # Get RGB observation
            rgb_image = self.obs["rgb"].copy()
            
            # Get agent state
            agent_state = self.env._task._sim.get_agent_state()
            
            # Calculate distances
            dist_to_next_waypoint = 0.0
            if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
                dist_to_next_waypoint = geodesic_distance(
                    agent_state.position,
                    self.waypoints[self.current_waypoint_idx]
                )
            
            dist_to_goal = 0.0
            if self.waypoints and len(self.waypoints) > 0:
                dist_to_goal = geodesic_distance(
                    agent_state.position,
                    self.waypoints[-1]
                )
            
            # Add episode info to RGB image
            rgb_bgr_with_info = self._add_episode_info_to_image(rgb_image)
            
            # Get top-down map if enabled
            if self.show_topdown:
                topdown_image = self._get_topdown_map(agent_state)
                
                # Resize and combine
                rgb_h, rgb_w = rgb_bgr_with_info.shape[:2]
                topdown_h, topdown_w = topdown_image.shape[:2]
                
                scale = rgb_h / topdown_h
                new_topdown_w = int(topdown_w * scale)
                topdown_resized = cv2.resize(
                    cv2.cvtColor(topdown_image, cv2.COLOR_RGB2BGR),
                    (new_topdown_w, rgb_h),
                    interpolation=cv2.INTER_CUBIC
                )
                
                top_row = np.concatenate([rgb_bgr_with_info, topdown_resized], axis=1)
            else:
                top_row = rgb_bgr_with_info
            
            # Create info panel
            instruction_text = self.obs["instruction"]["text"]
            panel_width = top_row.shape[1]
            info_panel = self._create_info_panel(
                panel_width, instruction_text, dist_to_next_waypoint, dist_to_goal
            )
            
            # Combine and display
            frame_bgr = np.concatenate([top_row, info_panel], axis=0)
            cv2.imshow(self.window_name, frame_bgr)
        
        except ValueError as e:
            error_msg = str(e)
            if "Camera view bounds exceed" in error_msg:
                print("Warning: Camera view exceeds map bounds. Try moving away from the edge.")
                self.at_boundary = True
            else:
                print(f"Warning: {e}")
            
            # Display error image
            if isinstance(self.config.SIMULATOR, dict):
                img_height = self.config.SIMULATOR.get("RGB_SENSOR", {}).get("HEIGHT", 480)
                img_width = self.config.SIMULATOR.get("RGB_SENSOR", {}).get("WIDTH", 640)
            else:
                img_height = getattr(self.config.SIMULATOR.RGB_SENSOR, "HEIGHT", 480)
                img_width = getattr(self.config.SIMULATOR.RGB_SENSOR, "WIDTH", 640)
            error_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            error_image_bgr = cv2.cvtColor(error_image, cv2.COLOR_RGB2BGR)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            warning_text = "Camera view exceeds map bounds!"
            instruction_text = "Use w/a/d to move"
            cv2.putText(error_image_bgr, warning_text, 
                       (50, img_height // 2 - 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(error_image_bgr, instruction_text, 
                       (50, img_height // 2 + 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow(self.window_name, error_image_bgr)
    
    def run(self):
        """Run the interactive viewer main loop."""
        print("\n" + "=" * 60)
        print("SatNav Task Viewer")
        print("=" * 60)
        print("Controls:")
        print("  'w' - Move forward")
        print("  'a' - Turn left")
        print("  'd' - Turn right")
        print("  't' - Toggle topdown view")
        print("  'SPACE' - Stop and show metrics")
        print("  'ESC' - Quit")
        print("=" * 60 + "\n")
        
        # Initial render
        self._render_and_display()
        
        # Event-driven loop
        while self.running:
            try:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ESC_KEY:
                    break
                elif key != 255:
                    should_continue = self.handle_keyboard(key)
                    if not should_continue:
                        break
                    self._render_and_display()
            
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        cv2.destroyAllWindows()
        print("\nViewer closed.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive task viewer for VLN tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).parent.parent
            / "resources"
            / "satnav_example_task.yaml"
        ),
        help="Path to task configuration YAML file"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    viewer = TaskViewer(str(config_path))
    viewer.run()


if __name__ == "__main__":
    main()

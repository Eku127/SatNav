# SatSim Viewer Applications

This directory contains two interactive viewer applications for exploring satellite maps:

1. **Free Viewer** (`free_viewer.py`) - Free exploration of satellite maps
2. **Task Viewer** (`task_viewer.py`) - Interactive VLN task viewer

## Quick Start

```bash
# Run free viewer (default)
python -m applications.satsim_viewer
python -m applications.satsim_viewer free

# Run task viewer
python -m applications.satsim_viewer task
```

## Free Viewer

Interactive viewer application for free exploration of satellite maps using SatSim.

### Features

- **Keyboard Navigation**: Navigate through satellite maps using simple keyboard controls
- **Real-time Visualization**: View RGB observations from the satellite map
- **State Display**: See current agent state in both WGS84 (longitude/latitude) and Mercator coordinates
- **Safe Boundary Detection**: Automatic detection and prevention of movements that would cause camera view to exceed map bounds
- **Boundary Warnings**: Visual and console warnings when agent reaches safe boundary limits
- **Configurable**: Customize camera parameters, initial position, and display settings via YAML config

### Requirements

- OpenCV (`opencv-python`) - for image display and keyboard input
- SatNav dependencies (rasterio, pyproj, scipy, numpy)

### Usage

**Basic Usage**:
```bash
# Using -m module syntax (default)
python -m applications.satsim_viewer
python -m applications.satsim_viewer free

# Or run directly
python -m applications.satsim_viewer.free_viewer
```

**Custom Configuration**:
```bash
# Using -m module syntax
python -m applications.satsim_viewer free --config /path/to/custom_config.yaml

# Or run directly
python -m applications.satsim_viewer.free_viewer --config /path/to/custom_config.yaml
```

### Configuration

Edit `config.yaml` to customize:

- **TIF_PATH**: Path to the satellite map TIF file
- **SIMULATOR**: Camera and movement parameters
  - `FORWARD_STEP_SIZE`: Step size for forward movement (meters)
  - `TURN_ANGLE`: Turn angle (degrees)
  - `RGB_SENSOR`: Camera parameters (width, height, HFOV)
- **AGENT**: Initial agent state
  - `ALTITUDE`: Initial altitude (meters)
  - `ROTATION`: Initial rotation/roll angle (degrees, 0=North)
  - `ALTITUDE_STEP_SIZE`: Step size for altitude changes (meters, for 'q' and 'e' keys)
- **DISPLAY**: Display settings
  - `WINDOW_NAME`: Window title
  - `FPS`: Update rate (frames per second)

### Controls

- **'w'**: Move forward
- **'s'**: Move backward
- **'a'**: Turn left
- **'d'**: Turn right
- **'q'**: Move up (increase altitude)
- **'e'**: Move down (decrease altitude)
- **'p'**: Save current image
- **ESC**: Quit

### Initial Position

The viewer automatically initializes the agent at the center of the loaded TIF file with:
- Position: Center of the map bounds (calculated from TIF bounds)
- Rotation: 0° (North)
- Altitude: As specified in config

**Note**: If the map is too small for the given altitude, the initial center position may not be within safe bounds. In this case, a warning will be displayed and you may need to reduce altitude (press 'e') to enable movement.

### Display Information

The viewer displays:
- **WGS84 Coordinates**: Longitude, latitude, altitude
- **Mercator Coordinates**: X, Y (in meters), altitude
- **Rotation**: Current roll angle (degrees)
- **Boundary Warnings**: Red warning text when agent reaches safe boundary limits

### Example

```bash
# Using default config (tests/test_data/map.tif)
python -m applications.satsim_viewer.free_viewer

# Using custom TIF file
# Edit config.yaml to set TIF_PATH to your file
python -m applications.satsim_viewer.free_viewer
```

### Safe Boundary Detection

The viewer uses **safe boundary detection** to prevent the camera view from exceeding map bounds:

- **Safe Boundary**: The navigable area is calculated by shrinking the map bounds by a safety margin
- **Safety Margin**: Depends on camera parameters (width, height, HFOV) and altitude
  - Higher altitude = larger camera view = larger safety margin
  - The margin ensures that regardless of rotation angle, the camera view will never exceed map bounds
- **Boundary Behavior**: When you try to move beyond the safe boundary:
  - Movement is blocked (agent stays at current position)
  - A warning message is displayed in the console
  - A red warning text appears on the image
  - You can still turn or change altitude to find a navigable path

### Notes

- The viewer uses OpenCV for display, which expects BGR color format
- Safe boundary detection prevents camera view from exceeding map bounds
- If the map is too small for the current altitude, you may need to reduce altitude to enable movement
- The viewer automatically calculates the center point from the TIF file bounds
- Boundary warnings appear both in the console and as overlay text on the image

---

## Task Viewer

Interactive viewer application for VLN tasks using SatNav. Allows users to navigate through VLN episodes with keyboard controls and view task metrics.

### Features

- **Task-based Navigation**: Navigate through VLN episodes from dataset
- **Keyboard Controls**: Simple keyboard controls for navigation (w/a/d for movement, SPACE for stop)
- **Visualization**: 
  - Left: Current RGB observation
  - Right: Top-down map with agent path and waypoints (toggleable)
  - Bottom: Instruction text and distance information
- **Metrics Display**: Show evaluation metrics after STOP action
- **Episode Management**: Automatically load next episode after completing current one

### Requirements

- OpenCV (`opencv-python`) - for image display and keyboard input
- SatNav dependencies

### Usage

**Basic Usage**:
```bash
# Using -m module syntax
python -m applications.satsim_viewer task

# Or run directly
python -m applications.satsim_viewer.task_viewer
```

**Custom Configuration**:
```bash
# Using -m module syntax
python -m applications.satsim_viewer task --config /path/to/vln_task.yaml

# Or run directly
python -m applications.satsim_viewer.task_viewer --config /path/to/vln_task.yaml
```

### Controls

- **'w'**: Move forward
- **'a'**: Turn left
- **'d'**: Turn right
- **'t'**: Toggle topdown view
- **'SPACE'**: Stop and show metrics, then load next episode
- **ESC**: Quit

### Display Information

The viewer displays:
- **Episode ID and Scene ID**: Shown in top-left corner of RGB image
- **Instruction**: Full instruction text in bottom panel
- **Distance to Next Waypoint**: Distance to current waypoint being navigated to
- **Distance to Goal**: Distance to final goal
- **Top-down Map**: Optional visualization showing agent path, waypoints, and camera view bounds

### Configuration

The Task Viewer uses the VLN task configuration file (e.g., `configs/vln_task.yaml`), which includes:
- **DATASET**: Dataset configuration (data path, scenes directory)
- **SIMULATOR**: Camera and movement parameters
- **TASK**: Task configuration (success distance, measurements, etc.)

The viewer automatically enables `TOP_DOWN_MAP` measurement if not already enabled.

---

## Input/Output Summary

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `TIF_PATH` | String | Path to input GeoTIFF file (EPSG:3857) |
| `SIMULATOR.FORWARD_STEP_SIZE` | Float | Step size for forward/backward movement (meters) |
| `SIMULATOR.TURN_ANGLE` | Float | Turn angle per keypress (degrees) |
| `SIMULATOR.RGB_SENSOR.WIDTH` | Integer | Camera image width (pixels) |
| `SIMULATOR.RGB_SENSOR.HEIGHT` | Integer | Camera image height (pixels) |
| `SIMULATOR.RGB_SENSOR.HFOV` | Float | Horizontal field of view (degrees) |
| `AGENT.ALTITUDE` | Float | Initial agent altitude (meters) |
| `AGENT.ROTATION` | Float | Initial rotation/roll angle (degrees, 0=North) |
| `AGENT.ALTITUDE_STEP_SIZE` | Float | Step size for altitude changes (meters) |
| `DISPLAY.WINDOW_NAME` | String | OpenCV window title |

### Output

| Output | Type | Description |
|--------|------|-------------|
| RGB Image | Array | Rendered satellite image (H×W×3, uint8) |
| WGS84 Coordinates | Dict | Agent position: `{longitude, latitude, altitude}` |
| Mercator Coordinates | Dict | Agent position: `{x, y, altitude}` (meters, EPSG:3857) |
| Rotation | Float | Current roll angle (degrees, 0=North) |
| Boundary Status | Boolean | Whether agent is at safe boundary limit |


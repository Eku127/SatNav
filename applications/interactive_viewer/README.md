# Interactive Viewer

Interactive viewer application for exploring satellite maps using SatSim.

## Features

- **Keyboard Navigation**: Navigate through satellite maps using simple keyboard controls
- **Real-time Visualization**: View RGB observations from the satellite map
- **State Display**: See current agent state in both WGS84 (longitude/latitude) and Mercator coordinates
- **Configurable**: Customize camera parameters, initial position, and display settings via YAML config

## Requirements

- OpenCV (`opencv-python`) - for image display and keyboard input
- SatNav dependencies (rasterio, pyproj, scipy, numpy)

## Usage

### Basic Usage

```bash
cd applications/interactive_viewer
python viewer.py
```

### Custom Configuration

```bash
python viewer.py --config /path/to/custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **TIF_PATH**: Path to the satellite map TIF file
- **SIMULATOR**: Camera and movement parameters
  - `FORWARD_STEP_SIZE`: Step size for forward movement (meters)
  - `TURN_ANGLE`: Turn angle (degrees)
  - `RGB_SENSOR`: Camera parameters (width, height, HFOV)
- **AGENT**: Initial agent state
  - `ALTITUDE`: Initial altitude (meters)
  - `ROTATION`: Initial rotation/roll angle (degrees, 0=North)
- **DISPLAY**: Display settings
  - `WINDOW_NAME`: Window title
  - `FPS`: Update rate (frames per second)

## Controls

- **'w'**: Move forward
- **'s'**: Move backward
- **'a'**: Turn left
- **'d'**: Turn right
- **'q'**: Move up (increase altitude)
- **'e'**: Move down (decrease altitude)
- **ESC**: Quit

## Initial Position

The viewer automatically initializes the agent at the center of the loaded TIF file with:
- Position: Center of the map bounds (calculated from TIF bounds)
- Rotation: 0° (North)
- Altitude: As specified in config

## Display Information

The viewer displays:
- **WGS84 Coordinates**: Longitude, latitude, altitude
- **Mercator Coordinates**: X, Y (in meters), altitude
- **Rotation**: Current roll angle (degrees)

## Example

```bash
# Using default config (tests/test_data/map.tif)
python viewer.py

# Using custom TIF file
# Edit config.yaml to set TIF_PATH to your file
python viewer.py
```

## Notes

- The viewer uses OpenCV for display, which expects BGR color format
- If camera view exceeds map bounds, a warning will be displayed
- The viewer automatically calculates the center point from the TIF file bounds


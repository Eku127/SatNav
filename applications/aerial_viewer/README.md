# Aerial Viewer

Non-interactive aerial 3D view renderer using CesiumJS and Google 3D Tiles.

## Features

- Render 3D aerial views from Google 3D Tiles
- **Sequence rendering** - Efficiently render multiple waypoints with browser reuse (~2x faster)
- **Automatic ground height query** - Automatically adjusts camera height based on terrain/building elevation
- Configurable camera position and orientation
- Support for multiple browsers (Edge, Chrome, Firefox)
- Headless mode support
- Command-line interface with config file support

## Example Output

The Aerial Viewer renders 3D aerial views compatible with SatSim's camera model:

<div style="display: flex; gap: 20px; align-items: center;">
  <div style="flex: 1;">
    <p><strong>SatSim Satellite View (2D orthographic)</strong></p>
    <img src="images/sat_crop_view.png" alt="SatSim Satellite View" style="width: 100%;">
  </div>
  <div style="flex: 1;">
    <p><strong>Aerial Viewer 3D View (3D perspective)</strong></p>
    <img src="images/aerial_view.png" alt="Aerial Viewer 3D View" style="width: 100%;">
  </div>
</div>

## Requirements

- Python 3.7+
- Selenium WebDriver
- Browser driver (Edge/Chrome/Firefox) - see installation instructions below
- Google 3D Tiles API key
- Browser (Chrome/Edge/Firefox) installed on the system

## Installation

### 1. Install Python Dependencies

```bash
pip install selenium omegaconf
```

**Optional (Recommended)**: Install `webdriver-manager` for automatic driver management:

```bash
pip install webdriver-manager
```

The `webdriver-manager` package automatically downloads and manages browser drivers, eliminating the need for manual driver installation.

### 2. Install Browser Driver

The application supports three browsers: **Chrome**, **Edge**, and **Firefox**. Choose one based on your system and preferences.

#### Option A: Automatic Driver Management (Recommended)

If you installed `webdriver-manager`, the application will automatically download and manage drivers. No manual installation needed!

#### Option B: Manual Driver Installation

If you prefer manual installation or `webdriver-manager` is not available:

##### Chrome / Chromium (Linux)

1. **Check Chrome version**:
   ```bash
   google-chrome --version
   # or
   chromium-browser --version
   ```

2. **Download ChromeDriver**:
   - Visit [ChromeDriver Downloads](https://chromedriver.chromium.org/downloads)
   - Download the version matching your Chrome version
   - Extract and place in a directory in your PATH (e.g., `/usr/local/bin/`)

3. **Make executable**:
   ```bash
   chmod +x /usr/local/bin/chromedriver
   ```

4. **Verify installation**:
   ```bash
   chromedriver --version
   ```

##### Chrome (Windows)

1. Download ChromeDriver from [ChromeDriver Downloads](https://chromedriver.chromium.org/downloads)
2. Extract `chromedriver.exe` to a directory in your PATH (e.g., `C:\Windows\System32\`)
3. Or place it in the same directory as your Python script

##### Chrome (macOS)

1. Install via Homebrew:
   ```bash
   brew install chromedriver
   ```

2. Or download manually from [ChromeDriver Downloads](https://chromedriver.chromium.org/downloads)

##### Edge (Linux)

Edge on Linux uses Chromium-based Edge, which requires EdgeDriver:

1. Download from [EdgeDriver Downloads](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)
2. Extract and place in PATH
3. Make executable: `chmod +x /usr/local/bin/msedgedriver`

##### Edge (Windows/macOS)

EdgeDriver usually comes with Edge browser. If not found:

1. Download from [EdgeDriver Downloads](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)
2. Place in PATH or same directory as script

##### Firefox (All Platforms)

1. Download GeckoDriver from [GeckoDriver Releases](https://github.com/mozilla/geckodriver/releases)
2. Extract and place in PATH
3. Make executable (Linux/macOS): `chmod +x /usr/local/bin/geckodriver`

## Usage

### Basic Usage

```bash
python -m applications.aerial_viewer
```

This will use the default configuration from `config.yaml` and save the output to `output/aerial_view.png`.

### With Custom Config

```bash
python -m applications.aerial_viewer --config /path/to/config.yaml
```

### SatSim-Compatible Mode

Use SatSim-compatible parameters (vertical down view, HFOV matching):

```bash
python -m applications.aerial_viewer \
    --latitude 40.7128 \
    --longitude -74.0060 \
    --altitude 100 \
    --rotation 0 \
    --hfov 90 \
    --output output/satsim_view.png
```

Legacy arguments are also supported for backward compatibility:

```bash
python -m applications.aerial_viewer \
    --lat 40.7128 \
    --lng -74.0060 \
    --height 100 \
    --heading 0 \
    --hfov 90
```

### Python API

```python
from applications.aerial_viewer import AerialRenderer

renderer = AerialRenderer("config.yaml")
try:
    # Use render_satsim_compatible for SatSim-compatible parameters
    output_path = renderer.render_satsim_compatible(
        longitude=-74.0060,
        latitude=40.7128,
        altitude=100,
        rotation=0,  # 0 = North
        hfov=90,     # Horizontal field of view
        width=640,   # Optional: output width
        height=480   # Optional: output height
    )
    print(f"Rendered to: {output_path}")
finally:
    renderer.close()
```

Or use the `render` method with legacy parameters:

```python
renderer = AerialRenderer("config.yaml")
try:
    output_path = renderer.render(
        lat=40.7128,
        lng=-74.0060,
        height=100,
        heading=0,  # Maps to rotation
        pitch=-90.0,  # Always vertical down in SatSim mode
        roll=0.0,
        hfov=90
    )
    print(f"Rendered to: {output_path}")
finally:
    renderer.close()
```

### Sequence Rendering (Optimized for Multiple Waypoints)

For rendering multiple waypoints efficiently, use the `render_sequence` method. It reuses the browser instance and can optionally reuse ground height queries for nearby points, achieving **~2x speedup** compared to individual rendering:

```python
from applications.aerial_viewer import AerialRenderer

renderer = AerialRenderer("config.yaml")
try:
    # Define waypoints as [longitude, latitude] or [longitude, latitude, altitude]
    waypoints = [
        [-121.999358, 37.510891],      # Uses default altitude
        [-121.998144, 37.510624],
        [-121.997000, 37.510400, 100], # Custom altitude: 100m
    ]
    
    # Optional: rotation angles for each waypoint (degrees, 0=North)
    rotations = [160.0, 160.0, 180.0]
    
    # Render sequence
    output_files = renderer.render_sequence(
        waypoints=waypoints,
        rotations=rotations,           # Optional, defaults to 0 for all
        altitude=50.0,                 # Default altitude if not in waypoint
        hfov=90.0,                     # Horizontal field of view
        output_dir="output/sequence",  # Output directory
        output_prefix="frame",         # Output filename prefix
        reuse_ground_height=True,      # Reuse ground height for nearby points
        ground_height_threshold=200.0, # Distance threshold (meters) for reusing
    )
    
    print(f"Rendered {len(output_files)} frames:")
    for f in output_files:
        print(f"  - {f}")
finally:
    renderer.close()
```

**Performance comparison:**

| Method | Time per frame | Notes |
|--------|---------------|-------|
| Individual rendering | ~29s/frame | New browser for each frame |
| Sequence rendering | ~15s/frame | Browser reuse, ~2x faster |

## Configuration

Edit `config.yaml` to set default parameters:

### API Configuration

```yaml
API:
  API_KEY: "YOUR_GOOGLE_API_KEY"  # Google 3D Tiles API key
```

### Agent & Camera Configuration (SatSim-Compatible)

```yaml
AGENT:
  LONGITUDE: -74.0060   # Longitude (degrees)
  LATITUDE: 40.7128     # Latitude (degrees)
  ALTITUDE: 100.0       # Altitude (meters)
  ROTATION: 0.0         # Rotation/roll angle (degrees, 0=North)

CAMERA:
  HFOV: 90             # Horizontal field of view (degrees)
  WIDTH: 640           # Output width (pixels)
  HEIGHT: 480          # Output height (pixels)
```

**Note**: The camera always renders vertical down view (pitch = -90°), matching SatSim's orthographic projection behavior.

### Browser Configuration

```yaml
BROWSER:
  DRIVER: "chrome"    # "edge", "chrome", or "firefox"
  HEADLESS: false     # WebGL requires non-headless mode for CesiumJS 3D rendering
```

### Rendering Configuration

```yaml
RENDERING:
  MAX_WAIT_TIME: 60.0   # Maximum wait time for 3D Tiles to load (seconds)
  CHECK_INTERVAL: 0.5   # Interval between status checks (seconds)
  STABILITY_CHECKS: 3    # Number of consecutive stable checks before proceeding
  OUTPUT_PATH: "output/aerial_view.png"  # Output image path
```

## Output

The renderer outputs a PNG image with the specified camera view. The image size matches the camera configuration (`CAMERA.WIDTH` × `CAMERA.HEIGHT`).

### Comparison with SatSim

The Aerial Viewer provides a 3D perspective view that matches SatSim's camera parameters:

- **SatSim**: Uses 2D orthographic projection from satellite imagery
- **Aerial Viewer**: Uses 3D perspective projection from Google 3D Tiles

Both views use the same camera parameters (position, altitude, HFOV, rotation) for compatibility.

## Troubleshooting

### Common Issues and Solutions

#### 1. WebDriver Connection Error

**Error**: `selenium.common.exceptions.WebDriverException: Message: Connection error`

**Causes & Solutions**:

- **System Proxy Interference**: System-wide proxy settings (`http_proxy`, `https_proxy`) can interfere with Selenium's local connection to browser drivers.

  **Solution**: The application automatically disables proxy for local connections. If you still encounter issues:
  ```bash
  # Temporarily unset proxy for the session
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  python -m applications.aerial_viewer
  ```

- **Driver Not Found**: Browser driver is not in PATH or not installed.

  **Solution**: 
  - Install `webdriver-manager`: `pip install webdriver-manager` (recommended)
  - Or manually install driver (see Installation section above)
  - Verify driver is in PATH: `which chromedriver` (Linux/macOS) or `where chromedriver` (Windows)

- **Driver Version Mismatch**: Driver version doesn't match browser version.

  **Solution**: 
  - Use `webdriver-manager` to automatically match versions
  - Or manually download matching driver version

#### 2. WebGL Initialization Failed

**Error**: `Error constructing CesiumWidget. The browser supports WebGL, but initialization failed.`

**Cause**: CesiumJS requires WebGL for 3D rendering. Headless Chrome has limited WebGL support.

**Solutions**:

1. **Use Non-Headless Mode** (Recommended):
   ```yaml
   BROWSER:
     HEADLESS: false
   ```
   - Requires a display (X11 on Linux, GUI on Windows/macOS)
   - For Linux servers without display, use Xvfb (see below)

2. **Use Xvfb (Linux servers without display)**:
   ```bash
   # Install Xvfb
   sudo apt-get install xvfb  # Ubuntu/Debian
   
   # Run with Xvfb
   xvfb-run -a python -m applications.aerial_viewer
   ```

3. **Try Different Browser**: Firefox or Edge may have better headless WebGL support

#### 3. Tiles Not Loading / Always 0 Tiles

**Symptom**: Status shows `tiles: 0` even after waiting

**Possible Causes**:

- **Network Issues**: Cannot connect to Google 3D Tiles API
- **API Key Issues**: Invalid or missing API key, or API not enabled
- **Proxy Blocking**: Corporate proxy blocking Google services

**Solutions**:

1. **Check API Key**:
   ```yaml
   API:
     API_KEY: "YOUR_VALID_API_KEY"
   ```
   - Ensure API key is valid
   - Enable "3D Tiles API" in Google Cloud Console

2. **Check Network Connection**:
   ```bash
   # Test connectivity to Google 3D Tiles
   curl "https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_API_KEY"
   ```

3. **Increase Wait Time**:
   ```yaml
   RENDERING:
     MAX_WAIT_TIME: 120.0  # Increase from default 60.0
   ```

4. **Check Browser Console**: The application logs browser console errors. Check output for JavaScript errors.

#### 4. DISPLAY Not Set (Linux)

**Error**: `Warning: DISPLAY not set, browser may not work properly`

**Solution**:

- **With Display**: Set `DISPLAY` environment variable:
  ```bash
  export DISPLAY=:0
  python -m applications.aerial_viewer
  ```

- **Without Display (Headless Server)**:
  - Use `HEADLESS: true` in config (may have WebGL issues)
  - Or use Xvfb: `xvfb-run -a python -m applications.aerial_viewer`

#### 5. Chrome Binary Not Found

**Error**: Chrome/Chromium executable not found

**Solution**:

- **Linux**: Install Chrome or Chromium:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install google-chrome-stable
  # or
  sudo apt-get install chromium-browser
  ```

- **macOS**: Install via Homebrew:
  ```bash
  brew install --cask google-chrome
  ```

- **Windows**: Chrome should be installed in default location

#### 6. Slow Rendering / Timeout

**Symptom**: Rendering takes too long or times out

**Solutions**:

1. **Increase Wait Time**:
   ```yaml
   RENDERING:
     MAX_WAIT_TIME: 120.0  # Increase wait time
     CHECK_INTERVAL: 1.0    # Check less frequently
   ```

2. **Check Network Speed**: 3D Tiles require good network connection

3. **Reduce Image Size** (faster rendering):
   ```yaml
   CAMERA:
     WIDTH: 640   # Smaller = faster
     HEIGHT: 480
   ```

#### 7. Permission Denied (Linux)

**Error**: `Permission denied` when running driver

**Solution**:
```bash
chmod +x /usr/local/bin/chromedriver
# or
chmod +x /usr/local/bin/geckodriver
```

### Debug Mode

To get more detailed error information:

1. **Check Browser Console Logs**: The application prints browser console errors
2. **Check Driver Logs**: ChromeDriver logs are saved to `/tmp/chromedriver.log`
3. **Run with Verbose Output**: Check Python output for detailed error messages

### System-Specific Notes

#### Linux

- **Recommended**: Use Chrome with `webdriver-manager` or manual ChromeDriver installation
- **Headless Servers**: Use Xvfb for non-headless mode, or accept limited WebGL in headless mode
- **Display**: Set `DISPLAY=:0` or use Xvfb

#### Windows

- **Recommended**: Use Chrome or Edge
- **Driver Path**: Drivers can be in PATH or same directory as script
- **No Display Issues**: GUI mode works natively

#### macOS

- **Recommended**: Use Chrome (via Homebrew) or Safari (if supported)
- **Driver**: Use `webdriver-manager` or Homebrew (`brew install chromedriver`)

## SatSim Compatibility

The application supports **SatSim-compatible mode** for matching satellite viewer camera settings:

### Parameter Mapping

| SatSim Parameter | Aerial Viewer | Notes |
|-----------------|---------------|-------|
| `longitude` | `AGENT.LONGITUDE` | Direct mapping |
| `latitude` | `AGENT.LATITUDE` | Direct mapping |
| `altitude` | `AGENT.ALTITUDE` | Direct mapping |
| `rotation` | `AGENT.ROTATION` → `HEADING` | Rotation becomes heading (0=North) |
| `HFOV` | `CAMERA.HFOV` | Horizontal field of view (converted to vertical FOV for CesiumJS) |
| `WIDTH` | `CAMERA.WIDTH` | Output image width |
| `HEIGHT` | `CAMERA.HEIGHT` | Output image height |
| (implicit) | `PITCH = -90°` | Always vertical down view |
| (implicit) | `ROLL = 0°` | No roll |

### Differences from SatSim

1. **Projection Model**:
   - **SatSim**: Orthographic projection (no perspective distortion)
   - **Aerial Viewer**: Perspective projection (3D rendering with perspective)
   - **Impact**: At low altitudes, perspective distortion may be visible. At high altitudes (>500m), the difference is minimal.

2. **3D Terrain**:
   - **SatSim**: 2D satellite imagery (flat)
   - **Aerial Viewer**: 3D Tiles with building heights and terrain
   - **Impact**: Buildings and terrain features appear in 3D, matching real-world appearance.

3. **Image Source**:
   - **SatSim**: High-resolution GeoTIFF files
   - **Aerial Viewer**: Google 3D Tiles (streamed, resolution depends on zoom level)
   - **Impact**: Image quality and detail level may differ.

### Usage Example: Matching SatSim View

```python
# SatSim configuration
satnav_config = {
    "SIMULATOR": {
        "RGB_SENSOR": {
            "WIDTH": 640,
            "HEIGHT": 480,
            "HFOV": 90
        }
    }
}

# Equivalent Aerial Viewer configuration
aerial_config = {
    "AGENT": {
        "LONGITUDE": -74.0060,
        "LATITUDE": 40.7128,
        "ALTITUDE": 100.0,
        "ROTATION": 0.0
    },
    "CAMERA": {
        "HFOV": 90,
        "WIDTH": 640,
        "HEIGHT": 480
    }
}
```

## Ground Height Auto-Query

The renderer automatically queries the ground/building height at the camera position to ensure proper altitude above terrain. This prevents the camera from being inside buildings or underground.

### How it works

1. **Grid sampling**: Queries a grid of points around the camera position (default: 7x7 grid over 2× altitude area)
2. **Maximum height**: Uses the maximum height found in the grid (handles buildings, terrain variations)
3. **Safety margin**: Adds a configurable safety margin (default: 2m) to avoid clipping
4. **Ellipsoid height**: Converts the requested altitude above ground to absolute ellipsoid height

### Example

```
User requests: altitude = 50m above ground
Ground height query: max height = -20.93m (below ellipsoid)
Safety margin: 2m
Final camera height: -20.93 + 2 + 50 = 31.07m (ellipsoid height)
```

### Configuration

Ground height query behavior can be tuned in `render_sequence`:

- `reuse_ground_height`: Reuse previous ground height for nearby points (default: `True`)
- `ground_height_threshold`: Distance threshold for reusing (default: `200m`)

## Notes

- The renderer uses CesiumJS to load Google 3D Tiles
- Rendering time depends on network speed and 3D Tiles loading
- The application uses **adaptive waiting** - it automatically detects when 3D Tiles are fully loaded
- **WebGL is required** for 3D rendering - headless mode may have limitations
- Make sure your Google API key has **3D Tiles API enabled** in Google Cloud Console
- The application automatically disables system proxy for local Selenium connections
- For best results, use **non-headless mode** with a display (or Xvfb on Linux servers)
- The camera automatically sets `PITCH=-90°` and `ROLL=0°` for vertical down view (SatSim-compatible)
- **FOV conversion**: Horizontal FOV (HFOV) is automatically converted to vertical FOV for CesiumJS based on aspect ratio
- **Ground height**: Camera altitude is automatically adjusted based on terrain/building height to ensure correct height above ground


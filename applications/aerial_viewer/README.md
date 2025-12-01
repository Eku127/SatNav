# Aerial Viewer

Non-interactive aerial 3D view renderer using CesiumJS and Google 3D Tiles.

## Features

- Render 3D aerial views from Google 3D Tiles
- Configurable camera position and orientation
- Support for multiple browsers (Edge, Chrome, Firefox)
- Headless mode support
- Command-line interface with config file support

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

### Override Parameters via Command Line

```bash
python -m applications.aerial_viewer \
    --lat 22.2854 \
    --lng 114.1570 \
    --height 500 \
    --pitch -30 \
    --output output/my_view.png
```

## Configuration

Edit `config.yaml` to set default parameters:

### API Configuration

```yaml
API:
  API_KEY: "YOUR_GOOGLE_API_KEY"  # Google 3D Tiles API key
```

### Camera Configuration

```yaml
CAMERA:
  LAT: 22.2854      # Latitude (degrees)
  LNG: 114.1570     # Longitude (degrees)
  HEIGHT: 500.0     # Height above ground (meters)
  HEADING: 0.0      # Heading angle (degrees, 0=North)
  PITCH: -30.0      # Pitch angle (degrees, negative = looking down)
  ROLL: 0.0         # Roll angle (degrees)
```

### Browser Configuration

```yaml
BROWSER:
  DRIVER: "edge"    # "edge", "chrome", or "firefox"
  HEADLESS: true    # Run in headless mode
  WINDOW_SIZE: [1920, 1080]  # Browser window size [width, height]
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

The renderer outputs a PNG image with the specified camera view. The image size matches the browser window size (default: 1920×1080 pixels).

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

3. **Reduce Window Size** (faster rendering):
   ```yaml
   BROWSER:
     WINDOW_SIZE: [1280, 720]  # Smaller = faster
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

## Notes

- The renderer uses CesiumJS to load Google 3D Tiles
- Rendering time depends on network speed and 3D Tiles loading
- The application uses **adaptive waiting** - it automatically detects when 3D Tiles are fully loaded
- **WebGL is required** for 3D rendering - headless mode may have limitations
- Make sure your Google API key has **3D Tiles API enabled** in Google Cloud Console
- The application automatically disables system proxy for local Selenium connections
- For best results, use **non-headless mode** with a display (or Xvfb on Linux servers)


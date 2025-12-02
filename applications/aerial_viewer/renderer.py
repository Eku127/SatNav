#!/usr/bin/env python3
"""Aerial view renderer using CesiumJS and Google 3D Tiles."""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

# Disable proxy for Selenium connections to local ChromeDriver/GeckoDriver
# This is necessary because system proxy settings can interfere with local connections
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

from omegaconf import OmegaConf
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.edge.service import Service as EdgeService

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False


class AerialRenderer:
    """Non-interactive aerial view renderer."""
    
    def __init__(self, config_path: str):
        """Initialize the renderer.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = OmegaConf.load(config_path)
        self.driver = None
        self.html_path = None
        
    def _create_driver(self):
        """Create and configure Selenium WebDriver."""
        browser = self.config.BROWSER.DRIVER.lower()
        headless = self.config.BROWSER.HEADLESS
        
        # Use camera dimensions (required)
        if not hasattr(self.config, "CAMERA"):
            raise ValueError("CAMERA configuration is required. Please add CAMERA section to config.yaml")
        window_size = [
            self.config.CAMERA.WIDTH,
            self.config.CAMERA.HEIGHT
        ]
        
        print(f"Initializing {browser} browser driver...")
        if not headless:
            display = os.environ.get("DISPLAY")
            if display:
                print(f"  DISPLAY={display}")
            else:
                print("  Warning: DISPLAY not set, browser may not work properly")
        
        try:
            if browser == "edge":
                options = EdgeOptions()
                if headless:
                    options.add_argument("--headless")
                options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
                if WEBDRIVER_MANAGER_AVAILABLE:
                    try:
                        service = EdgeService(EdgeChromiumDriverManager().install())
                        self.driver = webdriver.Edge(service=service, options=options)
                    except Exception as e:
                        print(f"  Warning: webdriver-manager failed, using system driver...")
                        self.driver = webdriver.Edge(options=options)
                else:
                    self.driver = webdriver.Edge(options=options)
                    
            elif browser == "chrome":
                options = ChromeOptions()
                if headless:
                    # Use old headless mode which has better WebGL support
                    options.add_argument("--headless")
                options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
                # Add additional options for stability
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                # WebGL configuration for headless mode
                options.add_argument("--enable-webgl")
                options.add_argument("--use-gl=angle")  # Use ANGLE for WebGL
                options.add_argument("--use-angle=swiftshader-webgl")  # SwiftShader for software WebGL
                options.add_argument("--ignore-gpu-blocklist")
                options.add_argument("--enable-features=VaapiVideoDecoder")
                # Do NOT disable GPU - needed for WebGL
                # options.add_argument("--disable-gpu")
                options.add_argument("--disable-extensions")
                options.add_argument("--disable-logging")
                options.add_argument("--log-level=3")
                # Use random port to avoid conflicts with existing Chrome instances
                import random
                debug_port = random.randint(9000, 9999)
                options.add_argument(f"--remote-debugging-port={debug_port}")
                # Additional options for better compatibility
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                # Try to set Chrome binary location if available
                chrome_binary_paths = ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/opt/google/chrome/chrome"]
                for chrome_path in chrome_binary_paths:
                    if Path(chrome_path).exists():
                        options.binary_location = chrome_path
                        print(f"  Using Chrome binary at: {chrome_path}")
                        break
                
                # Try to find chromedriver
                chromedriver_path = None
                for path in ["/usr/bin/chromedriver", "/usr/local/bin/chromedriver", shutil.which("chromedriver")]:
                    if path and (shutil.which(path) or Path(path).exists()):
                        chromedriver_path = path
                        print(f"  Found ChromeDriver at: {chromedriver_path}")
                        break
                
                if chromedriver_path:
                    # Use explicit chromedriver path
                    try:
                        service = ChromeService(executable_path=chromedriver_path, log_path="/tmp/chromedriver.log")
                        print("  Attempting to start Chrome with explicit ChromeDriver...")
                        self.driver = webdriver.Chrome(service=service, options=options)
                    except Exception as e:
                        print(f"  Warning: Headless mode failed ({type(e).__name__}): {e}")
                        # Try without headless mode if headless was enabled and DISPLAY is available
                        if headless and os.getenv("DISPLAY"):
                            print("  DISPLAY is available, trying without headless mode...")
                            options_no_headless = ChromeOptions()
                            options_no_headless.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
                            options_no_headless.add_argument("--no-sandbox")
                            options_no_headless.add_argument("--disable-dev-shm-usage")
                            if options.binary_location:
                                options_no_headless.binary_location = options.binary_location
                            try:
                                service = ChromeService(executable_path=chromedriver_path)
                                self.driver = webdriver.Chrome(service=service, options=options_no_headless)
                                print("  ✓ Success! Running in non-headless mode (GUI will be visible)")
                            except Exception as e2:
                                print(f"  Non-headless also failed: {type(e2).__name__}: {e2}")
                                raise Exception(f"All Chrome driver attempts failed. Last error: {e2}")
                        else:
                            print("  Trying without explicit service...")
                            try:
                                self.driver = webdriver.Chrome(options=options)
                            except Exception as e2:
                                print(f"  System driver also failed: {type(e2).__name__}: {e2}")
                                raise Exception(f"All Chrome driver attempts failed. Last error: {e2}")
                elif WEBDRIVER_MANAGER_AVAILABLE:
                    print("  Attempting to use ChromeDriver via webdriver-manager...")
                    try:
                        service = ChromeService(ChromeDriverManager().install())
                        self.driver = webdriver.Chrome(service=service, options=options)
                    except Exception as e:
                        print(f"  Warning: webdriver-manager failed ({type(e).__name__}), trying system driver...")
                        try:
                            self.driver = webdriver.Chrome(options=options)
                        except Exception as e2:
                            raise Exception(f"All Chrome driver attempts failed. Last error: {e2}")
                else:
                    self.driver = webdriver.Chrome(options=options)
                    
            elif browser == "firefox":
                options = FirefoxOptions()
                if headless:
                    options.add_argument("--headless")
                # Try to find geckodriver in common locations
                geckodriver_path = None
                for path in ["/snap/bin/geckodriver", "/usr/local/bin/geckodriver", "/usr/bin/geckodriver"]:
                    if shutil.which(path) or Path(path).exists():
                        geckodriver_path = path
                        break
                
                # Skip webdriver-manager if GitHub API is rate-limited, go straight to system driver
                if geckodriver_path:
                    print(f"  Using system GeckoDriver at: {geckodriver_path}")
                    try:
                        service = FirefoxService(executable_path=geckodriver_path)
                        self.driver = webdriver.Firefox(service=service, options=options)
                    except Exception as e:
                        print(f"  Warning: Explicit geckodriver path failed ({type(e).__name__}): {e}")
                        print("  Trying without explicit service...")
                        self.driver = webdriver.Firefox(options=options)
                elif WEBDRIVER_MANAGER_AVAILABLE:
                    print("  Attempting to use GeckoDriver via webdriver-manager...")
                    try:
                        service = FirefoxService(GeckoDriverManager().install())
                        self.driver = webdriver.Firefox(service=service, options=options)
                    except Exception as e:
                        print(f"  Warning: webdriver-manager failed ({type(e).__name__})")
                        print("  Trying default driver...")
                        self.driver = webdriver.Firefox(options=options)
                else:
                    self.driver = webdriver.Firefox(options=options)
                self.driver.set_window_size(window_size[0], window_size[1])
            else:
                raise ValueError(f"Unsupported browser: {browser}")
            
            print(f"✓ Browser driver initialized successfully")
            
        except Exception as e:
            print(f"✗ Failed to initialize browser driver: {e}")
            print(f"  Error type: {type(e).__name__}")
            print("\n  Troubleshooting:")
            print("    1. Check if browser is installed:")
            print("       - Firefox: which firefox")
            print("       - Chrome: which google-chrome")
            print("    2. Check if browser driver is in PATH:")
            print("       - GeckoDriver: which geckodriver")
            print("       - ChromeDriver: which chromedriver")
            print("    3. Install browser drivers:")
            print("       - Firefox: sudo apt install firefox-geckodriver")
            print("       - Chrome: Download ChromeDriver from https://chromedriver.chromium.org/")
            print("    4. If using headless mode, ensure browser supports it")
            print("    5. Check browser and driver version compatibility")
            print("    6. Try running without headless mode (HEADLESS: false) if you have a display")
            print("\n  Note: 'Connection error' usually means the browser driver cannot start the browser.")
            print("        This may be due to version mismatch, missing dependencies, or permissions.")
            raise
    
    def _prepare_html(self) -> Path:
        """Prepare HTML template with API key injected.
        
        Returns:
            Path to the prepared HTML file.
        """
        # Get template path
        template_dir = Path(__file__).parent
        template_path = template_dir / "template.html"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Read template
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace API key placeholder
        api_key = self.config.API.API_KEY
        html_content = html_content.replace("{{API_KEY}}", api_key)
        
        # Write to temporary file
        temp_html_path = template_dir / "temp_aerial_viewer.html"
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return temp_html_path
    
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
        """Render aerial view and save as image (SatSim-compatible mode).
        
        Args:
            lat: Latitude (degrees). If None, uses config value.
            lng: Longitude (degrees). If None, uses config value.
            height: Height above ground (meters). If None, uses config value.
            heading: Heading angle (degrees). If None, uses config value.
            pitch: Pitch angle (degrees). If None, uses -90.0 (vertical down).
            roll: Roll angle (degrees). If None, uses 0.0.
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
        pitch = pitch if pitch is not None else -90.0  # Always vertical down view
        roll = roll if roll is not None else 0.0
        hfov = hfov if hfov is not None else camera.HFOV
        
        output_path = output_path or self.config.RENDERING.OUTPUT_PATH
        
        print("Starting aerial view rendering (SatSim-compatible mode)...")
        print(f"  Camera position: lat={lat:.6f}, lng={lng:.6f}, height={height:.1f}m")
        print(f"  Camera orientation: heading={heading:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
        print(f"  Horizontal FOV: {hfov:.1f}°")
        
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Output path: {output_file}")
        
        # Create driver if not exists
        if self.driver is None:
            self._create_driver()
        
        # Prepare HTML
        print("Preparing HTML template...")
        self.html_path = self._prepare_html()
        print(f"✓ HTML template prepared: {self.html_path}")
        
        # Load HTML
        print("Loading HTML in browser...")
        file_url = f"file://{self.html_path.absolute()}"
        try:
            self.driver.set_page_load_timeout(30)  # 30 second timeout
            self.driver.get(file_url)
            print("✓ HTML loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load HTML: {e}")
            raise
        
        # Wait for initial load and check if page loaded correctly
        print("Waiting for initial load (5 seconds)...")
        time.sleep(5)
        
        # Check if CesiumJS and updateView function are available
        print("Checking if CesiumJS loaded...")
        try:
            cesium_loaded = self.driver.execute_script("return typeof Cesium !== 'undefined';")
            if not cesium_loaded:
                print("✗ Warning: CesiumJS not loaded! Network/proxy issue?")
            else:
                print("✓ CesiumJS loaded")
            
            # Check browser console for errors
            console_logs = self.driver.get_log('browser')
            if console_logs:
                print("Browser console logs:")
                for log in console_logs[-10:]:  # Last 10 logs
                    level = log.get('level', 'UNKNOWN')
                    message = log.get('message', '')
                    print(f"  [{level}] {message[:200]}")
            
            # Check for updateView or updateViewWithFOV
            updateview_available = self.driver.execute_script(
                "return typeof window.updateView === 'function';"
            )
            updateview_fov_available = self.driver.execute_script(
                "return typeof window.updateViewWithFOV === 'function';"
            )
            
            if not updateview_available and not updateview_fov_available:
                print("✗ Warning: updateView functions not available")
                # Check if there's a JavaScript error
                js_errors = self.driver.execute_script("""
                    var errors = [];
                    if (window.__jsErrors) errors = window.__jsErrors;
                    return errors;
                """)
                if js_errors:
                    print(f"  JavaScript errors: {js_errors}")
                # Wait more and retry
                print("Waiting additional 10 seconds...")
                time.sleep(10)
                updateview_available = self.driver.execute_script(
                    "return typeof window.updateView === 'function';"
                )
                updateview_fov_available = self.driver.execute_script(
                    "return typeof window.updateViewWithFOV === 'function';"
                )
                if not updateview_available and not updateview_fov_available:
                    # Get page source for debugging
                    print("Page source (first 500 chars):")
                    print(self.driver.page_source[:500])
                    raise Exception("updateView functions not available after waiting")
        except Exception as e:
            print(f"✗ JavaScript check failed: {e}")
            # Take screenshot of error state
            try:
                error_screenshot = str(output_file).replace('.png', '_error.png')
                self.driver.save_screenshot(error_screenshot)
                print(f"  Error screenshot saved to: {error_screenshot}")
            except:
                pass
            raise
        
        # Set container size before updating camera view
        if hasattr(self.config, "CAMERA"):
            target_width = self.config.CAMERA.WIDTH
            target_height = self.config.CAMERA.HEIGHT
            # Set container size early so CesiumJS can adjust
            self.driver.execute_script(f"""
                var container = document.getElementById('cesiumContainer');
                if (container) {{
                    container.style.width = '{target_width}px';
                    container.style.height = '{target_height}px';
                }}
            """)
            time.sleep(0.3)  # Wait for container resize
        
        # Update camera view with FOV
        print(f"Setting camera view...")
        if hfov is not None and updateview_fov_available:
            # Use FOV-aware update function
            self.driver.execute_script(
                f"window.updateViewWithFOV({lat}, {lng}, {height}, {heading}, {pitch}, {roll}, {hfov});"
            )
            print(f"✓ Camera view updated with HFOV={hfov}°")
        else:
            # Fallback to standard update function if FOV function not available
            self.driver.execute_script(
                f"window.updateView({lat}, {lng}, {height}, {heading}, {pitch}, {roll});"
            )
            print("✓ Camera view updated (without FOV)")
        
        # Ensure CesiumJS resizes after camera update
        self.driver.execute_script("""
            if (typeof viewer !== 'undefined') {
                viewer.resize();
            }
        """)
        time.sleep(0.5)  # Wait for resize to complete
        
        # Adaptive waiting for 3D Tiles to load
        print("Waiting for 3D Tiles to load and render...")
        max_wait_time = self.config.RENDERING.get("MAX_WAIT_TIME", 60.0)  # Maximum wait time in seconds
        check_interval = self.config.RENDERING.get("CHECK_INTERVAL", 0.5)  # Check every 0.5 seconds
        stability_checks = self.config.RENDERING.get("STABILITY_CHECKS", 3)  # Number of consecutive stable checks
        
        start_time = time.time()
        stable_count = 0
        last_status = None
        debug_done = False
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            try:
                status = self.driver.execute_script("""
                    if (typeof window.checkTilesetStatus === 'function') {
                        return window.checkTilesetStatus();
                    } else {
                        return { ready: false, progress: 0, stable: false, tilesLoaded: 0 };
                    }
                """)
                
                # Debug: If tiles count is 0 but tileset is ready, try alternative methods (only once)
                if status and status.get('tilesLoaded', -1) == 0 and status.get('ready', False) and not debug_done and elapsed < 10:
                    # Try to get tiles count using direct statistics access
                    try:
                        debug_info = self.driver.execute_script("""
                            try {
                                if (typeof tileset !== 'undefined') {
                                    const stats = tileset.statistics || {};
                                    return {
                                        hasStatistics: !!tileset.statistics,
                                        numberOfTilesWithContentReady: stats.numberOfTilesWithContentReady || 0,
                                        numberOfTilesLoading: stats.numberOfTilesLoading || 0,
                                        numberOfTilesTotal: stats.numberOfTilesTotal || 0,
                                        hasRoot: !!tileset.root,
                                        statisticsKeys: tileset.statistics ? Object.keys(tileset.statistics) : []
                                    };
                                }
                            } catch(e) {
                                return { error: e.toString() };
                            }
                            return { error: 'tileset not defined' };
                        """)
                        if debug_info and not debug_info.get('error'):
                            tiles_count = debug_info.get('numberOfTilesWithContentReady', 0)
                            if tiles_count > 0:
                                # Update status with correct value
                                status['tilesLoaded'] = tiles_count
                                print(f"  [DEBUG] Found {tiles_count} tiles via statistics.numberOfTilesWithContentReady")
                            elif debug_info.get('numberOfTilesTotal', 0) > 0:
                                # Use total as fallback
                                status['tilesLoaded'] = debug_info['numberOfTilesTotal']
                                print(f"  [DEBUG] Using total tiles count: {debug_info['numberOfTilesTotal']}")
                            else:
                                print(f"  [DEBUG] Statistics available but all counts are 0.")
                                print(f"         hasStatistics: {debug_info.get('hasStatistics')}, hasRoot: {debug_info.get('hasRoot')}")
                                print(f"         statistics keys: {debug_info.get('statisticsKeys', [])}")
                        debug_done = True  # Only debug once
                    except Exception as debug_e:
                        print(f"  [DEBUG] Error getting tiles count: {debug_e}")
                        debug_done = True
                
                if status:
                    ready = status.get('ready', False)
                    progress = status.get('progress', 0)
                    stable = status.get('stable', False)
                    tiles_loaded = status.get('tilesLoaded', 0)
                    should_print = False
                    if last_status is None:
                        should_print = True
                    elif last_status.get('tilesLoaded', 0) != tiles_loaded:
                        # Always print when tile count changes
                        should_print = True
                    elif int(elapsed * 2) != int((time.time() - start_time - check_interval) * 2):
                        # Print every 2 seconds
                        should_print = True
                    
                    if should_print:
                        if ready:
                            stable_str = "stable" if stable else "loading"
                            print(f"  [{elapsed:.1f}s] Tileset ready, tiles: {tiles_loaded}, status: {stable_str}")
                        else:
                            print(f"  [{elapsed:.1f}s] Loading tileset... tiles: {tiles_loaded}")
                        last_status = dict(status) if isinstance(status, dict) else status
                    
                    # Check if tileset is ready and rendering is stable
                    if ready and stable:
                        stable_count += 1
                        if stable_count >= stability_checks:
                            print(f"✓ 3D Tiles loaded and rendering stable after {elapsed:.1f} seconds")
                            break
                    else:
                        stable_count = 0  # Reset counter if not stable
                
            except Exception as e:
                # If status check fails, continue waiting
                pass
            
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        if elapsed >= max_wait_time:
            print(f"⚠ Maximum wait time ({max_wait_time}s) reached, proceeding with screenshot...")
        else:
            # Give a small buffer for final rendering
            time.sleep(0.5)
        
        # Take screenshot
        print("Taking screenshot...")
        try:
            # Get target dimensions from camera config
            target_width = self.config.CAMERA.WIDTH
            target_height = self.config.CAMERA.HEIGHT
            
            # Strategy: Render at larger size, then crop to target size
            # This ensures we have enough content and can crop center region to avoid black borders
            # Use 1.2x multiplier for render size (20% larger)
            render_width = int(target_width * 1.2)
            render_height = int(target_height * 1.2)
            
            print(f"  Target size: {target_width}x{target_height}")
            print(f"  Render size: {render_width}x{render_height} (1.2x for cropping)")
            
            # Set container to render size
            self.driver.execute_script(f"""
                var container = document.getElementById('cesiumContainer');
                if (container) {{
                    container.style.width = '{render_width}px';
                    container.style.height = '{render_height}px';
                }}
                if (typeof viewer !== 'undefined') {{
                    viewer.resize();
                }}
            """)
            
            # Wait for resize and rendering
            time.sleep(1.0)
            
            # Update browser window size to match render size (with some padding for browser chrome)
            self.driver.set_window_size(render_width + 50, render_height + 100)
            time.sleep(0.5)
            
            # Take screenshot of the container element
            canvas = self.driver.find_element("id", "cesiumContainer")
            temp_screenshot = str(output_file) + ".temp.png"
            canvas.screenshot(temp_screenshot)
            
            # Crop center region to target size
            from PIL import Image
            img = Image.open(temp_screenshot)
            actual_width, actual_height = img.size
            print(f"  Screenshot size: {actual_width}x{actual_height}")
            
            # Calculate crop box (center crop)
            crop_left = (actual_width - target_width) // 2
            crop_top = (actual_height - target_height) // 2
            crop_right = crop_left + target_width
            crop_bottom = crop_top + target_height
            
            print(f"  Cropping: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})")
            
            # Crop to center region
            cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Verify dimensions match target
            if cropped.size != (target_width, target_height):
                print(f"  Warning: Cropped size {cropped.size} doesn't match target, resizing...")
                cropped = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Save final image
            cropped.save(str(output_file))
            
            # Clean up temp file
            import os
            if os.path.exists(temp_screenshot):
                os.remove(temp_screenshot)
            
            # Verify final dimensions
            final_img = Image.open(output_file)
            actual_width, actual_height = final_img.size
            print(f"✓ Aerial view rendered and saved to: {output_file}")
            print(f"  Final size: {actual_width}x{actual_height} (target: {target_width}x{target_height})")
        except Exception as e:
            print(f"✗ Failed to take screenshot: {e}")
            raise
        
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
        - Supports HFOV matching SatSim's horizontal field of view
        - Supports custom output dimensions
        
        Args:
            longitude: Agent longitude (degrees).
            latitude: Agent latitude (degrees).
            altitude: Agent altitude (meters).
            rotation: Agent rotation/roll angle (degrees, 0=North).
            hfov: Horizontal field of view (degrees). Defaults to config value or 90.
            width: Output image width (pixels). If None, uses config or browser window width.
            height: Output image height (pixels). If None, uses config or browser window height.
            output_path: Output image path. If None, uses config value.
            
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
        
        # Convert SatSim parameters to CesiumJS parameters
        lat = latitude
        lng = longitude
        height = altitude
        heading = rotation  # SatSim rotation → CesiumJS heading
        pitch = -90.0       # Always vertical down view (SatSim style)
        roll = 0.0          # No roll
        
        # Handle custom dimensions
        if width is not None or height is not None:
            # Temporarily override camera dimensions
            original_camera_width = self.config.CAMERA.WIDTH
            original_camera_height = self.config.CAMERA.HEIGHT
            if width is not None and height is not None:
                self.config.CAMERA.WIDTH = width
                self.config.CAMERA.HEIGHT = height
            elif width is not None:
                # Keep aspect ratio if only width provided
                aspect = width / (self.config.CAMERA.HEIGHT if height is None else height)
                new_height = int(width / aspect)
                self.config.CAMERA.WIDTH = width
                self.config.CAMERA.HEIGHT = new_height
            elif height is not None:
                # Keep aspect ratio if only height provided
                aspect = (self.config.CAMERA.WIDTH if width is None else width) / height
                new_width = int(height * aspect)
                self.config.CAMERA.WIDTH = new_width
                self.config.CAMERA.HEIGHT = height
            
            # Recreate driver with new window size if it exists
            if self.driver is not None:
                self.close()
                self.driver = None
        
        # Render using standard render method with FOV
        return self.render(
            lat=lat,
            lng=lng,
            height=height,
            heading=heading,
            pitch=pitch,
            roll=roll,
            hfov=hfov,
            output_path=output_path
        )
    
    def close(self):
        """Close browser and cleanup."""
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        # Cleanup temporary HTML
        if self.html_path and self.html_path.exists():
            self.html_path.unlink()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Render aerial 3D view using CesiumJS and Google 3D Tiles (SatSim-compatible mode)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (overrides config)"
    )
    parser.add_argument(
        "--longitude", type=float, default=None, help="Longitude (degrees)"
    )
    parser.add_argument(
        "--latitude", type=float, default=None, help="Latitude (degrees)"
    )
    parser.add_argument(
        "--altitude", type=float, default=None, help="Altitude (meters)"
    )
    parser.add_argument(
        "--rotation", type=float, default=None, help="Rotation (degrees, 0=North)"
    )
    parser.add_argument(
        "--hfov", type=float, default=None, help="Horizontal field of view (degrees)"
    )
    # Legacy arguments for backward compatibility (mapped to SatSim parameters)
    parser.add_argument(
        "--lat", type=float, default=None, help="Latitude (degrees, legacy, use --latitude)"
    )
    parser.add_argument(
        "--lng", type=float, default=None, help="Longitude (degrees, legacy, use --longitude)"
    )
    parser.add_argument(
        "--height", type=float, default=None, help="Height (meters, legacy, use --altitude)"
    )
    parser.add_argument(
        "--heading", type=float, default=None, help="Heading (degrees, legacy, use --rotation)"
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return
    
    # Create renderer
    renderer = AerialRenderer(str(config_path))
    
    try:
        # Use SatSim-compatible rendering
        # Map legacy arguments to new arguments
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



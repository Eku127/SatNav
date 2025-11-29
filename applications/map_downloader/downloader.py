#!/usr/bin/env python3
"""Google Maps downloader for fetching satellite map tiles."""

import math
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from pyproj import Geod
from tqdm import tqdm


class GoogleMapDownloader:
    """
    Google Maps downloader.
    
    This class initializes with configuration arguments (parsed by ArgumentParser)
    and provides methods to fetch map tiles, convert between length and pixels,
    and compute geographic offsets.
    """

    def __init__(self, config):
        """Initialize downloader with configuration.
        
        Args:
            config: Configuration object with attributes:
                - api_key: Google Maps API key
                - zoom: Zoom level
                - signature: Optional signature for API requests
                - x: Center longitude
                - y: Center latitude
                - download_mode: Optional download mode ("sequential" or "parallel")
                - max_workers: Optional max workers for parallel download
        """
        self.api_key = config.api_key
        self.zoom = config.zoom
        self.signature = config.signature
        self.x = config.x
        self.y = config.y
        self.download_mode = getattr(config, 'download_mode', 'sequential')
        self.max_workers = getattr(config, 'max_workers', None)
        
        # Create session for connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount('https://', adapter)
    
    def length_to_pixels(self, length_m: float) -> int:
        """
        Convert ground length (meters) to pixel count at a given zoom level and latitude.
        
        Args:
            length_m (float): Ground length in meters.
        
        Returns:
            int: Equivalent pixel count.
        """
        R = 6378137  # Earth radius (meters)
        C = 2 * math.pi * R  # Earth circumference (meters)
        phi = math.radians(self.y)  # Latitude in radians
        resolution = (C * math.cos(phi)) / (256 * (2 ** self.zoom))
        pixels = length_m / resolution
        return math.ceil(pixels)

    def pixels_to_length(self, pixels: int) -> float:
        """
        Convert pixel count to ground length (meters) at a given zoom level and latitude.
        
        Args:
            pixels (int): Pixel count.
        
        Returns:
            float: Ground length in meters.
        """
        R = 6378137
        C = 2 * math.pi * R
        phi = math.radians(self.y)
        resolution = (C * math.cos(phi)) / (256 * (2 ** self.zoom))
        length_m = pixels * resolution
        return length_m

    def _download_single_tile(self, r: int, c: int, target_w: int, target_h: int, 
                               tile_w: int, tile_h: int, maptype: str, max_retries: int = 3) -> Tuple[int, int, bytes]:
        """Download a single tile with retry mechanism.
        
        Args:
            r: Row index
            c: Column index
            target_w: Target image width
            target_h: Target image height
            tile_w: Tile width
            tile_h: Tile height
            maptype: Map type
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (r, c, tile_content)
            
        Raises:
            Exception: If download fails after all retries
        """
        new_lon, new_lat = self.offset_center(
            lon=self.x,
            lat=self.y,
            dx_pixels=c * tile_w + tile_w // 2 - target_w // 2,
            dy_pixels=r * tile_h + tile_h // 2 - target_h // 2
        )
        params = {
            "center": f"{new_lat},{new_lon}",
            "zoom": self.zoom,
            "size": f"{640}x{640}",
            "maptype": maptype,
            "key": self.api_key
        }
        if self.signature:
            params["signature"] = self.signature

        # Retry logic for network errors
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Increased timeout: (connect timeout, read timeout)
                # Connect timeout increased to 15s for SSL handshake in concurrent scenarios
                response = self.session.get(
                    "https://maps.googleapis.com/maps/api/staticmap",
                    params=params,
                    timeout=(15, 30)  # (connect timeout, read timeout)
                )
                response.raise_for_status()
                return (r, c, response.content)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
            except Exception as e:
                # For other errors (e.g., API errors), don't retry
                raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to download tile after {max_retries} attempts: {last_exception}")

    def get_map(self, size=(1000, 500), maptype="satellite") -> np.ndarray:
        """
        Fetch map image using Google Maps Static API.
        
        If requested size exceeds 640x640, the image will be fetched in tiles,
        stitched together, and cropped to the target size.
        
        Args:
            size (tuple): (width, height) of the target image in pixels.
            maptype (str): Map type (roadmap, satellite, hybrid, terrain).
        
        Returns:
            np.ndarray: Map image as NumPy array (H, W, 3) uint8.
        """
        target_w, target_h = size
        tile_w, tile_h = 640, 600

        cols = math.ceil(target_w / tile_w)
        rows = math.ceil(target_h / tile_h)

        canvas_w = cols * tile_w
        canvas_h = rows * tile_h

        big_image = Image.new("RGB", (canvas_w, canvas_h))

        total_tiles = rows * cols
        
        # Determine download mode
        use_parallel = self.download_mode == "parallel"
        
        if use_parallel:
            # Parallel download mode
            if self.max_workers is None:
                # Default: use CPU count, but cap at 20 to avoid API rate limits
                self.max_workers = min(os.cpu_count() or 4, 20)
            
            print(f"  Download mode: Parallel ({self.max_workers} workers)")
            
            tile_positions = [(r, c) for r in range(rows) for c in range(cols)]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                with tqdm(total=total_tiles, desc="Downloading tiles", unit="tile") as pbar:
                    # Submit all download tasks
                    futures = {
                        executor.submit(
                            self._download_single_tile,
                            r, c, target_w, target_h, tile_w, tile_h, maptype
                        ): (r, c)
                        for r, c in tile_positions
                    }
                    
                    # Process completed downloads
                    failed_tiles = []
                    for future in as_completed(futures):
                        try:
                            r, c, content = future.result()
                            tile_img = Image.open(BytesIO(content)).convert("RGB")
                            big_image.paste(
                                tile_img.crop((0, 0, tile_w, tile_h)),
                                (c * tile_w, r * tile_h)
                            )
                            pbar.update(1)
                        except Exception as e:
                            r, c = futures[future]
                            failed_tiles.append((r, c, e))
                            # Don't fail immediately, collect failed tiles for retry
                    
                    # Retry failed tiles
                    if failed_tiles:
                        print(f"\n⚠ {len(failed_tiles)} tiles failed, retrying...")
                        retry_futures = {}
                        for r, c, _ in failed_tiles:
                            retry_futures[
                                executor.submit(
                                    self._download_single_tile,
                                    r, c, target_w, target_h, tile_w, tile_h, maptype, max_retries=5
                                )
                            ] = (r, c)
                        
                        for future in as_completed(retry_futures):
                            try:
                                r, c, content = future.result()
                                tile_img = Image.open(BytesIO(content)).convert("RGB")
                                big_image.paste(
                                    tile_img.crop((0, 0, tile_w, tile_h)),
                                    (c * tile_w, r * tile_h)
                                )
                                pbar.update(1)
                            except Exception as e:
                                r, c = retry_futures[future]
                                raise Exception(f"Failed to download tile at ({r}, {c}) after retries: {e}")
        else:
            # Sequential download mode (original)
            print(f"  Download mode: Sequential")
            
            with tqdm(total=total_tiles, desc="Downloading tiles", unit="tile") as pbar:
                for r in range(rows):
                    for c in range(cols):
                        _, _, content = self._download_single_tile(
                            r, c, target_w, target_h, tile_w, tile_h, maptype
                        )
                        tile_img = Image.open(BytesIO(content)).convert("RGB")
                        big_image.paste(
                            tile_img.crop((0, 0, tile_w, tile_h)),
                            (c * tile_w, r * tile_h)
                        )
                        pbar.update(1)

        final_img = big_image.crop((0, 0, target_w, target_h))
        return np.array(final_img)

    def offset_center(self, lon: float, lat: float, dx_pixels: int, dy_pixels: int) -> Tuple[float, float]:
        """
        Compute new center coordinates given pixel offsets.
        
        Args:
            lon (float): Current longitude.
            lat (float): Current latitude.
            dx_pixels (int): Horizontal pixel offset (positive = right).
            dy_pixels (int): Vertical pixel offset (positive = down).
        
        Returns:
            tuple: (new_lon, new_lat)
        """
        dx_m = self.pixels_to_length(dx_pixels)
        dy_m = self.pixels_to_length(dy_pixels)

        geod = Geod(ellps="WGS84")

        new_lon, new_lat, _ = geod.fwd(lon, lat, 90, dx_m)   # East-West offset
        new_lon, new_lat, _ = geod.fwd(new_lon, new_lat, 0, -dy_m)  # North-South offset

        return new_lon, new_lat


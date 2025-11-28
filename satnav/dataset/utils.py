from jsonargparse import ArgumentParser
import rasterio
from rasterio.vrt import WarpedVRT
from pyproj import Geod, Transformer
import argparse
from rasterio.transform import from_origin
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cv2
import math
import requests
from PIL import Image
from io import BytesIO


class GoogleMapDownloader:
    """
    Google Maps downloader.
    
    This class initializes with configuration arguments (parsed by ArgumentParser)
    and provides methods to fetch map tiles, convert between length and pixels,
    and compute geographic offsets.
    """

    def __init__(self, config):
        """Initialize downloader with configuration."""
        self.api_key = config.api_key
        self.zoom = config.zoom
        self.signature = config.signature
        self.x = config.x
        self.y = config.y
    
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

    def get_map(self, size=(1000, 500), maptype="satellite") -> np.ndarray:
        """
        Fetch map image using Google Maps Static API.
        
        If requested size exceeds 640x640, the image will be fetched in tiles,
        stitched together, and cropped to the target size.
        
        Args:
            size (tuple): (width, height) of the target image in pixels.
            maptype (str): Map type (roadmap, satellite, hybrid, terrain).
        
        Returns:
            np.ndarray: Map image as NumPy array.
        """
        target_w, target_h = size
        tile_w, tile_h = 640, 600

        cols = math.ceil(target_w / tile_w)
        rows = math.ceil(target_h / tile_h)

        canvas_w = cols * tile_w
        canvas_h = rows * tile_h

        big_image = Image.new("RGB", (canvas_w, canvas_h))

        for r in range(rows):
            for c in range(cols):
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

                response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params)
                if response.status_code != 200:
                    raise Exception(f"Request failed: {response.status_code}, {response.text}")

                tile_img = Image.open(BytesIO(response.content)).convert("RGB")
                big_image.paste(tile_img.crop((0, 0, tile_w, tile_h)), (c * tile_w, r * tile_h))

        final_img = big_image.crop((0, 0, target_w, target_h))
        return np.array(final_img)

    def offset_center(self, lon: float, lat: float, dx_pixels: int, dy_pixels: int):
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


def generate_geotiff(latlng1=(39.900, 116.397), latlng2=(39.903, 116.400),
                     api_key="", zoom=19, out_path="output.tif", signature=None):
    """
    Generate a GeoTIFF file for a rectangular region defined by two lat/lon coordinates.
    
    Args:
        latlng1 (tuple): Lower-left corner (lat, lon).
        latlng2 (tuple): Upper-right corner (lat, lon).
        api_key (str): Google Maps API key.
        zoom (int): Zoom level.
        out_path (str): Output file path.
        signature (str, optional): Optional signature for API requests.
    """

    def rectangle_info(lon1, lat1, lon2, lat2):
        geod = Geod(ellps="WGS84")
        mid_lon = (lon1 + lon2) / 2
        mid_lat = (lat1 + lat2) / 2
        _, _, w_length = geod.inv(lon1, lat1, lon2, lat1)
        _, _, h_length = geod.inv(lon1, lat1, lon1, lat2)
        return mid_lon, mid_lat, h_length, w_length
    
    lat1, lon1 = latlng1
    lat2, lon2 = latlng2
    mid_lon, mid_lat, h_length, w_length = rectangle_info(lon1, lat1, lon2, lat2)

    args = argparse.Namespace(
        api_key=api_key,
        signature=signature,
        zoom=zoom,
        x=mid_lon,
        y=mid_lat
    )

    downloader = GoogleMapDownloader(args)
    image_np = downloader.get_map(
        size=(downloader.length_to_pixels(w_length), downloader.length_to_pixels(h_length)),
        maptype="satellite"
    )

    def save_geotiff(img_np, lon_min, lat_min, lon_max, lat_max, out_path="output.tif"):
        """
        Save NumPy image array as a GeoTIFF (EPSG:3857).
        
        Args:
            img_np (np.ndarray): Image array (H, W, C).
            lon_min (float): Lower-left longitude.
            lat_min (float): Lower-left latitude.
            lon_max (float): Upper-right longitude.
            lat_max (float): Upper-right latitude.
            out_path (str): Output file path.
        """
        height, width, bands = img_np.shape

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)

        pixel_size_x = (x_max - x_min) / width
        pixel_size_y = (y_max - y_min) / height

        transform = from_origin(x_min, y_max, pixel_size_x, pixel_size_y)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=img_np.dtype,
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            for i in range(bands):
                dst.write(img_np[:, :, i], i + 1)

        print(f"GeoTIFF saved to {out_path}")

    save_geotiff(image_np, lon1, lat1, lon2, lat2, out_path=out_path)
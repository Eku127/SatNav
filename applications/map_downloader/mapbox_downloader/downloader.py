#!/usr/bin/env python3
"""Mapbox raster tile downloader for building satellite mosaics."""

from __future__ import annotations

import html
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

WEB_MERCATOR_LAT_LIMIT = 85.05112878


@dataclass(frozen=True)
class TileJSONMetadata:
    """TileJSON metadata returned by Mapbox."""

    tileset_id: str
    tilejson_url: str
    minzoom: Optional[int]
    maxzoom: Optional[int]
    bounds: Optional[Tuple[float, float, float, float]]
    attribution_html: str
    attribution_text: str
    name: Optional[str]
    mapbox_logo: Optional[bool]


@dataclass(frozen=True)
class TileWindow:
    """Tile coverage plus exact crop bounds in pixel space."""

    left: int
    top: int
    right: int
    bottom: int
    crop_left_px: float
    crop_top_px: float
    crop_right_px: float
    crop_bottom_px: float
    width_px: int
    height_px: int

    @property
    def cols(self) -> int:
        return self.right - self.left + 1

    @property
    def rows(self) -> int:
        return self.bottom - self.top + 1

    @property
    def total_tiles(self) -> int:
        return self.cols * self.rows


@dataclass(frozen=True)
class DownloadResult:
    """Final stitched image plus metadata."""

    image: np.ndarray
    bbox_wgs84: Tuple[float, float, float, float]
    tile_window: TileWindow
    tilejson: TileJSONMetadata


class MapboxTileDownloader:
    """Download Mapbox raster tiles and stitch them into an RGB image."""

    def __init__(self, config):
        self.access_token = config.access_token
        self.zoom = int(config.zoom)
        self.tileset_id = getattr(config, "tileset_id", "mapbox.satellite")
        self.image_format = getattr(config, "image_format", "jpg90").lstrip(".")
        self.scale = int(getattr(config, "scale", 1))
        self.download_mode = getattr(config, "download_mode", "parallel")
        self.max_workers = getattr(config, "max_workers", None)
        self.use_env_proxy = bool(getattr(config, "use_env_proxy", False))

        if self.scale not in (1, 2):
            raise ValueError("scale must be 1 or 2")
        if self.download_mode not in ("sequential", "parallel"):
            raise ValueError("download_mode must be 'sequential' or 'parallel'")

        self.tile_size = 256 * self.scale
        self._tilejson: Optional[TileJSONMetadata] = None

        self.session = requests.Session()
        pool_size = max(10, (self.max_workers or 8) * 2)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self.session.trust_env = self.use_env_proxy
        self.session.headers.update({"User-Agent": "mapbox-downloader/1.0"})
        self.session.mount("https://", adapter)

    @staticmethod
    def _strip_html_tags(value: Optional[str]) -> str:
        if not value:
            return ""
        no_tags = re.sub(r"<[^>]+>", " ", value)
        normalized = html.unescape(no_tags)
        return " ".join(normalized.split())

    @staticmethod
    def _clamp_latitude(latitude: float) -> float:
        if latitude < -WEB_MERCATOR_LAT_LIMIT or latitude > WEB_MERCATOR_LAT_LIMIT:
            raise ValueError(
                "Latitude must be within Web Mercator limits "
                f"[-{WEB_MERCATOR_LAT_LIMIT}, {WEB_MERCATOR_LAT_LIMIT}]"
            )
        return latitude

    @staticmethod
    def lon_to_tile_x(longitude: float, zoom: int) -> float:
        n = 2**zoom
        return (longitude + 180.0) / 360.0 * n

    @staticmethod
    def lat_to_tile_y(latitude: float, zoom: int) -> float:
        latitude = MapboxTileDownloader._clamp_latitude(latitude)
        lat_rad = math.radians(latitude)
        n = 2**zoom
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * n / 2.0

    def fetch_tilejson(self) -> TileJSONMetadata:
        """Retrieve TileJSON metadata for the configured tileset."""
        if self._tilejson is not None:
            return self._tilejson

        tilejson_url = f"https://api.mapbox.com/v4/{self.tileset_id}.json"
        response = self.session.get(
            tilejson_url,
            params={"access_token": self.access_token},
            timeout=(15, 30),
        )
        response.raise_for_status()

        data = response.json()
        bounds = data.get("bounds")
        bounds_tuple = tuple(bounds) if isinstance(bounds, list) and len(bounds) == 4 else None

        self._tilejson = TileJSONMetadata(
            tileset_id=self.tileset_id,
            tilejson_url=tilejson_url,
            minzoom=int(data["minzoom"]) if data.get("minzoom") is not None else None,
            maxzoom=int(data["maxzoom"]) if data.get("maxzoom") is not None else None,
            bounds=bounds_tuple,
            attribution_html=data.get("attribution", ""),
            attribution_text=self._strip_html_tags(data.get("attribution")),
            name=data.get("name"),
            mapbox_logo=data.get("mapbox_logo"),
        )
        return self._tilejson

    def _build_tile_url(self, x: int, y: int) -> str:
        scale_suffix = "@2x" if self.scale == 2 else ""
        return (
            f"https://api.mapbox.com/v4/{self.tileset_id}/"
            f"{self.zoom}/{x}/{y}{scale_suffix}.{self.image_format}"
        )

    def _download_single_tile(
        self,
        x: int,
        y: int,
        max_retries: int = 3,
    ) -> Tuple[int, int, bytes]:
        """Download a single tile with retry support."""
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    self._build_tile_url(x, y),
                    params={"access_token": self.access_token},
                    timeout=(15, 30),
                )
                response.raise_for_status()
                return x, y, response.content
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
            except Exception as exc:
                last_exception = exc
                break

        raise RuntimeError(
            f"Failed to download tile z={self.zoom}, x={x}, y={y}: {last_exception}"
        )

    def _build_tile_window(
        self,
        x_min_f: float,
        x_max_f: float,
        y_north_f: float,
        y_south_f: float,
    ) -> TileWindow:
        max_index = (2**self.zoom) - 1

        left = max(0, min(max_index, math.floor(x_min_f)))
        right = max(0, min(max_index, math.ceil(x_max_f) - 1))
        top = max(0, min(max_index, math.floor(y_north_f)))
        bottom = max(0, min(max_index, math.ceil(y_south_f) - 1))

        if left > right or top > bottom:
            raise ValueError("Requested area does not overlap any tiles at the selected zoom")

        crop_left_px = (x_min_f - left) * self.tile_size
        crop_right_px = (x_max_f - left) * self.tile_size
        crop_top_px = (y_north_f - top) * self.tile_size
        crop_bottom_px = (y_south_f - top) * self.tile_size

        width_px = max(1, int(round(crop_right_px - crop_left_px)))
        height_px = max(1, int(round(crop_bottom_px - crop_top_px)))

        return TileWindow(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            crop_left_px=crop_left_px,
            crop_top_px=crop_top_px,
            crop_right_px=crop_right_px,
            crop_bottom_px=crop_bottom_px,
            width_px=width_px,
            height_px=height_px,
        )

    def _stitch_tiles(self, tile_window: TileWindow) -> Image.Image:
        canvas = Image.new(
            "RGB",
            (tile_window.cols * self.tile_size, tile_window.rows * self.tile_size),
        )
        tile_positions = [
            (x, y)
            for y in range(tile_window.top, tile_window.bottom + 1)
            for x in range(tile_window.left, tile_window.right + 1)
        ]

        if self.download_mode == "parallel":
            worker_count = self.max_workers or min(os.cpu_count() or 4, 16)
            print(f"  Download mode: Parallel ({worker_count} workers)")
            failed_tiles = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                with tqdm(total=tile_window.total_tiles, desc="Downloading tiles", unit="tile") as pbar:
                    futures = {
                        executor.submit(self._download_single_tile, x, y): (x, y)
                        for x, y in tile_positions
                    }
                    for future in as_completed(futures):
                        try:
                            x, y, content = future.result()
                            tile_img = Image.open(BytesIO(content)).convert("RGB")
                            canvas.paste(
                                tile_img,
                                (
                                    (x - tile_window.left) * self.tile_size,
                                    (y - tile_window.top) * self.tile_size,
                                ),
                            )
                            pbar.update(1)
                        except Exception as exc:
                            failed_tiles.append((*futures[future], exc))

                    if failed_tiles:
                        print(f"\n  Retrying {len(failed_tiles)} failed tiles...")
                        retry_futures = {
                            executor.submit(self._download_single_tile, x, y, 5): (x, y)
                            for x, y, _ in failed_tiles
                        }
                        for future in as_completed(retry_futures):
                            x, y = retry_futures[future]
                            x, y, content = future.result()
                            tile_img = Image.open(BytesIO(content)).convert("RGB")
                            canvas.paste(
                                tile_img,
                                (
                                    (x - tile_window.left) * self.tile_size,
                                    (y - tile_window.top) * self.tile_size,
                                ),
                            )
                            pbar.update(1)
        else:
            print("  Download mode: Sequential")
            with tqdm(total=tile_window.total_tiles, desc="Downloading tiles", unit="tile") as pbar:
                for x, y in tile_positions:
                    _, _, content = self._download_single_tile(x, y)
                    tile_img = Image.open(BytesIO(content)).convert("RGB")
                    canvas.paste(
                        tile_img,
                        (
                            (x - tile_window.left) * self.tile_size,
                            (y - tile_window.top) * self.tile_size,
                        ),
                    )
                    pbar.update(1)

        return canvas

    def _crop_canvas(self, canvas: Image.Image, tile_window: TileWindow) -> Image.Image:
        return canvas.transform(
            (tile_window.width_px, tile_window.height_px),
            Image.Transform.EXTENT,
            (
                tile_window.crop_left_px,
                tile_window.crop_top_px,
                tile_window.crop_right_px,
                tile_window.crop_bottom_px,
            ),
            resample=Image.Resampling.BICUBIC,
        )

    def download_bbox(
        self,
        latlng1: Tuple[float, float],
        latlng2: Tuple[float, float],
    ) -> DownloadResult:
        """Download an exact WGS84 bounding box and return the cropped RGB image."""
        lat1, lon1 = latlng1
        lat2, lon2 = latlng2

        lat_min = min(lat1, lat2)
        lat_max = max(lat1, lat2)
        lon_min = min(lon1, lon2)
        lon_max = max(lon1, lon2)

        if lat_min >= lat_max:
            raise ValueError("Bounding box latitude range must be non-zero")
        if lon_min >= lon_max:
            raise ValueError("Bounding box longitude range must be non-zero")
        if lon_min < -180.0 or lon_max > 180.0:
            raise ValueError("Longitude must stay within [-180, 180]")

        self._clamp_latitude(lat_min)
        self._clamp_latitude(lat_max)

        tilejson = self.fetch_tilejson()
        if tilejson.minzoom is not None and self.zoom < tilejson.minzoom:
            raise ValueError(
                f"Zoom {self.zoom} is below tileset minimum zoom {tilejson.minzoom}"
            )

        x_min_f = self.lon_to_tile_x(lon_min, self.zoom)
        x_max_f = self.lon_to_tile_x(lon_max, self.zoom)
        y_north_f = self.lat_to_tile_y(lat_max, self.zoom)
        y_south_f = self.lat_to_tile_y(lat_min, self.zoom)

        tile_window = self._build_tile_window(
            x_min_f=x_min_f,
            x_max_f=x_max_f,
            y_north_f=y_north_f,
            y_south_f=y_south_f,
        )
        canvas = self._stitch_tiles(tile_window)
        cropped = self._crop_canvas(canvas, tile_window)

        return DownloadResult(
            image=np.array(cropped),
            bbox_wgs84=(lon_min, lat_min, lon_max, lat_max),
            tile_window=tile_window,
            tilejson=tilejson,
        )

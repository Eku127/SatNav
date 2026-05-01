#!/usr/bin/env python3
"""Google Map Tiles API XYZ downloader for stitched satellite mosaics."""

from __future__ import annotations

import html
import math
import os
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
class GoogleTileSession:
    """Session information returned by Google Map Tiles API."""

    session: str
    expiry: Optional[str]
    tile_width: int
    tile_height: int
    image_format: str


@dataclass(frozen=True)
class GoogleViewportMetadata:
    """Viewport metadata returned by Google Map Tiles API."""

    copyright: str
    max_zoom_rects: list


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
class GoogleXYZDownloadResult:
    """Final stitched image plus metadata."""

    image: np.ndarray
    bbox_wgs84: Tuple[float, float, float, float]
    tile_window: TileWindow
    session: GoogleTileSession
    viewport: GoogleViewportMetadata


class GoogleXYZTileDownloader:
    """Download Google Map Tiles API 2D tiles by z/x/y and stitch them."""

    def __init__(self, config):
        self.api_key = config.api_key
        self.zoom = int(config.zoom)
        self.map_type = getattr(config, "map_type", "satellite")
        self.language = getattr(config, "language", "en-US")
        self.region = getattr(config, "region", "US")
        self.image_format = getattr(config, "image_format", "jpeg")
        self.scale = getattr(config, "scale", "scaleFactor1x")
        self.high_dpi = bool(getattr(config, "high_dpi", False))
        self.download_mode = getattr(config, "download_mode", "parallel")
        self.max_workers = getattr(config, "max_workers", None)
        self.use_env_proxy = bool(getattr(config, "use_env_proxy", False))

        if self.download_mode not in ("sequential", "parallel"):
            raise ValueError("download_mode must be 'sequential' or 'parallel'")
        if self.map_type not in ("roadmap", "satellite", "terrain"):
            raise ValueError("map_type must be roadmap, satellite, or terrain")
        if self.image_format not in ("jpeg", "png"):
            raise ValueError("image_format must be jpeg or png")

        self.session = requests.Session()
        self.session.trust_env = self.use_env_proxy
        pool_size = max(10, (self.max_workers or 8) * 2)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self.session.mount("https://", adapter)

        self._tile_session: Optional[GoogleTileSession] = None
        self.tile_size = 256

    @staticmethod
    def _strip_html_tags(value: Optional[str]) -> str:
        if not value:
            return ""
        no_tags = html.unescape(value)
        return " ".join(no_tags.split())

    @staticmethod
    def _raise_for_status_with_body(response: requests.Response) -> None:
        """Raise HTTP errors with a short body snippet for API debugging."""
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{exc}; response body: {response.text[:500]}"
            ) from exc

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
        return (longitude + 180.0) / 360.0 * (2**zoom)

    @staticmethod
    def lat_to_tile_y(latitude: float, zoom: int) -> float:
        latitude = GoogleXYZTileDownloader._clamp_latitude(latitude)
        lat_rad = math.radians(latitude)
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * (2**zoom) / 2.0

    def create_session(self) -> GoogleTileSession:
        """Create or reuse a Google Map Tiles API session token."""
        if self._tile_session is not None:
            return self._tile_session

        body = {
            "mapType": self.map_type,
            "language": self.language,
            "region": self.region,
            "imageFormat": self.image_format,
            "scale": self.scale,
            "highDpi": self.high_dpi,
        }
        if self.map_type == "terrain":
            body["layerTypes"] = ["layerRoadmap"]

        response = self.session.post(
            "https://tile.googleapis.com/v1/createSession",
            params={"key": self.api_key},
            json=body,
            timeout=(15, 30),
        )
        self._raise_for_status_with_body(response)
        data = response.json()

        self._tile_session = GoogleTileSession(
            session=data["session"],
            expiry=data.get("expiry"),
            tile_width=int(data.get("tileWidth", 256)),
            tile_height=int(data.get("tileHeight", 256)),
            image_format=data.get("imageFormat", self.image_format),
        )
        if self._tile_session.tile_width != self._tile_session.tile_height:
            raise ValueError("Non-square Google tiles are not supported")
        self.tile_size = self._tile_session.tile_width
        return self._tile_session

    def fetch_viewport(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> GoogleViewportMetadata:
        """Fetch viewport attribution and max-zoom metadata."""
        tile_session = self.create_session()
        response = self.session.get(
            "https://tile.googleapis.com/tile/v1/viewport",
            params={
                "session": tile_session.session,
                "key": self.api_key,
                "zoom": self.zoom,
                "north": lat_max,
                "south": lat_min,
                "east": lon_max,
                "west": lon_min,
            },
            timeout=(15, 30),
        )
        self._raise_for_status_with_body(response)
        data = response.json()
        return GoogleViewportMetadata(
            copyright=self._strip_html_tags(data.get("copyright", "")),
            max_zoom_rects=data.get("maxZoomRects", []),
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

        return TileWindow(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            crop_left_px=crop_left_px,
            crop_top_px=crop_top_px,
            crop_right_px=crop_right_px,
            crop_bottom_px=crop_bottom_px,
            width_px=max(1, int(round(crop_right_px - crop_left_px))),
            height_px=max(1, int(round(crop_bottom_px - crop_top_px))),
        )

    def _download_single_tile(
        self,
        x: int,
        y: int,
        max_retries: int = 3,
    ) -> Tuple[int, int, bytes]:
        """Download one z/x/y tile."""
        tile_session = self.create_session()
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"https://tile.googleapis.com/v1/2dtiles/{self.zoom}/{x}/{y}",
                    params={
                        "session": tile_session.session,
                        "key": self.api_key,
                    },
                    timeout=(15, 30),
                )
                self._raise_for_status_with_body(response)
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
            f"Failed to download Google XYZ tile z={self.zoom}, x={x}, y={y}: {last_exception}"
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
    ) -> GoogleXYZDownloadResult:
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
        tile_session = self.create_session()
        viewport = self.fetch_viewport(lon_min, lat_min, lon_max, lat_max)

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

        return GoogleXYZDownloadResult(
            image=np.array(cropped),
            bbox_wgs84=(lon_min, lat_min, lon_max, lat_max),
            tile_window=tile_window,
            session=tile_session,
            viewport=viewport,
        )

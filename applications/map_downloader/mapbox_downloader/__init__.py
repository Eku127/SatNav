"""Mapbox-based map downloader package."""

from .downloader import DownloadResult, MapboxTileDownloader, TileJSONMetadata, TileWindow


def __getattr__(name):
    if name == "generate_geotiff":
        from .generate_geotiff import generate_geotiff

        return generate_geotiff
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DownloadResult",
    "MapboxTileDownloader",
    "TileJSONMetadata",
    "TileWindow",
    "generate_geotiff",
]

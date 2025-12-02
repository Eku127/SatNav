"""Map downloader application for generating GeoTIFF files from Google Maps."""

from .downloader import GoogleMapDownloader

# Lazy import to avoid RuntimeWarning when running as module with python -m
# Use: from applications.map_downloader.generate_geotiff import generate_geotiff
# Or: from applications.map_downloader import generate_geotiff (will work but triggers import)
def __getattr__(name):
    if name == "generate_geotiff":
        from .generate_geotiff import generate_geotiff
        return generate_geotiff
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["GoogleMapDownloader", "generate_geotiff"]


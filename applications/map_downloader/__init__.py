"""Map downloader application for SatNav.

The package provides provider-specific downloaders under:

- applications.map_downloader.google_downloader
- applications.map_downloader.mapbox_downloader
"""

from .main import main

__all__ = ["main"]

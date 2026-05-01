#!/usr/bin/env python3
"""Provider dispatcher for SatNav map downloaders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional


PROVIDERS = ("google", "mapbox")


def _load_default_provider() -> str:
    config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        return "google"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "google"
    provider = str(config.get("DEFAULT_PROVIDER", "google")).strip().lower()
    return provider if provider in PROVIDERS else "google"


def _provider_main(provider: str) -> Callable[[], None]:
    if provider == "google":
        from .google_downloader.generate_geotiff import main as google_main

        return google_main
    if provider == "mapbox":
        from .mapbox_downloader.generate_geotiff import main as mapbox_main

        return mapbox_main
    raise ValueError(f"Unsupported map provider: {provider}")


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to a provider-specific downloader CLI."""
    args = list(sys.argv[1:] if argv is None else argv)

    provider_explicit = bool(args and args[0] in PROVIDERS)
    if provider_explicit:
        provider = args.pop(0)
    else:
        provider = _load_default_provider()

    if args and args[0] in ("-h", "--help") and not provider_explicit:
        parser = argparse.ArgumentParser(
            prog="python -m applications.map_downloader",
            description="Download satellite map tiles and export GeoTIFF files.",
        )
        parser.add_argument(
            "provider",
            nargs="?",
            choices=PROVIDERS,
            default="google",
            help="Tile provider to use. Defaults to google for backward compatibility.",
        )
        parser.add_argument(
            "provider_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed to the selected provider.",
        )
        parser.print_help()
        return 0

    sys.argv = [sys.argv[0]] + args
    _provider_main(provider)()
    return 0

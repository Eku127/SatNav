#!/usr/bin/env python3
"""Generate GeoTIFF files from Google Map Tiles API XYZ imagery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import yaml
from PIL import Image, ImageDraw, ImageFont
from pyproj import Geod, Transformer
from rasterio.transform import from_origin

from .downloader import GoogleXYZTileDownloader


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file if present."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cfg_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely read nested dict values."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def read_windows_env_var(name: str) -> Optional[str]:
    """Read a Windows environment variable from the registry as a fallback."""
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    locations = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for root, subkey in locations:
        try:
            with winreg.OpenKey(root, subkey) as handle:
                value, _ = winreg.QueryValueEx(handle, name)
                if value:
                    return str(value)
        except OSError:
            continue
    return None


def resolve_api_key(config: Dict[str, Any]) -> str:
    """Resolve the Google API key from env, registry fallback, or config."""
    return (
        os.environ.get("GOOGLE_MAPS_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or read_windows_env_var("GOOGLE_MAPS_API_KEY")
        or read_windows_env_var("GOOGLE_API_KEY")
        or cfg_get(config, "GOOGLE", "API_KEY")
        or cfg_get(config, "API", "API_KEY")
        or ""
    )


def to_console_text(value: str) -> str:
    """Return text that can be printed safely in the current console encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def center_to_corners(
    center_lat: float,
    center_lon: float,
    height_m: float,
    width_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Convert center point and dimensions to lower-left / upper-right corners."""
    geod = Geod(ellps="WGS84")
    half_height = height_m / 2.0
    half_width = width_m / 2.0

    lon_n, lat_n, _ = geod.fwd(center_lon, center_lat, 0, half_height)
    lon_s, lat_s, _ = geod.fwd(center_lon, center_lat, 180, half_height)

    lon_nw, lat_nw, _ = geod.fwd(lon_n, lat_n, 270, half_width)
    lon_ne, lat_ne, _ = geod.fwd(lon_n, lat_n, 90, half_width)
    lon_sw, lat_sw, _ = geod.fwd(lon_s, lat_s, 270, half_width)
    lon_se, lat_se, _ = geod.fwd(lon_s, lat_s, 90, half_width)

    lat_min = min(lat_nw, lat_ne, lat_sw, lat_se)
    lat_max = max(lat_nw, lat_ne, lat_sw, lat_se)
    lon_min = min(lon_nw, lon_ne, lon_sw, lon_se)
    lon_max = max(lon_nw, lon_ne, lon_sw, lon_se)

    return (lat_min, lon_min), (lat_max, lon_max)


def corners_to_center(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> Tuple[float, float, float, float]:
    """Convert two corners to center point plus dimensions in meters."""
    geod = Geod(ellps="WGS84")
    center_lat = (lat1 + lat2) / 2.0
    center_lon = (lon1 + lon2) / 2.0
    _, _, width_m = geod.inv(lon1, lat1, lon2, lat1)
    _, _, height_m = geod.inv(lon1, lat1, lon1, lat2)
    return center_lat, center_lon, abs(height_m), abs(width_m)


def save_geotiff(
    image: np.ndarray,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    out_path: Path,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Save an RGB image as a georeferenced GeoTIFF in EPSG:3857."""
    if image.ndim != 3:
        raise ValueError("Expected image shape (height, width, bands)")

    height, width, bands = image.shape
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    transform = from_origin(
        x_min,
        y_max,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=image.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        for band_index in range(bands):
            dst.write(image[:, :, band_index], band_index + 1)
        if tags:
            dst.update_tags(**{key: value for key, value in tags.items() if value})


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def default_google_logo_path() -> Path:
    """Return the packaged default Google Maps logo asset path."""
    return Path(__file__).parent / "assets" / "GoogleMaps_Logo_WithDarkOutline_1x.png"


def resolve_logo_path(logo_path: Optional[str]) -> Optional[Path]:
    """Resolve a configured logo path, falling back to the project-root logo."""
    candidates = []
    if logo_path:
        configured = Path(logo_path).expanduser()
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.extend(
                [
                    Path.cwd() / configured,
                    Path(__file__).resolve().parents[1] / configured,
                    Path(__file__).resolve().parent / configured,
                ]
            )
    candidates.append(default_google_logo_path())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text into lines that fit within max_width pixels."""
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def add_google_attribution_panel(
    image: np.ndarray,
    attribution: str,
    logo_path: Optional[str] = None,
    margin_px: int = 16,
    padding_px: int = 14,
    panel_opacity: int = 176,
    logo_width_px: int = 105,
    text_size_px: int = 14,
    max_width_ratio: float = 0.78,
) -> np.ndarray:
    """Add one bottom-left Google logo and attribution panel to the final image."""
    if not attribution:
        attribution = "Google"

    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    font = load_font(text_size_px)

    logo = None
    if logo_path:
        logo = Image.open(logo_path).convert("RGBA")
        target_logo_width = max(1, int(logo_width_px))
        target_logo_height = max(1, int(round(logo.height * (target_logo_width / logo.width))))
        logo = logo.resize((target_logo_width, target_logo_height), Image.Resampling.LANCZOS)

    max_panel_width = max(240, int(base.width * max_width_ratio))
    logo_width = logo.width if logo is not None else 0
    logo_height = logo.height if logo is not None else 0
    spacer = 10 if logo is not None else 0
    inner_max_width = max(80, max_panel_width - padding_px * 2 - logo_width - spacer)
    lines = wrap_text_to_width(measure_draw, attribution, font, inner_max_width) or [attribution]

    line_gap = 4
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = line_bbox[3] - line_bbox[1]
    text_width = max(
        draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        for line in lines
    )
    text_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap

    content_width = logo_width + spacer + text_width
    content_height = max(logo_height, text_height)
    panel_width = min(base.width - margin_px * 2, max(content_width + padding_px * 2, 240))
    panel_height = padding_px * 2 + content_height
    left = max(0, margin_px)
    top = max(0, base.height - margin_px - panel_height)

    draw.rounded_rectangle(
        (left, top, left + panel_width, top + panel_height),
        radius=10,
        fill=(0, 0, 0, panel_opacity),
    )
    text_x = left + padding_px + logo_width + spacer
    text_y = top + padding_px + max(0, (content_height - text_height) // 2)
    if logo is not None:
        logo_y = top + padding_px + max(0, (content_height - logo_height) // 2)
        overlay.alpha_composite(logo, (left + padding_px, logo_y))

    y = text_y
    for line in lines:
        draw.text((text_x, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height + line_gap

    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def generate_xyz_geotiff(
    latlng1: Tuple[float, float],
    latlng2: Tuple[float, float],
    api_key: str,
    zoom: int = 19,
    out_path: str = "output.tif",
    maptype: str = "satellite",
    download_mode: str = "parallel",
    max_workers: Optional[int] = None,
    image_format: str = "jpeg",
    scale: str = "scaleFactor1x",
    high_dpi: bool = False,
    language: str = "en-US",
    region: str = "US",
    use_env_proxy: bool = False,
    add_attribution_panel: bool = True,
    logo_path: Optional[str] = None,
    logo_width_px: int = 105,
) -> Path:
    """Download Google Map Tiles API XYZ imagery and save it as a GeoTIFF."""
    lat1, lon1 = latlng1
    lat2, lon2 = latlng2
    center_lat, center_lon, height_m, width_m = corners_to_center(lat1, lon1, lat2, lon2)

    print("=" * 60)
    print("Region Information:")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Dimensions: {height_m:.2f}m (height) x {width_m:.2f}m (width)")
    print(f"  Zoom level: {zoom}")
    print("  Backend: Google Map Tiles API XYZ")
    print(f"  Map type: {maptype}")
    print("=" * 60)

    args = argparse.Namespace(
        api_key=api_key,
        zoom=zoom,
        map_type=maptype,
        language=language,
        region=region,
        image_format=image_format,
        scale=scale,
        high_dpi=high_dpi,
        download_mode=download_mode,
        max_workers=max_workers,
        use_env_proxy=use_env_proxy,
    )
    downloader = GoogleXYZTileDownloader(args)
    result = downloader.download_bbox(latlng1, latlng2)

    print("\nDownload completed:")
    print(f"  Pixels: {result.image.shape[1]} x {result.image.shape[0]}")
    print(
        "  Tile coverage: "
        f"{result.tile_window.cols} x {result.tile_window.rows} = {result.tile_window.total_tiles}"
    )
    print(f"  Tile size: {result.session.tile_width} x {result.session.tile_height}")
    if result.viewport.copyright:
        print(f"  Attribution: {to_console_text(result.viewport.copyright)}")
    print("Generating GeoTIFF...")

    final_image = result.image
    watermark_mode = "disabled"
    resolved_logo_path = resolve_logo_path(logo_path)
    if add_attribution_panel:
        final_image = add_google_attribution_panel(
            image=result.image,
            attribution=result.viewport.copyright,
            logo_path=str(resolved_logo_path) if resolved_logo_path else None,
            logo_width_px=logo_width_px,
        )
        watermark_mode = "logo+text" if resolved_logo_path else "text"

    lon_min, lat_min, lon_max, lat_max = result.bbox_wgs84
    output_path = Path(out_path)
    save_geotiff(
        image=final_image,
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        out_path=output_path,
        tags={
            "source": "Google Map Tiles API",
            "backend": "xyz",
            "zoom": str(zoom),
            "maptype": maptype,
            "image_format": image_format,
            "scale": scale,
            "high_dpi": str(high_dpi),
            "tile_window": (
                f"{result.tile_window.left},{result.tile_window.top},"
                f"{result.tile_window.right},{result.tile_window.bottom}"
            ),
            "attribution": result.viewport.copyright,
            "watermark_mode": watermark_mode,
            "watermark_position": "bottom-left" if add_attribution_panel else "",
            "watermark_logo": str(resolved_logo_path) if resolved_logo_path else "",
        },
    )

    print("\nGeoTIFF saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {final_image.shape[1]} x {final_image.shape[0]} pixels")
    print("  CRS: EPSG:3857 (Web Mercator)")
    if add_attribution_panel and resolved_logo_path:
        print(f"  Watermark panel: bottom-left (logo + attribution, logo width={logo_width_px}px)")
    return output_path


# Backward-compatible name for callers that imported google_downloader.generate_geotiff.
generate_geotiff = generate_xyz_geotiff


def main() -> None:
    """Command-line entry point."""
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")
    region_type = cfg_get(config, "REGION", "TYPE", default="center")

    parser = argparse.ArgumentParser(
        description="Generate GeoTIFF files from Google Map Tiles API XYZ imagery."
    )
    parser.add_argument(
        "--backend",
        choices=["xyz"],
        default=cfg_get(config, "DOWNLOAD", "BACKEND", default="xyz"),
        help="Compatibility flag. Release builds only support xyz.",
    )
    parser.add_argument(
        "--type",
        choices=["corners", "center"],
        default=region_type,
        help=f"Region definition type (default: {region_type})",
    )
    parser.add_argument("--lat1", type=float, default=cfg_get(config, "REGION", "CORNERS", "LAT1"))
    parser.add_argument("--lon1", type=float, default=cfg_get(config, "REGION", "CORNERS", "LON1"))
    parser.add_argument("--lat2", type=float, default=cfg_get(config, "REGION", "CORNERS", "LAT2"))
    parser.add_argument("--lon2", type=float, default=cfg_get(config, "REGION", "CORNERS", "LON2"))
    parser.add_argument("--center-lat", type=float, default=cfg_get(config, "REGION", "CENTER", "LAT"))
    parser.add_argument("--center-lon", type=float, default=cfg_get(config, "REGION", "CENTER", "LON"))
    parser.add_argument("--height-m", type=float, default=cfg_get(config, "REGION", "CENTER", "HEIGHT_M"))
    parser.add_argument("--width-m", type=float, default=cfg_get(config, "REGION", "CENTER", "WIDTH_M"))

    parser.add_argument(
        "--api-key",
        type=str,
        default=resolve_api_key(config),
        help="Google Maps API key (env: GOOGLE_MAPS_API_KEY or GOOGLE_API_KEY)",
    )
    parser.add_argument("--zoom", type=int, default=cfg_get(config, "DOWNLOAD", "ZOOM", default=19))
    parser.add_argument(
        "--maptype",
        choices=["roadmap", "satellite", "terrain"],
        default=cfg_get(config, "DOWNLOAD", "MAPTYPE", default="satellite"),
    )
    parser.add_argument(
        "--download-mode",
        choices=["sequential", "parallel"],
        default=cfg_get(config, "DOWNLOAD", "MODE", default="parallel"),
    )
    parser.add_argument("--max-workers", type=int, default=cfg_get(config, "DOWNLOAD", "MAX_WORKERS"))
    parser.add_argument(
        "--xyz-image-format",
        choices=["jpeg", "png"],
        default=cfg_get(config, "GOOGLE_XYZ", "IMAGE_FORMAT", default="jpeg"),
        help="Google Map Tiles API tile image format",
    )
    parser.add_argument(
        "--xyz-scale",
        default=cfg_get(config, "GOOGLE_XYZ", "SCALE", default="scaleFactor1x"),
        help="Google Map Tiles API scale, such as scaleFactor1x or scaleFactor2x",
    )
    parser.add_argument(
        "--xyz-high-dpi",
        action="store_true",
        default=cfg_get(config, "GOOGLE_XYZ", "HIGH_DPI", default=False),
        help="Request high-DPI Google Map Tiles API sessions",
    )
    parser.add_argument(
        "--xyz-language",
        default=cfg_get(config, "GOOGLE_XYZ", "LANGUAGE", default="en-US"),
        help="Google Map Tiles API language",
    )
    parser.add_argument(
        "--xyz-region",
        default=cfg_get(config, "GOOGLE_XYZ", "REGION", default="US"),
        help="Google Map Tiles API region",
    )
    parser.add_argument(
        "--no-attribution-panel",
        action="store_true",
        help="Skip the bottom-left attribution panel",
    )
    parser.add_argument(
        "--logo-path",
        type=str,
        default=cfg_get(config, "WATERMARK", "LOGO_PATH", default=""),
        help="Optional Google Maps logo path for the bottom-left watermark panel",
    )
    parser.add_argument(
        "--logo-width-px",
        type=int,
        default=cfg_get(config, "WATERMARK", "LOGO_WIDTH_PX", default=105),
        help="Fixed Google Maps logo width for the bottom-left watermark panel",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        default=cfg_get(config, "NETWORK", "USE_ENV_PROXY", default=False),
        help="Allow requests to inherit system/environment proxy settings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg_get(config, "OUTPUT", "OUTPUT_DIR", default="."),
    )
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.backend != "xyz":
        print("Error: release builds only support the Google Map Tiles API xyz backend.", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print(
            "Error: Google Maps API key is required. Set GOOGLE_MAPS_API_KEY, "
            "config.yaml -> GOOGLE.API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.zoom < 0:
        print("Error: zoom must be >= 0.", file=sys.stderr)
        sys.exit(1)
    if args.logo_width_px <= 0:
        print("Error: --logo-width-px must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.type == "center":
        if None in (args.center_lat, args.center_lon, args.height_m, args.width_m):
            print(
                "Error: center mode requires --center-lat, --center-lon, --height-m, and --width-m.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.height_m <= 0 or args.width_m <= 0:
            print("Error: height and width must be positive.", file=sys.stderr)
            sys.exit(1)
        (lat1, lon1), (lat2, lon2) = center_to_corners(
            args.center_lat,
            args.center_lon,
            args.height_m,
            args.width_m,
        )
    else:
        if None in (args.lat1, args.lon1, args.lat2, args.lon2):
            print(
                "Error: corners mode requires --lat1, --lon1, --lat2, and --lon2.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.lat1 >= args.lat2:
            print("Error: lat1 must be less than lat2.", file=sys.stderr)
            sys.exit(1)
        if args.lon1 >= args.lon2:
            print("Error: lon1 must be less than lon2.", file=sys.stderr)
            sys.exit(1)
        lat1, lon1, lat2, lon2 = args.lat1, args.lon1, args.lat2, args.lon2

    center_lat, center_lon, height_m, width_m = corners_to_center(lat1, lon1, lat2, lon2)
    filename_pattern = cfg_get(
        config,
        "OUTPUT",
        "FILENAME_PATTERN_CENTER",
        default="google_xyz_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{zoom}.tif",
    )
    if args.output is None:
        output_filename = filename_pattern.format(
            center_lat=center_lat,
            center_lon=center_lon,
            height_m=height_m,
            width_m=width_m,
            zoom=args.zoom,
        )
        output_path = Path(args.output_dir) / output_filename
    else:
        output_path = Path(args.output)

    generate_xyz_geotiff(
        latlng1=(lat1, lon1),
        latlng2=(lat2, lon2),
        api_key=args.api_key,
        zoom=args.zoom,
        out_path=str(output_path),
        maptype=args.maptype,
        download_mode=args.download_mode,
        max_workers=args.max_workers,
        image_format=args.xyz_image_format,
        scale=args.xyz_scale,
        high_dpi=args.xyz_high_dpi,
        language=args.xyz_language,
        region=args.xyz_region,
        use_env_proxy=args.use_env_proxy,
        add_attribution_panel=not args.no_attribution_panel,
        logo_path=args.logo_path or None,
        logo_width_px=args.logo_width_px,
    )


if __name__ == "__main__":
    main()

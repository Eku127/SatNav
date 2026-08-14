#!/usr/bin/env python3
"""Generate GeoTIFF files from Mapbox raster tiles."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pyproj import Geod, Transformer
from rasterio.transform import from_origin

from .downloader import DownloadResult, MapboxTileDownloader, WEB_MERCATOR_LAT_LIMIT


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


def resolve_access_token(config: Dict[str, Any]) -> str:
    """Resolve the Mapbox access token from env, registry fallback, or config."""
    return (
        os.environ.get("MAPBOX_ACCESS_TOKEN")
        or read_windows_env_var("MAPBOX_ACCESS_TOKEN")
        or cfg_get(config, "MAPBOX", "ACCESS_TOKEN")
        or cfg_get(config, "API", "ACCESS_TOKEN")
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


def default_logo_path() -> Path:
    """Return the bundled default Mapbox logo asset path."""
    return Path(__file__).parent / "assets" / "mapbox_logo_black.png"


def resolve_logo_path(logo_path: Optional[str]) -> Optional[Path]:
    """Resolve a configured logo path, falling back to the bundled asset."""
    if logo_path:
        candidate = Path(logo_path).expanduser()
        return candidate if candidate.exists() else None

    bundled = default_logo_path()
    return bundled if bundled.exists() else None


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a legible sans-serif font, with a safe fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def crop_logo_margins(logo: Image.Image) -> Image.Image:
    """Trim empty or near-black margins from the bundled logo image."""
    rgba = logo.convert("RGBA")
    grayscale = ImageOps.grayscale(rgba)
    mask = grayscale.point(lambda value: 255 if value > 12 else 0)
    bbox = mask.getbbox()
    return rgba.crop(bbox) if bbox else rgba


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
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def add_mapbox_watermark(
    image: np.ndarray,
    result: DownloadResult,
    logo_path: Optional[str] = None,
    margin_px: int = 16,
    padding_px: int = 14,
    panel_opacity: int = 176,
    logo_width_px: int = 220,
    text_size_px: int = 14,
    max_width_ratio: float = 0.55,
) -> np.ndarray:
    """Add a bottom-left attribution panel to the final GeoTIFF image."""
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))

    logo = None
    resolved_logo_path = resolve_logo_path(logo_path)
    if resolved_logo_path is not None:
        logo = crop_logo_margins(Image.open(resolved_logo_path))
        target_logo_width = max(1, int(logo_width_px))
        resized_height = max(1, int(round(logo.height * (target_logo_width / logo.width))))
        logo = logo.resize((target_logo_width, resized_height), Image.Resampling.LANCZOS)

    text = result.tilejson.attribution_text or result.tilejson.attribution_html or "Attribution unavailable"
    font = load_font(text_size_px)
    max_panel_width = max(240, int(base.width * max_width_ratio))
    inner_max_width = max_panel_width - (padding_px * 2)
    lines = wrap_text_to_width(measure_draw, text, font, inner_max_width) or [text]

    line_gap = 4
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = line_bbox[3] - line_bbox[1]
    text_width = max(
        draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        for line in lines
    )
    text_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap

    logo_width = logo.width if logo is not None else 0
    logo_height = logo.height if logo is not None else 0
    spacer = 8 if logo is not None else 0

    content_width = max(logo_width, text_width)
    panel_width = min(base.width - (margin_px * 2), max(content_width + padding_px * 2, 240))
    panel_height = padding_px * 2 + logo_height + spacer + text_height

    left = max(0, margin_px)
    top = max(0, base.height - margin_px - panel_height)

    draw.rounded_rectangle(
        (left, top, left + panel_width, top + panel_height),
        radius=10,
        fill=(0, 0, 0, panel_opacity),
    )

    content_x = left + padding_px
    current_y = top + padding_px
    if logo is not None:
        overlay.alpha_composite(logo, dest=(content_x, current_y))
        current_y += logo_height + spacer

    for line in lines:
        draw.text((content_x, current_y), line, fill=(255, 255, 255, 255), font=font)
        current_y += line_height + line_gap

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(composited)


def save_geotiff(
    image: np.ndarray,
    result: DownloadResult,
    out_path: Path,
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Save the stitched RGB image as a georeferenced GeoTIFF in EPSG:3857."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    lon_min, lat_min, lon_max, lat_max = result.bbox_wgs84
    height, width, bands = image.shape

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height
    transform = from_origin(x_min, y_max, pixel_size_x, pixel_size_y)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=image.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        for band_index in range(3):
            dst.write(image[:, :, band_index], band_index + 1)

        tags = {
            "source": "Mapbox Raster Tiles API",
            "tileset_id": result.tilejson.tileset_id,
            "zoom": str(extra_tags.get("zoom", "")) if extra_tags else "",
            "bbox_wgs84": f"{lon_min:.8f},{lat_min:.8f},{lon_max:.8f},{lat_max:.8f}",
            "attribution": result.tilejson.attribution_text or result.tilejson.attribution_html,
        }
        if extra_tags:
            tags.update({key: value for key, value in extra_tags.items() if value})
        dst.update_tags(**{key: value for key, value in tags.items() if value})


def default_attribution_path(out_path: Path) -> Path:
    """Return a sidecar attribution path next to the GeoTIFF."""
    return out_path.with_suffix(".attribution.txt")


def write_attribution_file(
    out_path: Path,
    result: DownloadResult,
    zoom: int,
    scale: int,
    image_format: str,
    logo_width_px: int,
) -> Path:
    """Write a sidecar text file with attribution and provenance metadata."""
    target = default_attribution_path(out_path)
    lon_min, lat_min, lon_max, lat_max = result.bbox_wgs84
    window = result.tile_window

    content = "\n".join(
        [
            "Mapbox imagery attribution and provenance",
            "",
            f"GeoTIFF: {out_path}",
            f"Tileset: {result.tilejson.tileset_id}",
            f"Zoom: {zoom}",
            f"Scale: @{scale}x" if scale > 1 else "Scale: 1x",
            f"Raster format: {image_format}",
            f"Watermark logo width: {logo_width_px}px",
            f"BBox (lon_min, lat_min, lon_max, lat_max): {lon_min:.8f}, {lat_min:.8f}, {lon_max:.8f}, {lat_max:.8f}",
            f"Output pixels: {window.width_px} x {window.height_px}",
            f"Tile coverage: x={window.left}..{window.right}, y={window.top}..{window.bottom}",
            "",
            "Attribution:",
            result.tilejson.attribution_text or result.tilejson.attribution_html or "No attribution returned by TileJSON.",
            "",
            "Note:",
            "If you publish or redistribute this imagery, make sure the required attribution is shown in your final application or document.",
        ]
    )

    target.write_text(content, encoding="utf-8")
    return target


def generate_geotiff(
    latlng1: Tuple[float, float],
    latlng2: Tuple[float, float],
    access_token: str,
    zoom: int = 19,
    out_path: str = "output.tif",
    tileset_id: str = "mapbox.satellite",
    image_format: str = "jpg90",
    scale: int = 1,
    download_mode: str = "parallel",
    max_workers: Optional[int] = None,
    use_env_proxy: bool = False,
    write_attribution_sidecar: bool = True,
    add_output_watermark: bool = True,
    logo_path: Optional[str] = None,
    logo_width_px: int = 220,
) -> Path:
    """Download a Mapbox satellite mosaic and save it as GeoTIFF."""
    lat1, lon1 = latlng1
    lat2, lon2 = latlng2
    center_lat, center_lon, height_m, width_m = corners_to_center(lat1, lon1, lat2, lon2)

    print("=" * 60)
    print("Region Information:")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Dimensions: {height_m:.2f}m (height) x {width_m:.2f}m (width)")
    print(f"  Zoom level: {zoom}")
    print(f"  Tileset: {tileset_id}")
    print(f"  Tile scale: @{scale}x" if scale > 1 else "  Tile scale: 1x")
    print("=" * 60)

    args = argparse.Namespace(
        access_token=access_token,
        zoom=zoom,
        tileset_id=tileset_id,
        image_format=image_format,
        scale=scale,
        download_mode=download_mode,
        max_workers=max_workers,
        use_env_proxy=use_env_proxy,
    )
    downloader = MapboxTileDownloader(args)
    tilejson = downloader.fetch_tilejson()

    print("\nTileset Metadata:")
    print(f"  Name: {tilejson.name or tilejson.tileset_id}")
    if tilejson.minzoom is not None:
        print(f"  Min zoom: {tilejson.minzoom}")
    if tilejson.maxzoom is not None:
        print(f"  Max zoom: {tilejson.maxzoom}")
        if zoom > tilejson.maxzoom:
            print("  Warning: requested zoom is above TileJSON maxzoom; the server may overzoom imagery.")
    if tilejson.attribution_text:
        print(f"  Attribution: {to_console_text(tilejson.attribution_text)}")

    print("\nStarting download...\n")
    result = downloader.download_bbox(latlng1, latlng2)

    print(f"\nDownload completed: {result.image.shape[1]} x {result.image.shape[0]} pixels")
    print(
        "Tile coverage: "
        f"{result.tile_window.cols} x {result.tile_window.rows} = {result.tile_window.total_tiles} tiles"
    )
    print("Generating GeoTIFF...")

    output_path = Path(out_path)
    final_image = result.image
    resolved_logo = resolve_logo_path(logo_path)
    watermark_mode = "disabled"
    if add_output_watermark:
        final_image = add_mapbox_watermark(
            image=result.image,
            result=result,
            logo_path=logo_path,
            logo_width_px=logo_width_px,
        )
        watermark_mode = "logo+text" if resolved_logo is not None else "text-only"

    extra_tags = {
        "zoom": str(zoom),
        "scale": str(scale),
        "image_format": image_format,
        "tile_window": (
            f"{result.tile_window.left},{result.tile_window.top},"
            f"{result.tile_window.right},{result.tile_window.bottom}"
        ),
        "watermark_mode": watermark_mode,
        "watermark_position": "bottom-left" if add_output_watermark else "",
        "watermark_logo_path": str(resolved_logo) if resolved_logo is not None else "",
        "watermark_logo_width_px": str(logo_width_px) if add_output_watermark else "",
    }
    save_geotiff(final_image, result, output_path, extra_tags=extra_tags)

    print("\nGeoTIFF saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {final_image.shape[1]} x {final_image.shape[0]} pixels")
    print("  CRS: EPSG:3857 (Web Mercator)")
    if add_output_watermark:
        if resolved_logo is not None:
            print(
                "  Watermark panel: bottom-left "
                f"(official logo + attribution text, logo width={logo_width_px}px)"
            )
        else:
            print("  Watermark panel: bottom-left (attribution text only; no logo asset found)")

    if write_attribution_sidecar:
        attribution_path = write_attribution_file(
            out_path=output_path,
            result=result,
            zoom=zoom,
            scale=scale,
            image_format=image_format,
            logo_width_px=logo_width_px,
        )
        print(f"  Attribution file: {attribution_path}")

    return output_path


def main() -> None:
    """Command-line entry point."""
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "config.yaml")

    region_type = cfg_get(config, "REGION", "TYPE", default="corners")

    parser = argparse.ArgumentParser(
        description="Generate GeoTIFF files from Mapbox raster tiles."
    )
    parser.add_argument(
        "--type",
        choices=["corners", "center"],
        default=region_type,
        help=f"Region definition type (default: {region_type})",
    )

    parser.add_argument(
        "--lat1",
        type=float,
        default=cfg_get(config, "REGION", "CORNERS", "LAT1"),
        help="Lower-left latitude",
    )
    parser.add_argument(
        "--lon1",
        type=float,
        default=cfg_get(config, "REGION", "CORNERS", "LON1"),
        help="Lower-left longitude",
    )
    parser.add_argument(
        "--lat2",
        type=float,
        default=cfg_get(config, "REGION", "CORNERS", "LAT2"),
        help="Upper-right latitude",
    )
    parser.add_argument(
        "--lon2",
        type=float,
        default=cfg_get(config, "REGION", "CORNERS", "LON2"),
        help="Upper-right longitude",
    )

    parser.add_argument(
        "--center-lat",
        type=float,
        default=cfg_get(config, "REGION", "CENTER", "LAT"),
        help="Center latitude",
    )
    parser.add_argument(
        "--center-lon",
        type=float,
        default=cfg_get(config, "REGION", "CENTER", "LON"),
        help="Center longitude",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=cfg_get(config, "REGION", "CENTER", "HEIGHT_M"),
        help="North-south size in meters",
    )
    parser.add_argument(
        "--width-m",
        type=float,
        default=cfg_get(config, "REGION", "CENTER", "WIDTH_M"),
        help="East-west size in meters",
    )

    access_token_default = resolve_access_token(config)
    parser.add_argument(
        "--access-token",
        "--api-key",
        dest="access_token",
        type=str,
        default=access_token_default,
        help="Mapbox access token (env: MAPBOX_ACCESS_TOKEN)",
    )

    parser.add_argument(
        "--tileset-id",
        type=str,
        default=cfg_get(config, "MAPBOX", "TILESET_ID", default="mapbox.satellite"),
        help="Mapbox raster tileset id",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=cfg_get(config, "DOWNLOAD", "ZOOM", default=19),
        help="Zoom level",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        type=str,
        default=cfg_get(config, "DOWNLOAD", "FORMAT", default="jpg90"),
        help="Raster tile format (jpg, jpg80, jpg90, webp, png, ...)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[1, 2],
        default=cfg_get(config, "DOWNLOAD", "SCALE", default=1),
        help="Tile scale factor: 1 or 2. Use 2 only when @2x output is needed.",
    )
    parser.add_argument(
        "--download-mode",
        choices=["sequential", "parallel"],
        default=cfg_get(config, "DOWNLOAD", "MODE", default="parallel"),
        help="Download mode",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=cfg_get(config, "DOWNLOAD", "MAX_WORKERS"),
        help="Maximum worker count for parallel download",
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
        help="Output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output GeoTIFF path",
    )
    parser.add_argument(
        "--no-attribution-file",
        action="store_true",
        help="Do not write the sidecar attribution text file",
    )
    parser.add_argument(
        "--logo-path",
        type=str,
        default=cfg_get(config, "WATERMARK", "LOGO_PATH", default=""),
        help="Optional official Mapbox logo asset path",
    )
    parser.add_argument(
        "--logo-width-px",
        type=int,
        default=cfg_get(config, "WATERMARK", "LOGO_WIDTH_PX", default=220),
        help="Fixed logo width for the bottom-left watermark panel",
    )
    parser.add_argument(
        "--no-watermark",
        action="store_true",
        help="Skip the bottom-left Mapbox attribution panel on the final TIFF",
    )

    args = parser.parse_args()

    if not args.access_token:
        print(
            "Error: Mapbox access token is required. Set MAPBOX_ACCESS_TOKEN, "
            "config.yaml -> MAPBOX.ACCESS_TOKEN, or pass --access-token.",
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

    if not (-WEB_MERCATOR_LAT_LIMIT <= lat1 <= WEB_MERCATOR_LAT_LIMIT):
        print(
            f"Error: lat1 must be within [-{WEB_MERCATOR_LAT_LIMIT}, {WEB_MERCATOR_LAT_LIMIT}].",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (-WEB_MERCATOR_LAT_LIMIT <= lat2 <= WEB_MERCATOR_LAT_LIMIT):
        print(
            f"Error: lat2 must be within [-{WEB_MERCATOR_LAT_LIMIT}, {WEB_MERCATOR_LAT_LIMIT}].",
            file=sys.stderr,
        )
        sys.exit(1)

    center_lat, center_lon, height_m, width_m = corners_to_center(lat1, lon1, lat2, lon2)
    tileset_slug = args.tileset_id.replace(".", "_")

    filename_pattern = cfg_get(
        config,
        "OUTPUT",
        "FILENAME_PATTERN_CENTER",
        default="mapbox_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{zoom}.tif",
    )
    if args.output is None:
        try:
            output_filename = filename_pattern.format(
                center_lat=center_lat,
                center_lon=center_lon,
                height_m=height_m,
                width_m=width_m,
                zoom=args.zoom,
                tileset_id=tileset_slug,
            )
        except KeyError as exc:
            print(f"Error: invalid filename pattern placeholder: {exc}", file=sys.stderr)
            sys.exit(1)

        out_path = Path(args.output_dir) / output_filename
    else:
        out_path = Path(args.output)

    generate_geotiff(
        latlng1=(lat1, lon1),
        latlng2=(lat2, lon2),
        access_token=args.access_token,
        zoom=args.zoom,
        out_path=str(out_path),
        tileset_id=args.tileset_id,
        image_format=args.image_format,
        scale=args.scale,
        download_mode=args.download_mode,
        max_workers=args.max_workers,
        use_env_proxy=args.use_env_proxy,
        write_attribution_sidecar=not args.no_attribution_file,
        add_output_watermark=not args.no_watermark and cfg_get(config, "WATERMARK", "ENABLED", default=True),
        logo_path=args.logo_path or None,
        logo_width_px=args.logo_width_px,
    )


if __name__ == "__main__":
    main()

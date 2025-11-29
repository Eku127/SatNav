#!/usr/bin/env python3
"""Generate GeoTIFF files from Google Maps satellite imagery."""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from omegaconf import OmegaConf
from pyproj import Geod, Transformer
from rasterio.transform import from_origin

from .downloader import GoogleMapDownloader


def center_to_corners(
    center_lat: float,
    center_lon: float,
    height_m: float,
    width_m: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate corner coordinates from center point and dimensions.
    
    This function calculates all four corners of a rectangular region centered
    at the given point, then returns the bounding box (lower-left and upper-right).
    
    Args:
        center_lat: Center latitude (degrees).
        center_lon: Center longitude (degrees).
        height_m: North-South dimension in meters.
        width_m: East-West dimension in meters.
    
    Returns:
        Tuple of ((lat1, lon1), (lat2, lon2)) where:
        - (lat1, lon1) is the lower-left corner (minimum lat, minimum lon)
        - (lat2, lon2) is the upper-right corner (maximum lat, maximum lon)
    """
    geod = Geod(ellps="WGS84")
    
    # Calculate half dimensions
    half_height = height_m / 2.0
    half_width = width_m / 2.0
    
    # Calculate all four corners from center
    # Directions: 0=North, 90=East, 180=South, 270=West
    
    # North edge (move North from center)
    lon_n, lat_n, _ = geod.fwd(center_lon, center_lat, 0, half_height)
    # South edge (move South from center)
    lon_s, lat_s, _ = geod.fwd(center_lon, center_lat, 180, half_height)
    
    # Calculate four corners
    # Northwest corner: from North edge, move West
    lon_nw, lat_nw, _ = geod.fwd(lon_n, lat_n, 270, half_width)
    # Northeast corner: from North edge, move East
    lon_ne, lat_ne, _ = geod.fwd(lon_n, lat_n, 90, half_width)
    # Southwest corner: from South edge, move West
    lon_sw, lat_sw, _ = geod.fwd(lon_s, lat_s, 270, half_width)
    # Southeast corner: from South edge, move East
    lon_se, lat_se, _ = geod.fwd(lon_s, lat_s, 90, half_width)
    
    # Find bounding box (min/max lat and lon)
    lats = [lat_nw, lat_ne, lat_sw, lat_se]
    lons = [lon_nw, lon_ne, lon_sw, lon_se]
    
    lat_min = min(lats)
    lat_max = max(lats)
    lon_min = min(lons)
    lon_max = max(lons)
    
    return ((lat_min, lon_min), (lat_max, lon_max))


def corners_to_center(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> Tuple[float, float, float, float]:
    """
    Calculate center point and dimensions from corner coordinates.
    
    This is the inverse of center_to_corners. It calculates the center point
    and dimensions (height_m, width_m) from two corner coordinates.
    
    Args:
        lat1: Lower-left latitude (degrees).
        lon1: Lower-left longitude (degrees).
        lat2: Upper-right latitude (degrees).
        lon2: Upper-right longitude (degrees).
    
    Returns:
        Tuple of (center_lat, center_lon, height_m, width_m) where:
        - center_lat: Center latitude
        - center_lon: Center longitude
        - height_m: North-South dimension in meters
        - width_m: East-West dimension in meters
    """
    geod = Geod(ellps="WGS84")
    
    # Calculate center point (simple average, good approximation for small regions)
    center_lat = (lat1 + lat2) / 2.0
    center_lon = (lon1 + lon2) / 2.0
    
    # Calculate dimensions using geodesic distance
    # Width: distance along latitude (East-West)
    _, _, width_m = geod.inv(lon1, lat1, lon2, lat1)
    # Height: distance along longitude (North-South)
    _, _, height_m = geod.inv(lon1, lat1, lon1, lat2)
    
    return (center_lat, center_lon, abs(height_m), abs(width_m))


def generate_geotiff(
    latlng1: Tuple[float, float] = (39.900, 116.397),
    latlng2: Tuple[float, float] = (39.903, 116.400),
    api_key: str = "",
    zoom: int = 19,
    out_path: str = "output.tif",
    signature: Optional[str] = None,
    download_mode: str = "sequential",
    max_workers: Optional[int] = None
) -> None:
    """
    Generate a GeoTIFF file for a rectangular region defined by two lat/lon coordinates.
    
    This function downloads satellite imagery from Google Maps Static API for the specified
    region and saves it as a GeoTIFF file in EPSG:3857 (Web Mercator) projection.
    
    Args:
        latlng1 (tuple): Lower-left corner (lat, lon).
        latlng2 (tuple): Upper-right corner (lat, lon).
        api_key (str): Google Maps API key.
        zoom (int): Zoom level (higher = more detail, typically 15-20).
        out_path (str): Output file path.
        signature (str, optional): Optional signature for API requests.
    
    Example:
        >>> generate_geotiff(
        ...     latlng1=(22.54, 114.06),
        ...     latlng2=(22.55, 114.07),
        ...     api_key="YOUR_API_KEY",
        ...     zoom=19,
        ...     out_path="hongkong.tif"
        ... )
    """
    def rectangle_info(lon1: float, lat1: float, lon2: float, lat2: float) -> Tuple[float, float, float, float]:
        """Calculate rectangle center and dimensions."""
        geod = Geod(ellps="WGS84")
        mid_lon = (lon1 + lon2) / 2
        mid_lat = (lat1 + lat2) / 2
        _, _, w_length = geod.inv(lon1, lat1, lon2, lat1)
        _, _, h_length = geod.inv(lon1, lat1, lon1, lat2)
        return mid_lon, mid_lat, h_length, w_length
    
    lat1, lon1 = latlng1
    lat2, lon2 = latlng2
    mid_lon, mid_lat, h_length, w_length = rectangle_info(lon1, lat1, lon2, lat2)

    # Print region information
    print("=" * 60)
    print("Region Information:")
    print(f"  Center: ({mid_lat:.6f}, {mid_lon:.6f})")
    print(f"  Dimensions: {abs(h_length):.2f}m (height) × {abs(w_length):.2f}m (width)")
    print(f"  Zoom level: {zoom}")
    print("=" * 60)

    args = argparse.Namespace(
        api_key=api_key,
        signature=signature,
        zoom=zoom,
        x=mid_lon,
        y=mid_lat,
        download_mode=download_mode,
        max_workers=max_workers
    )

    downloader = GoogleMapDownloader(args)
    
    # Calculate pixel dimensions
    pixel_width = downloader.length_to_pixels(abs(w_length))
    pixel_height = downloader.length_to_pixels(abs(h_length))
    
    # Calculate number of tiles needed
    tile_w, tile_h = 640, 600
    cols = math.ceil(pixel_width / tile_w)
    rows = math.ceil(pixel_height / tile_h)
    total_tiles = cols * rows
    
    print(f"\nImage Dimensions:")
    print(f"  Pixels: {pixel_width} × {pixel_height}")
    print(f"  Tiles needed: {cols} × {rows} = {total_tiles} tiles")
    print(f"\nStarting download...\n")
    
    image_np = downloader.get_map(
        size=(pixel_width, pixel_height),
        maptype="satellite"
    )
    
    print(f"\n✓ Download completed! ({image_np.shape[1]} × {image_np.shape[0]} pixels)")
    print("Generating GeoTIFF file...")

    def save_geotiff(
        img_np: np.ndarray,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        out_path: str = "output.tif"
    ) -> None:
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

        print(f"\n✓ GeoTIFF saved successfully!")
        print(f"  File: {out_path}")
        print(f"  Size: {width} × {height} pixels")
        print(f"  CRS: EPSG:3857 (Web Mercator)")

    save_geotiff(image_np, lon1, lat1, lon2, lat2, out_path=out_path)


def main():
    """Command-line entry point for generate_geotiff."""
    # Find config.yaml relative to this file
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    
    # Load config if it exists
    config = None
    if config_path.exists():
        config = OmegaConf.load(config_path)
    
    # Determine region type from config
    region_type = "corners"  # default
    if config and hasattr(config, "REGION") and hasattr(config.REGION, "TYPE"):
        region_type = config.REGION.TYPE
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate GeoTIFF files from Google Maps satellite imagery."
    )
    
    # Region type argument
    parser.add_argument(
        "--type", type=str, choices=["corners", "center"], default=region_type,
        help=f"Region definition type: 'corners' (two corner points) or 'center' (center point + dimensions) (default: {region_type} from config)"
    )
    
    # Corner-based arguments (for TYPE="corners")
    if config and hasattr(config, "REGION") and hasattr(config.REGION, "CORNERS"):
        lat1_default = config.REGION.CORNERS.LAT1
        lon1_default = config.REGION.CORNERS.LON1
        lat2_default = config.REGION.CORNERS.LAT2
        lon2_default = config.REGION.CORNERS.LON2
    else:
        lat1_default = None
        lon1_default = None
        lat2_default = None
        lon2_default = None
    
    parser.add_argument(
        "--lat1", type=float, default=lat1_default,
        help=f"Lower-left latitude (degrees) (default: {lat1_default} from config)" if lat1_default is not None else "Lower-left latitude (degrees)"
    )
    parser.add_argument(
        "--lon1", type=float, default=lon1_default,
        help=f"Lower-left longitude (degrees) (default: {lon1_default} from config)" if lon1_default is not None else "Lower-left longitude (degrees)"
    )
    parser.add_argument(
        "--lat2", type=float, default=lat2_default,
        help=f"Upper-right latitude (degrees) (default: {lat2_default} from config)" if lat2_default is not None else "Upper-right latitude (degrees)"
    )
    parser.add_argument(
        "--lon2", type=float, default=lon2_default,
        help=f"Upper-right longitude (degrees) (default: {lon2_default} from config)" if lon2_default is not None else "Upper-right longitude (degrees)"
    )
    
    # Center-based arguments (for TYPE="center")
    if config and hasattr(config, "REGION") and hasattr(config.REGION, "CENTER"):
        center_lat_default = config.REGION.CENTER.LAT
        center_lon_default = config.REGION.CENTER.LON
        height_m_default = config.REGION.CENTER.HEIGHT_M
        width_m_default = config.REGION.CENTER.WIDTH_M
    else:
        center_lat_default = None
        center_lon_default = None
        height_m_default = None
        width_m_default = None
    
    parser.add_argument(
        "--center-lat", type=float, default=center_lat_default,
        help=f"Center latitude (degrees) (default: {center_lat_default} from config)" if center_lat_default is not None else "Center latitude (degrees)"
    )
    parser.add_argument(
        "--center-lon", type=float, default=center_lon_default,
        help=f"Center longitude (degrees) (default: {center_lon_default} from config)" if center_lon_default is not None else "Center longitude (degrees)"
    )
    parser.add_argument(
        "--height-m", type=float, default=height_m_default,
        help=f"North-South dimension in meters (default: {height_m_default} from config)" if height_m_default is not None else "North-South dimension in meters"
    )
    parser.add_argument(
        "--width-m", type=float, default=width_m_default,
        help=f"East-West dimension in meters (default: {width_m_default} from config)" if width_m_default is not None else "East-West dimension in meters"
    )
    
    # Optional arguments with config defaults
    api_key_default = config.API.API_KEY if config and hasattr(config, "API") else ""
    parser.add_argument(
        "--api-key", type=str, default=api_key_default,
        help=f"Google Maps API key (default: from config or empty)"
    )
    
    zoom_default = config.DOWNLOAD.ZOOM if config and hasattr(config, "DOWNLOAD") else 19
    parser.add_argument(
        "--zoom", type=int, default=zoom_default,
        help=f"Zoom level (default: {zoom_default} from config)"
    )
    
    signature_default = config.API.SIGNATURE if config and hasattr(config, "API") and config.API.SIGNATURE else None
    parser.add_argument(
        "--signature", type=str, default=signature_default,
        help="Optional signature for API requests (default: from config or None)"
    )
    
    # Download mode arguments
    download_mode_default = config.DOWNLOAD.MODE if config and hasattr(config, "DOWNLOAD") and hasattr(config.DOWNLOAD, "MODE") else "sequential"
    parser.add_argument(
        "--download-mode", type=str, choices=["sequential", "parallel"], default=download_mode_default,
        help=f"Download mode: 'sequential' or 'parallel' (default: {download_mode_default} from config)"
    )
    
    max_workers_default = config.DOWNLOAD.MAX_WORKERS if config and hasattr(config, "DOWNLOAD") and hasattr(config.DOWNLOAD, "MAX_WORKERS") else None
    parser.add_argument(
        "--max-workers", type=int, default=max_workers_default,
        help=f"Maximum concurrent workers for parallel download (default: {max_workers_default} from config or auto)"
    )
    
    # Output path handling
    output_dir_default = config.OUTPUT.OUTPUT_DIR if config and hasattr(config, "OUTPUT") else "."
    parser.add_argument(
        "--output-dir", type=str, default=output_dir_default,
        help=f"Output directory (default: {output_dir_default} from config)"
    )
    
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (default: auto-generated from coordinates and zoom)"
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: API key is required. Set it in config.yaml or use --api-key argument.", file=sys.stderr)
        sys.exit(1)
    
    # Determine coordinates based on type
    if args.type == "center":
        # Center-based definition
        if args.center_lat is None or args.center_lon is None or args.height_m is None or args.width_m is None:
            print("Error: Center-based definition requires --center-lat, --center-lon, --height-m, and --width-m.", file=sys.stderr)
            print("Set them in config.yaml (REGION.CENTER section) or use command-line arguments.", file=sys.stderr)
            sys.exit(1)
        
        if args.height_m <= 0 or args.width_m <= 0:
            print(f"Error: Height ({args.height_m}) and width ({args.width_m}) must be positive.", file=sys.stderr)
            sys.exit(1)
        
        # Convert center + dimensions to corners
        (lat1, lon1), (lat2, lon2) = center_to_corners(
            args.center_lat, args.center_lon, args.height_m, args.width_m
        )
    else:
        # Corner-based definition
        if args.lat1 is None or args.lon1 is None or args.lat2 is None or args.lon2 is None:
            print("Error: Corner-based definition requires --lat1, --lon1, --lat2, and --lon2.", file=sys.stderr)
            print("Set them in config.yaml (REGION.CORNERS section) or use command-line arguments.", file=sys.stderr)
            sys.exit(1)
        
        # Validate coordinate order
        if args.lat1 >= args.lat2:
            print(f"Error: lat1 ({args.lat1}) must be less than lat2 ({args.lat2}).", file=sys.stderr)
            sys.exit(1)
        if args.lon1 >= args.lon2:
            print(f"Error: lon1 ({args.lon1}) must be less than lon2 ({args.lon2}).", file=sys.stderr)
            sys.exit(1)
        
        lat1, lon1 = args.lat1, args.lon1
        lat2, lon2 = args.lat2, args.lon2
    
    # Calculate center information for filename (always use center format)
    if args.type == "center":
        # Already have center info from args
        center_lat = args.center_lat
        center_lon = args.center_lon
        height_m = args.height_m
        width_m = args.width_m
    else:
        # Convert corners to center info
        center_lat, center_lon, height_m, width_m = corners_to_center(
            lat1, lon1, lat2, lon2
        )
    
    # Generate output filename if not provided (always use center format)
    if args.output is None:
        if config and hasattr(config, "OUTPUT"):
            # Always use center-based pattern
            if hasattr(config.OUTPUT, "FILENAME_PATTERN_CENTER"):
                filename_pattern = config.OUTPUT.FILENAME_PATTERN_CENTER
            else:
                # Fall back to default center pattern
                filename_pattern = "map_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{zoom}.tif"
            
            output_filename = filename_pattern.format(
                center_lat=center_lat,
                center_lon=center_lon,
                height_m=height_m,
                width_m=width_m,
                zoom=args.zoom
            )
        else:
            # Default filename using center format
            output_filename = f"map_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{args.zoom}.tif"
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(output_dir / output_filename)
    else:
        out_path = args.output
    
    # Call generate_geotiff
    generate_geotiff(
        latlng1=(lat1, lon1),
        latlng2=(lat2, lon2),
        api_key=args.api_key,
        zoom=args.zoom,
        out_path=out_path,
        signature=args.signature if args.signature else None,
        download_mode=args.download_mode,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()


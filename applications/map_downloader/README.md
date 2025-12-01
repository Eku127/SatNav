# Map Downloader

Map downloader application for generating GeoTIFF files from Google Maps Static API.

## Overview

This application provides tools to download satellite imagery from Google Maps and save it as GeoTIFF files that can be used with SatNav simulator. The downloaded maps are automatically georeferenced in EPSG:3857 (Web Mercator) projection.

## Features

- Download satellite imagery from Google Maps Static API
- Automatic georeferencing (EPSG:3857)
- Support for large images (automatic tiling and stitching)
- Configurable zoom levels and map types
- **Two region definition methods**: corners-based or center-based
- **Unified filename format**: All outputs use center format for consistency

## Requirements

- Google Maps API key (get one from [Google Cloud Console](https://developers.google.com/maps/documentation/maps-static/get-api-key))
- Python packages: `rasterio`, `pyproj`, `requests`, `PIL`, `numpy`

## Installation

The map downloader is part of the SatNav applications. Install SatNav dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface (Recommended)

The command-line interface automatically reads configuration from `config.yaml` and allows you to override any parameter.

#### Method 1: Using Config Defaults (Easiest)

Simply edit `config.yaml` with your desired region and run:

```bash
python -m applications.map_downloader
```

All parameters (coordinates, API key, zoom, output directory) will be read from `config.yaml`.

#### Method 2: Corner-based Definition

Define region by two corner coordinates:

```bash
python -m applications.map_downloader \
    --type corners \
    --lat1 22.54 --lon1 114.06 \
    --lat2 22.55 --lon2 114.07 \
    --api-key YOUR_API_KEY \
    --zoom 19
```

Or use config defaults:

```yaml
# config.yaml
REGION:
  TYPE: "corners"
  CORNERS:
    LAT1: 22.54
    LON1: 114.06
    LAT2: 22.55
    LON2: 114.07
```

Then run:
```bash
python -m applications.map_downloader
```

#### Method 3: Center-based Definition

Define region by center point and dimensions:

```bash
python -m applications.map_downloader \
    --type center \
    --center-lat 22.545 \
    --center-lon 114.065 \
    --height-m 1000 \
    --width-m 1000 \
    --api-key YOUR_API_KEY \
    --zoom 19
```

Or use config defaults:

```yaml
# config.yaml
REGION:
  TYPE: "center"
  CENTER:
    LAT: 22.545
    LON: 114.065
    HEIGHT_M: 1000.0  # meters
    WIDTH_M: 1000.0   # meters
```

Then run:
```bash
python -m applications.map_downloader
```

#### Command-line Arguments

**Region Type:**
- `--type {corners,center}`: Region definition type (default: from config)

**Corner-based Arguments:**
- `--lat1 FLOAT`: Lower-left latitude (degrees)
- `--lon1 FLOAT`: Lower-left longitude (degrees)
- `--lat2 FLOAT`: Upper-right latitude (degrees)
- `--lon2 FLOAT`: Upper-right longitude (degrees)

**Center-based Arguments:**
- `--center-lat FLOAT`: Center latitude (degrees)
- `--center-lon FLOAT`: Center longitude (degrees)
- `--height-m FLOAT`: North-South dimension (meters)
- `--width-m FLOAT`: East-West dimension (meters)

**Other Arguments:**
- `--api-key STR`: Google Maps API key (default: from config)
- `--zoom INT`: Zoom level (default: from config)
- `--signature STR`: Optional API signature (default: from config)
- `--download-mode {sequential,parallel}`: Download mode (default: from config)
- `--max-workers INT`: Max concurrent workers for parallel mode (default: from config or auto)
- `--output-dir STR`: Output directory (default: from config)
- `--output STR`: Output filename (default: auto-generated)

### Python API

```python
from applications.map_downloader import generate_geotiff

# Generate GeoTIFF using corner coordinates
generate_geotiff(
    latlng1=(22.54, 114.06),  # Lower-left corner (lat, lon)
    latlng2=(22.55, 114.07),  # Upper-right corner (lat, lon)
    api_key="YOUR_API_KEY",
    zoom=19,
    out_path="hongkong.tif"
)
```

## Configuration

Edit `config.yaml` to set default parameters. All parameters can be overridden via command-line arguments.

### API Configuration

```yaml
API:
  API_KEY: "YOUR_API_KEY"  # Required: Google Maps API key
  SIGNATURE: null          # Optional: API signature for signed URLs
```

### Region Configuration

Choose one of two methods to define the region:

#### Method 1: Corner-based (TYPE="corners")

```yaml
REGION:
  TYPE: "corners"
  CORNERS:
    LAT1: 39.900   # Lower-left latitude
    LON1: 116.397  # Lower-left longitude
    LAT2: 39.903   # Upper-right latitude
    LON2: 116.400  # Upper-right longitude
```

#### Method 2: Center-based (TYPE="center")

```yaml
REGION:
  TYPE: "center"
  CENTER:
    LAT: 39.9015      # Center latitude
    LON: 116.3985     # Center longitude
    HEIGHT_M: 300.0   # North-South dimension (meters)
    WIDTH_M: 300.0    # East-West dimension (meters)
```

### Download Parameters

```yaml
DOWNLOAD:
  ZOOM: 19              # Zoom level (15-20, higher = more detail)
  MAPTYPE: "satellite"  # Map type: satellite, roadmap, hybrid, terrain
  
  # Download mode: "sequential" or "parallel"
  # - "sequential": Download tiles one by one (slower but more stable)
  # - "parallel": Download multiple tiles concurrently (faster, 4-9x speedup)
  MODE: "parallel"
  
  # Maximum concurrent workers for parallel download
  # Only used when MODE="parallel"
  # Default: min(CPU cores, 20) to avoid API rate limits
  # Recommended: 10-20 for most cases
  MAX_WORKERS: 15
```

### Output Configuration

```yaml
OUTPUT:
  OUTPUT_DIR: "data/scene_datasets"  # Default output directory
  
  # Filename pattern (always uses center format)
  # Placeholders: {center_lat}, {center_lon}, {height_m}, {width_m}, {zoom}
  FILENAME_PATTERN_CENTER: "map_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{zoom}.tif"
```

**Note:** Regardless of input type (corners or center), filenames always use center format. When using corner-based input, corners are automatically converted to center format for the filename.

## Parameters

### Zoom Level

Zoom level determines the resolution of the downloaded map:

- **15**: ~5 meters per pixel (city-level overview)
- **17**: ~1.2 meters per pixel (neighborhood level)
- **19**: ~0.3 meters per pixel (building level, recommended)
- **20**: ~0.15 meters per pixel (very high detail)

### Map Types

- `satellite`: Satellite imagery (recommended for SatNav)
- `roadmap`: Road map
- `hybrid`: Satellite imagery with road labels
- `terrain`: Terrain map

### Download Modes

The downloader supports two modes for fetching map tiles:

#### Sequential Mode (`MODE: "sequential"`)

- Downloads tiles one by one in order
- **Pros**: More stable, lower memory usage, respects API rate limits
- **Cons**: Slower (takes ~45 seconds for 132 tiles)
- **Use when**: Network is unstable, API has strict rate limits, or downloading small regions

#### Parallel Mode (`MODE: "parallel"`)

- Downloads multiple tiles concurrently using thread pool
- **Pros**: Much faster (4-9x speedup, ~5-10 seconds for 132 tiles)
- **Cons**: Higher memory usage, may hit API rate limits if `MAX_WORKERS` is too high
- **Use when**: Downloading large regions, stable network connection
- **Configuration**: Set `MAX_WORKERS` (default: min(CPU cores, 20))
  - Recommended: 10-20 workers for most cases
  - Lower (5-10) if hitting API rate limits
  - Higher (20-30) only if you have high API quota

**Performance Comparison:**

| Tiles | Sequential | Parallel (15 workers) | Speedup |
|-------|-----------|----------------------|---------|
| 49 tiles (1km×1km) | ~20s | ~3-5s | 4-7x |
| 132 tiles (2km×2km) | ~45s | ~5-10s | 4-9x |
| 400 tiles (4km×4km) | ~150s | ~15-25s | 6-10x |

## Coordinate System

- **Input**: WGS84 (EPSG:4326) - latitude/longitude in degrees
- **Output**: Web Mercator (EPSG:3857) - meters

The output GeoTIFF files are automatically georeferenced and can be directly used with SatNav simulator.

## API Limits

Google Maps Static API has usage limits:

- **Free tier**: 28,000 map loads per month
- **Paid tier**: Higher limits available

Check [Google Maps Pricing](https://developers.google.com/maps/billing-and-pricing/pricing) for details.

## Examples

### Example 1: Download using corner coordinates

**Config file (`config.yaml`):**
```yaml
REGION:
  TYPE: "corners"
  CORNERS:
    LAT1: 22.54
    LON1: 114.06
    LAT2: 22.55
    LON2: 114.07
API:
  API_KEY: "YOUR_API_KEY"
DOWNLOAD:
  ZOOM: 19
```

**Command:**
```bash
python -m applications.map_downloader
```

**Output filename:** `map_center_22.545000_114.065000_h1111w1111_z19.tif`
(Note: corners are converted to center format for filename)

### Example 2: Download using center point and dimensions

**Config file (`config.yaml`):**
```yaml
REGION:
  TYPE: "center"
  CENTER:
    LAT: 39.9015
    LON: 116.3985
    HEIGHT_M: 1000.0
    WIDTH_M: 1000.0
API:
  API_KEY: "YOUR_API_KEY"
DOWNLOAD:
  ZOOM: 19
```

**Command:**
```bash
python -m applications.map_downloader
```

**Output filename:** `map_center_39.901500_116.398500_h1000w1000_z19.tif`

### Example 3: Override config with command-line arguments

```bash
python -m applications.map_downloader \
    --type center \
    --center-lat 22.545 \
    --center-lon 114.065 \
    --height-m 500 \
    --width-m 500 \
    --zoom 20 \
    --output-dir /custom/path
```

### Example 4: Python API

```python
from applications.map_downloader import generate_geotiff

# Using corner coordinates
generate_geotiff(
    latlng1=(22.54, 114.06),
    latlng2=(22.55, 114.07),
    api_key="YOUR_API_KEY",
    zoom=19,
    out_path="data/scene_datasets/hongkong.tif"
)
```

## Troubleshooting

### API Key Errors

- Make sure your API key is valid and has Static Maps API enabled
- Check API key restrictions in Google Cloud Console

### Request Failures

- Verify coordinates are valid (lat: -90 to 90, lon: -180 to 180)
- Check internet connection
- Verify API quota hasn't been exceeded

### Large Image Downloads

- For very large regions, use lower zoom levels
- The downloader automatically tiles and stitches large images
- Be aware of API usage limits

## Filename Format

All output files use a unified center-based filename format, regardless of input type:

**Format:** `map_center_{center_lat:.6f}_{center_lon:.6f}_h{height_m:.0f}w{width_m:.0f}_z{zoom}.tif`

**Examples:**
- `map_center_39.901500_116.398500_h300w300_z19.tif`
- `map_center_22.545000_114.065000_h1000w1000_z20.tif`

When using corner-based input, the corners are automatically converted to center format for the filename. This ensures consistent naming across all downloads.

## Performance Tips

1. **Use parallel mode** for large downloads (2km×2km or larger) to save time
2. **Adjust MAX_WORKERS** based on your API quota:
   - Free tier: Use 5-10 workers to avoid rate limits
   - Paid tier: Can use 15-20 workers for faster downloads
3. **Monitor API usage** - parallel mode makes more requests per second
4. **Network stability**: Parallel mode benefits more from stable network connections

## Notes

- Downloaded maps are cached locally as GeoTIFF files
- Each download consumes API quota
- Higher zoom levels provide more detail but require more API calls
- Maps are automatically georeferenced for use with SatNav
- **All filenames use center format** for consistency, even when input is corner-based
- Large images are automatically tiled and stitched
- **Parallel mode** uses connection pooling and concurrent downloads for 4-9x speedup

## Input/Output Summary

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `API_KEY` | String | Google Maps Static API key (required) |
| `REGION.TYPE` | String | Region definition type: `"corners"` or `"center"` |
| `REGION.CORNERS.LAT1` | Float | Lower-left corner latitude (degrees, WGS84) |
| `REGION.CORNERS.LON1` | Float | Lower-left corner longitude (degrees, WGS84) |
| `REGION.CORNERS.LAT2` | Float | Upper-right corner latitude (degrees, WGS84) |
| `REGION.CORNERS.LON2` | Float | Upper-right corner longitude (degrees, WGS84) |
| `REGION.CENTER.LAT` | Float | Center point latitude (degrees, WGS84) |
| `REGION.CENTER.LON` | Float | Center point longitude (degrees, WGS84) |
| `REGION.CENTER.HEIGHT_M` | Float | North-South dimension (meters) |
| `REGION.CENTER.WIDTH_M` | Float | East-West dimension (meters) |
| `DOWNLOAD.ZOOM` | Integer | Zoom level (15-20, higher = more detail) |
| `DOWNLOAD.MODE` | String | Download mode: `"sequential"` or `"parallel"` |
| `DOWNLOAD.MAX_WORKERS` | Integer | Max concurrent workers for parallel mode |
| `OUTPUT.OUTPUT_DIR` | String | Output directory path |

### Output

| Output | Type | Description |
|--------|------|-------------|
| GeoTIFF File | File | Georeferenced TIFF image file |
| CRS | EPSG:3857 | Web Mercator projection (meters) |
| Filename Format | String | `map_center_{lat}_{lon}_h{height}w{width}_z{zoom}.tif` |
| Image Format | RGB | 3-channel RGB image (uint8) |


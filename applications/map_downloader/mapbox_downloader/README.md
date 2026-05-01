# Mapbox Raster Tiles Downloader

Download Mapbox raster tile imagery and export EPSG:3857 GeoTIFF files for SatNav scenes.
SatNav episode scenes are generated at zoom level `19`.

Use the top-level provider entry:

```bash
python -m applications.map_downloader mapbox --help
```

Direct module execution also works:

```bash
python -m applications.map_downloader.mapbox_downloader --help
```

## Access Token

Create a Mapbox access token:

- Token page: https://console.mapbox.com/account/access-tokens/
- Official token docs: https://docs.mapbox.com/accounts/guides/tokens/

Configure the token with an environment variable:

```bash
export MAPBOX_ACCESS_TOKEN="your-mapbox-token"
```

You can also pass `--access-token` explicitly.

## Single Scene

Download by lower-left and upper-right WGS84 corners:

```bash
python -m applications.map_downloader mapbox \
  --type corners \
  --lat1 46.161791698085 \
  --lon1 6.082932204008 \
  --lat2 46.179030896969 \
  --lon2 6.116151362658 \
  --zoom 19 \
  --output output/map_downloader/Geneva-1.tif
```

Download by center point and size:

```bash
python -m applications.map_downloader mapbox \
  --type center \
  --center-lat 46.170411 \
  --center-lon 6.099542 \
  --height-m 1900 \
  --width-m 2500 \
  --zoom 19 \
  --output output/map_downloader/Geneva-1.tif
```

## Common Options

- `--tileset-id`: Mapbox tileset, default `mapbox.satellite`.
- `--format`: tile image format, default `jpg90`.
- `--scale`: `1` or `2`.
- `--download-mode`: `parallel` or `sequential`.
- `--max-workers`: parallel worker count.
- `--output-dir`: output directory when `--output` is not set.
- `--use-env-proxy`: use proxy settings from the environment.
- `--no-watermark`: skip the bottom-left Mapbox attribution panel.
- `--no-attribution-file`: skip the attribution sidecar text file.

## Batch Scenes

Mapbox currently supports single-scene downloads only. Use the Google provider's `--scene-config` mode for batch downloading SatNav `scenes_list.yaml`.

By default, the final TIFF includes a bottom-left Mapbox attribution panel and writes an attribution sidecar file.

# Google Map Tiles Downloader

Download Google Map Tiles API imagery and export EPSG:3857 GeoTIFF files for SatNav scenes.
SatNav episode scenes are generated at zoom level `19`.

Use the top-level provider entry:

```bash
python -m applications.map_downloader google --help
```

Direct module execution also works:

```bash
python -m applications.map_downloader.google_downloader --help
```

## API Key

Enable Google Map Tiles API and create an API key:

- Map Tiles API: https://console.cloud.google.com/apis/library/tile.googleapis.com
- Credentials: https://console.cloud.google.com/google/maps-apis/credentials
- Official guide: https://developers.google.com/maps/documentation/tile/get-api-key

Configure the key with an environment variable:

```bash
export GOOGLE_MAPS_API_KEY="your-google-key"
```

You can also pass `--api-key` explicitly.

## Single Scene

Download by lower-left and upper-right WGS84 corners:

```bash
python -m applications.map_downloader google \
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
python -m applications.map_downloader google \
  --type center \
  --center-lat 46.170411 \
  --center-lon 6.099542 \
  --height-m 1900 \
  --width-m 2500 \
  --zoom 19 \
  --output output/map_downloader/Geneva-1.tif
```

## Batch Scenes

Google supports `--scene-config` batch mode. Batch options are read from the YAML first; explicit CLI flags override them.

Use this mode with a dataset scene-list YAML when generating SatNav scenes.

Test config:

```bash
applications/map_downloader/test_config/test_scenes_list.yaml
```

Dry run:

```bash
python -m applications.map_downloader google \
  --scene-config applications/map_downloader/test_config/test_scenes_list.yaml \
  --dry-run
```

Download:

```bash
python -m applications.map_downloader google \
  --scene-config applications/map_downloader/test_config/test_scenes_list.yaml
```

For a full dataset build, pass your own scene list YAML:

```bash
python -m applications.map_downloader google \
  --scene-config <scene-list.yaml>
```

## Config Fields

```yaml
download:
  zoom: 19
  mode: parallel
  max_workers: 12

output:
  output_dir: output/map_downloader/scenes
  filename_template: "{scene_id}.tif"
  skip_existing: true

network:
  use_env_proxy: false
```

Useful CLI overrides:

- `--output-dir`
- `--zoom`
- `--scene-id`
- `--limit`
- `--overwrite`
- `--dry-run`
- `--use-env-proxy`

If Google `createSession` fails because `tile.googleapis.com` is unreachable, retry with `--use-env-proxy`.

By default, the final TIFF includes a bottom-left Google attribution panel.

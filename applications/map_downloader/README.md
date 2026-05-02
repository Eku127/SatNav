# SatNav Map Downloader

Download satellite GeoTIFF scenes required by SatNav episodes for training and evaluation.
SatNav episode data uses scenes downloaded at zoom level `19`.

Supported providers:

- Google Map Tiles API
- Mapbox Raster Tiles API

The output files are standard GeoTIFFs named by scene id, for example `Geneva-1.tif`.

## Install

From the repository root:

```bash
pip install -r requirements.txt
pip install -r applications/map_downloader/requirements.txt
```

If the SatNav conda environment is already configured, activate it directly:

```bash
conda activate satnav
```

Run the CLI:

```bash
python -m applications.map_downloader google --help
python -m applications.map_downloader mapbox --help
```

## Get API Keys

Google:

1. Create or select a Google Cloud project with billing enabled.
2. Enable the Map Tiles API: https://console.cloud.google.com/apis/library/tile.googleapis.com
3. Create an API key: https://console.cloud.google.com/google/maps-apis/credentials
4. Restrict the key to the Map Tiles API and, when possible, to trusted IPs.

Official guide: https://developers.google.com/maps/documentation/tile/get-api-key

Mapbox:

1. Create a Mapbox account.
2. Create an access token: https://console.mapbox.com/account/access-tokens/
3. Use a token with only the scopes needed for reading map tiles, such as `styles:tiles`.

Official token docs: https://docs.mapbox.com/accounts/guides/tokens/

## Configure Keys

Use environment variables. Do not commit real keys.

```bash
export GOOGLE_MAPS_API_KEY="your-google-key"
export MAPBOX_ACCESS_TOKEN="your-mapbox-token"
```

You can also pass keys explicitly:

```bash
python -m applications.map_downloader google --api-key "$GOOGLE_MAPS_API_KEY" ...
python -m applications.map_downloader mapbox --access-token "$MAPBOX_ACCESS_TOKEN" ...
```

## Single Image Download

Use `--type corners` when you already know the lower-left and upper-right WGS84 coordinates.

Google:

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

Mapbox:

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

You can also use `--type center` with `--center-lat`, `--center-lon`, `--height-m`, and `--width-m`.

## Batch Scene Download

Batch mode downloads all scenes listed in a YAML/JSON config. It is currently implemented for the Google provider.
Batch options are read from the scene config first; explicit CLI flags override them.

Note: use this mode with a dataset scene-list YAML when generating SatNav scenes.

Minimal config:

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

scenes:
- scene_id: Geneva-1
  corners:
    lat1: 46.161791698085
    lon1: 6.082932204008
    lat2: 46.179030896969
    lon2: 6.116151362658
```

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

Useful batch options:

- `download.zoom`: tile zoom level, default `19`.
- `download.mode`: `parallel` or `sequential`.
- `download.max_workers`: parallel worker count.
- `output.output_dir`: directory for generated `.tif` scenes.
- `output.filename_template`: default `{scene_id}.tif`.
- `output.skip_existing`: skip existing outputs when `true`.
- `--scene-id Geneva-1`: download only selected scenes. Can be repeated.
- `--limit 2`: download the first N scenes.
- `--overwrite`: replace existing `.tif` files.
- `--dry-run`: validate config and print planned outputs without downloading.
- `--use-env-proxy`: use environment proxy settings.

If Google `createSession` fails because `tile.googleapis.com` is unreachable, retry with `--use-env-proxy`.

For a full dataset build, pass your own scene list YAML:

```bash
python -m applications.map_downloader google \
  --scene-config <scene-list.yaml>
```

## Notes

- Default zoom is `19`.
- Coordinates are WGS84 latitude/longitude.
- Put generated scenes outside source-controlled directories, for example `output/map_downloader/scenes` or your dataset `scenes/` directory.
- Keep provider attribution and follow Google/Mapbox terms for any redistributed outputs.

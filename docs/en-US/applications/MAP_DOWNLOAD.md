# Download Satellite Scenes

This guide uses `applications/map_downloader` to mosaic map tiles into
GeoTIFF scenes readable by SatSim. The geographic bounds for the 59
SatNav-Episodes-v0.1 scenes are defined in `scenes_list.yaml`; complete
[Episode Download](../dataset/DATA_DOWNLOAD.md) first if you do not have it.

> The downloader provides technical functionality only. It does not grant
> permission to download, store, distribute, or use map content for machine
> learning. Verify the provider's current terms and your authorization before
> use.

See [SatSim observations](../concepts/SATSIM.md) for how geographic transforms and local raster crops produce RGB.

## 1. Prepare the environment

Complete [Installation](../getting-started/INSTALLATION.md), then install the
application dependencies and configure paths from the repository root:

```bash
python -m pip install -e '.[applications]'

export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
mkdir -p "$SATNAV_SCENES_DIR"
```

Use absolute paths when the data is stored on another volume.

## 2. Configure provider credentials

Configure only the provider you will use. Google Map Tiles API is recommended
for the batch scene workflow.

### Google Map Tiles API

1. Create or select a billing-enabled Google Cloud project.
2. Enable the [Map Tiles API](https://console.cloud.google.com/apis/library/tile.googleapis.com).
3. Create an API key on the [credentials page](https://console.cloud.google.com/google/maps-apis/credentials).
4. Restrict the key to Map Tiles API, and restrict trusted IPs when possible.

```bash
export GOOGLE_MAPS_API_KEY="your-api-key"
```

See the [official Google setup guide](https://developers.google.com/maps/documentation/tile/get-api-key).

### Mapbox

1. Create a Mapbox account.
2. Create a token under [Access Tokens](https://console.mapbox.com/account/access-tokens/).
3. Grant only the minimum tile-read scopes, such as `styles:tiles`.

```bash
export MAPBOX_ACCESS_TOKEN="your-access-token"
```

See the [Mapbox token guide](https://docs.mapbox.com/accounts/guides/tokens/).
Never commit real credentials.

## 3. Build all 59 scenes

`--scene-config` batch mode currently supports Google only. Start with a dry
run; it validates the scene configuration and output paths without downloading
tiles:

```bash
python -m applications.map_downloader google \
  --scene-config "$SATNAV_DATA_ROOT/scenes_list.yaml" \
  --output-dir "$SATNAV_SCENES_DIR" \
  --dry-run
```

Then download the scenes:

```bash
python -m applications.map_downloader google \
  --scene-config "$SATNAV_DATA_ROOT/scenes_list.yaml" \
  --output-dir "$SATNAV_SCENES_DIR"
```

Existing GeoTIFFs are skipped, so rerunning the same command resumes an
interrupted download. Common options:

- `--scene-id Geneva-1`: process one scene; repeat the option for more;
- `--limit 2`: process the first two configured scenes;
- `--overwrite`: regenerate existing files;
- `--max-workers N`: set download concurrency;
- `--use-env-proxy`: honor proxy environment variables.

Check API quotas, cost, and available disk space before downloading.

## 4. Download one scene

Google and Mapbox both support single-scene download. This example defines an
area with two WGS84 corners:

```bash
python -m applications.map_downloader google \
  --type corners \
  --lat1 46.161791698085 --lon1 6.082932204008 \
  --lat2 46.179030896969 --lon2 6.116151362658 \
  --zoom 19 \
  --output "$SATNAV_SCENES_DIR/Geneva-1.tif"
```

For Mapbox, replace `google` with `mapbox`; the bounds are unchanged. You can
also use `--type center` with `--center-lat`, `--center-lon`, `--height-m`, and
`--width-m`.

SatNav scenes use zoom level `19`. The downloader preserves provider-required
attribution; Mapbox also writes a matching `.attribution.txt` file.

## 5. Validate the data configuration

After downloading, run:

```bash
bash scripts/validation/data_validation.sh
```

The script checks:

- that 59 GeoTIFF files exist;
- train, val_seen, and val_unseen episode counts;
- that episode data loads successfully;
- that every referenced scene exists.

Successful validation ends with:

```text
SatNav data configuration is complete.
```

## 6. Troubleshooting

### Why does the downloader raise `ModuleNotFoundError`?

Install application dependencies from the repository root:

```bash
python -m pip install -e '.[applications]'
```

### Why does API authentication fail?

Verify that the credential variable is exported in the current shell, the API
is enabled, billing is active, and key restrictions or token scopes are
correct. Never put credentials in repository configs.

### How do I resume an interrupted download?

Rerun the same command. Do not add `--overwrite`, or completed scenes will be
downloaded again.

### Why can I not connect to `tile.googleapis.com`?

If your network requires a proxy, add `--use-env-proxy` so the downloader uses
the current proxy environment variables.

### Why are Google satellite tiles unavailable?

In addition to quota and coverage limits, projects with an EEA billing address
cannot request 2D satellite tiles. Check the returned error against Google's
[error guide](https://developers.google.com/maps/documentation/tile/error_handling).

## 7. Terms of use

Google Map Tiles API policies restrict unauthorized prefetching, storage, and
offline use, and classify image analysis and machine interpretation as
disallowed non-visualization uses. Unless your agreement grants additional
rights, do not use its output for offline training or evaluation. Read the
[Map Tiles API policies](https://developers.google.com/maps/documentation/tile/policies)
and [billing guide](https://developers.google.com/maps/documentation/tile/usage-and-billing).

Mapbox users should read the
[Raster Tiles API documentation](https://docs.mapbox.com/api/maps/raster-tiles/)
and applicable service terms. SatNav distributes no third-party satellite
imagery and does not obtain or sublicense map-content rights for users.

After preparing scenes, inspect them with [SatSim Viewer](VIEWER.md) or generate
offline training data with [Trajectory Generation](TRAJECTORY_GENERATION.md).

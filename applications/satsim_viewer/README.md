# SatSim Viewer

Interactive viewers for inspecting SatNav satellite maps and VLN episodes.

Two modes are available:

- `free`: move freely on a GeoTIFF map with SatSim.
- `task`: step through SatNav VLN episodes with any configured simulator and
  view task metrics.

The default free-viewer map is:

```bash
applications/resources/map.tif
```

This bundled scene is a procedurally generated CC0 example, not provider map
or satellite imagery. See [resource provenance](../resources/README.md).

The default task-viewer config is:

```bash
applications/resources/satnav_example_task.yaml
```

## Run

From the repository root:

```bash
conda activate satnav
```

Free viewer:

```bash
python -m applications.satsim_viewer
python -m applications.satsim_viewer free
```

Task viewer:

```bash
python -m applications.satsim_viewer task
```

Use a custom config:

```bash
python -m applications.satsim_viewer free --config applications/satsim_viewer/config.yaml
python -m applications.satsim_viewer task --config applications/resources/satnav_example_task.yaml
```

Run a task automatically with SatNav's `ReferencePathFollower`:

```bash
python -m applications.satsim_viewer task \
  --config applications/resources/satnav_example_task.yaml \
  --autoplay-reference
```

## Free Viewer Config

Edit `applications/satsim_viewer/config.yaml`:

```yaml
TIF_PATH: applications/resources/map.tif

SIMULATOR:
  FORWARD_STEP_SIZE: 30
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 512
    HEIGHT: 512
    HFOV: 90

AGENT:
  ALTITUDE: 100.0
  ROTATION: 0.0
  ALTITUDE_STEP_SIZE: 10.0
```

## Task Viewer Config

The task viewer uses a normal SatNav task YAML. The included test config points to a small episode file and the shared sample map:

```yaml
DATASET:
  DATA_PATH: applications/resources/satnav_example_episodes.json
  SCENES_DIR: applications/resources/
```

Scene files must be named by scene id, for example `map.tif`.

### Top-down compatibility

Task viewer accepts `--topdown auto|satellite|vector|off` (default: `auto`):

- `auto` preserves the existing SatSim behavior by enabling the satellite
  `TOP_DOWN_MAP` measure. For an external simulator it uses the viewer-local
  vector map instead.
- `satellite` explicitly enables SatSim's GeoTIFF-backed measure.
- `vector` draws the episode reference path, actual agent path, goal radius,
  and heading. It uses only public episode and simulator state, so it works
  with render-service backends that do not expose a GeoTIFF.
- `off` displays RGB and task text only. `--no-topdown` is an alias.

The vector view is a navigation debug view, not satellite or rendered imagery.
It does not change task measurements, observations, actions, or episode data.

## Controls

Free viewer:

- `w`: move forward
- `s`: move backward
- `a`: turn left
- `d`: turn right
- `q`: increase altitude
- `e`: decrease altitude
- `p`: save current image
- `Esc`: quit

Task viewer:

- `w`: move forward
- `a`: turn left
- `d`: turn right
- `t`: toggle top-down map
- `p`: pause/resume reference autoplay (when enabled)
- `n`: execute one reference action while autoplay is paused
- `Space`: stop episode and show metrics
- `Esc`: quit

## Notes

- OpenCV opens an interactive window, so a desktop display or X forwarding is required.
- If the camera view exceeds map bounds, reduce altitude or move away from the edge.
- SatSim satellite maps read WGS84 coordinates from EPSG:3857 GeoTIFF scenes.
- Vector top-down supports WGS84 and episode `local_enu` coordinate frames.

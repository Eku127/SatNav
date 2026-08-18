# HUGE 3DGS simulator backend

This application renders SatNav observations from HUGE-Bench 3D Gaussian
scenes. It is a second **process**, not a second repository or a cloud service:

```text
SatNavDataset + Env
  -> satnav.sims.huge3dgs_wrapper.Huge3DGSSimWrapper
  -> authenticated local Unix socket
  -> python -m applications.huge3dgs_renderer render-server
  -> 448x448 HWC uint8 RGB bytes
```

The process boundary keeps Torch, CUDA, gsplat, PLY readers, and multi-gigabyte
scene assets out of the framework/evaluation environment. One persistent GPU
process loads a scene once and can serve multiple local evaluation ranks.

## Merge inventory

The SatNav side requires:

- `satnav/sims/huge3dgs_wrapper.py`
- `satnav/sims/huge3dgs_ipc.py`
- `configs/huge3dgs_eval.example.yaml`
- `configs/huge3dgs_scene_adapter.example.yaml`
- the existing dynamic external-simulator factory in `satnav.sims`

The renderer side requires this entire directory, not only the two wrapper
files:

- `applications/huge3dgs_renderer/__main__.py`
- `applications/huge3dgs_renderer/rendering.py`
- `applications/huge3dgs_renderer/scene_manifest.py`
- `applications/huge3dgs_renderer/gsplat_backend.py`
- `applications/huge3dgs_renderer/requirements.txt`
- `configs/huge3dgs_renderer.example.yaml`
- `configs/huge3dgs_renderer_scenes.example.json`

No project-specific validation launcher is required. The renderer is started
by its module entry point, and a one-frame integration smoke uses SatNav's
existing classic evaluator (see below).

No change to `satnav/sims/satsim_wrapper.py` is required.

## Two isolated environments

The normal SatNav environment needs NumPy, PyProj, OmegaConf, Pillow, and the
existing SatNav dependencies. It does not need Torch or gsplat.

The renderer environment needs a mutually compatible set of:

- Python, PyTorch, and CUDA runtime;
- gsplat and a matching prebuilt `gsplat_cuda` extension;
- NumPy, SciPy, Plyfile, PyProj, OmegaConf, and Pillow;
- an NVIDIA GPU with enough memory for the selected scene.

The validated target used Python 3.10, PyTorch 2.8.0+cu128, gsplat 1.5.3, and
an H100. Install the platform-appropriate PyTorch build first, then install
`applications/huge3dgs_renderer/requirements.txt`. Other versions are allowed
only after the same runtime and golden render validation.

For the validated CUDA 12.8 environment:

```bash
python3.10 -m venv .venv-huge3dgs
source .venv-huge3dgs/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r applications/huge3dgs_renderer/requirements.txt
```

## Local files that are intentionally not committed

The public coordinate-adapter manifest maps episode logical IDs and synthetic
WGS84/Web-Mercator origins to renderer ENU scene IDs. It contains no asset
paths.

The public renderer scene manifest uses
`$SATNAV_HUGE3DGS_ASSET_ROOT/data_3d/...` paths. Each scene record contains the
renderer scene ID, source/license/checksum metadata, ENU transform and bounds,
camera contract, and renderer version. The ENU transform maps one frozen scene
ground datum to up=0. `SATNAV_HUGE3DGS_FLIGHT_HEIGHT_M` selects one absolute
normalized-ENU flight plane for an evaluation run, and the episode altitude
must match it. The camera remains on that plane for the entire episode. The
renderer never adds the configured height to a per-position mesh/roof height;
the full mesh is used only for geometry-clearance checks.

Download only `2_city`, `3_road`, and `4_lake` from the official HUGE-Bench
asset repository, retain its `data_3d/<scene>/...` layout, and verify the
published SHA256 values before rendering:

```bash
python -m pip install "huggingface_hub[cli]"
hf download yu781986168/3DGS_Mesh_Envs \
  --repo-type dataset \
  --include "archives/3DGS_Mesh_Envs_2_city.tar" \
            "archives/3DGS_Mesh_Envs_3_road.tar" \
            "archives/3DGS_Mesh_Envs_4_lake.tar" \
            "SHA256SUMS.txt" \
  --local-dir /absolute/path/to/extracted/HUGE/assets

cd /absolute/path/to/extracted/HUGE/assets
echo "dc3fb805140b133a6e5667c4d5cdfddea37621629374e9af3bdeabeddab2cee2  archives/3DGS_Mesh_Envs_2_city.tar" | sha256sum -c -
echo "8ed4d08b5fbc5a20a9d90dd8378f22fab5ebb726d8eb378c57356fb8f19831b7  archives/3DGS_Mesh_Envs_3_road.tar" | sha256sum -c -
echo "242e17c2b70c2bb91822a73fcfbb0c3a7eba46fcb8d76f8d866e11c2f013c722  archives/3DGS_Mesh_Envs_4_lake.tar" | sha256sum -c -
tar -xf archives/3DGS_Mesh_Envs_2_city.tar
tar -xf archives/3DGS_Mesh_Envs_3_road.tar
tar -xf archives/3DGS_Mesh_Envs_4_lake.tar
```

The assets, socket, auth key, optional platform-specific prebuilt extension,
render cache, and review videos remain outside Git.

Create an ignored renderer config from the public template and set:

```bash
cp configs/huge3dgs_renderer.example.yaml \
  configs/local_huge3dgs_renderer.yaml

export SATNAV_HUGE3DGS_ENDPOINT=/absolute/path/to/huge3dgs.sock
export SATNAV_HUGE3DGS_AUTHKEY=choose-a-local-auth-key
export SATNAV_HUGE3DGS_ASSET_ROOT=/absolute/path/to/extracted/HUGE/assets
export SATNAV_GSPLAT_EXTENSION=/absolute/path/to/gsplat_cuda.so
export SATNAV_HUGE3DGS_FLIGHT_HEIGHT_M=50
```

## Start and validate the renderer

The standard public entry point is:

```bash
python -m applications.huge3dgs_renderer render-server \
  --config configs/local_huge3dgs_renderer.yaml
```

`SATNAV_GSPLAT_EXTENSION` is optional. When set, the application loads that
prebuilt extension directly and avoids implicit JIT compilation. Without it,
the installed gsplat package controls extension loading.

Then run SatNav in the normal environment with
`configs/huge3dgs_eval.example.yaml`. Its simulator section is:

```yaml
SIMULATOR:
  TYPE: huge_3dgs
  CLASS: satnav.sims.huge3dgs_wrapper.Huge3DGSSimWrapper
  FORWARD_STEP_SIZE: 10
  TURN_ANGLE: 15
  SCENE_ADAPTER_MANIFEST: configs/huge3dgs_scene_adapter.example.yaml
  RGB_SENSOR:
    WIDTH: 448
    HEIGHT: 448
    HFOV: 90
  RENDERER:
    ENDPOINT: ${oc.env:SATNAV_HUGE3DGS_ENDPOINT}
    AUTHKEY: ${oc.env:SATNAV_HUGE3DGS_AUTHKEY}
    FIXED_HEIGHT_M: ${oc.env:SATNAV_HUGE3DGS_FLIGHT_HEIGHT_M}
    REQUIRE_GEOMETRY_CLEARANCE: true
    MIN_GEOMETRY_CLEARANCE_M: 10
    REQUIRE_RENDER_COMPLETE: true
    MAX_INVALID_FRACTION: 0.02
```

The wrapper preserves episode coordinates at the public API and converts them
to local ENU only for renderer requests. `FIXED_HEIGHT_M` is configurable; it
is a fixed absolute flight plane, not an offset from the mesh. Every episode
position must carry the same altitude as the configured value. A missing
renderer, coordinate record, scene, clearance result, or complete RGB frame is
an explicit error. There is no fallback to SatSim or GeoTIFF cropping.

## Interactive task viewer

After starting the renderer, the existing task viewer can display live 3DGS
RGB and automatically run SatNav's `ReferencePathFollower`:

```bash
export SATNAV_HUGE3DGS_EPISODES=/absolute/path/to/VLN_episodes.json
python -m applications.satsim_viewer task \
  --config configs/huge3dgs_eval.example.yaml \
  --topdown auto \
  --autoplay-reference
```

For this external simulator, `--topdown auto` selects a renderer-independent
vector path view and does not instantiate the SatSim-only `TOP_DOWN_MAP`
measure. The left panel is the real 3DGS RGB observation; the right panel is a
debug drawing of the reference and executed paths. Press `p` to pause/resume,
`n` to execute one action while paused, and `Esc` to quit. OpenCV requires a
graphical desktop or X11 forwarding with a non-empty `DISPLAY`.

## Minimal integration smoke

Set the generated episode file and run one step through the existing
`SatNavDataset + Env` evaluator in the normal SatNav environment:

```bash
export SATNAV_HUGE3DGS_EPISODES=/absolute/path/to/VLN_episodes.json
python -m baselines.classic.evaluate \
  --method reference \
  --config configs/huge3dgs_eval.example.yaml \
  --limit 1 \
  --max-steps 1 \
  --output-dir output/huge3dgs_smoke \
  --fail-fast \
  --fail-on-episode-error
```

This smoke checks dataset loading, dynamic simulator construction, coordinate
conversion, IPC, scene loading, and a real 448x448 RGB render. `--max-steps 1`
intentionally makes it a transport/render check, not a navigation metric run.
Remove that option for a complete reference-path evaluation.

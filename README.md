# SatNav: Continuous-State Vision-and-Language Navigation via Satellite Maps

## Overview

SatNav is a platform for evaluating continuous-state Vision-and-Language Navigation (VLN) on satellite maps. It supports navigation in geographic coordinate space, where agents move continuously with longitude, latitude, altitude, and heading.

The core simulator is **SatSim**:

- **SatSim**: a 2D satellite-map simulator that renders RGB observations from local GeoTIFF files.

**Key Features**:

- Uses satellite maps as visual environments instead of indoor 3D scans.
- Represents agent position with longitude, latitude, and altitude.
- Represents orientation with heading, where `0` degrees means north.
- Renders 2D RGB observations from local GeoTIFF files.
- Supports offline trajectory generation, baseline training, and online rollout evaluation.

---

## 1. Installation

```bash
# Create the tracked base environment.
conda env create -f environments/satnav/conda.yml
conda activate satnav

# Core simulator, dataset, and framework-independent evaluator.
pip install -e .

# Optional classic baselines. Install the PyTorch build for your CUDA first;
# this CUDA 12.1 combination has been tested:
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[classic]'

# Optional map, trajectory, and video applications.
pip install -e '.[applications]'
```

The standalone wheel contract covers core Python imports and packaged example
resources. Repository-level classic/VLM launchers and shell workflows require
a SatNav source checkout installed in editable mode, as shown above; they are
not standalone-wheel entrypoints.

For a manually managed environment, use Python 3.8 or newer and run the same
editable-install commands from the repository root.

```bash
conda create -n satnav python=3.8
```

## 2. Local Configuration

All public configs and scripts use repository-relative example resources by
default. Machine paths, private dataset locations, credentials, and maintainer
settings should stay in Git-ignored local overlays.

For shared shell settings, copy the tracked template:

```bash
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

Training, evaluation, and quickstart shell entrypoints load this file
automatically when it exists. Seq2Seq and CMA also support component-specific
overlays:

```bash
mkdir -p scripts/seq2seq/.local scripts/cma/.local
cp scripts/seq2seq/local.env.example scripts/seq2seq/.local/env.sh
cp scripts/cma/local.env.example scripts/cma/.local/env.sh
```

Canonical classic evaluation has its own template:

```bash
mkdir -p baselines/classic/.local
cp baselines/classic/local.env.example baselines/classic/.local/env.sh
```

Each external VLM has a separate environment and ignored overlay. Follow the
corresponding baseline guide rather than trying to share the core or classic
environment:

```text
baselines/vlm/streamvln/.local/env.sh
baselines/vlm/navila/.local/env.sh
baselines/vlm/uninavid/.local/env.sh
baselines/vlm/openfly/.local/env.sh
```

Existing environment variables such as `CONDA_INIT`, `CONDA_ENV`,
`CONFIG_PATH`, `OUTPUT_ROOT`, and `CUDA_DEVICES` remain supported. Explicit
shell environment variables take precedence over values loaded from `.local`;
`.local` values take precedence over public repository-relative defaults. Use
an environment variable for a one-off override and an ignored `.local/env.sh`
for persistent machine paths.

For real dataset paths, copy the relevant public baseline config to an ignored
`configs/local_*.yaml` file, edit it locally, and select it through
`CONFIG_PATH` or the corresponding component setting. For example:

```bash
cp configs/baselines/seq2seq_offline_train.yaml configs/local_seq2seq_train.yaml
CONFIG_PATH=configs/local_seq2seq_train.yaml \
  bash scripts/seq2seq/train_offline_ddp.sh
```

The `.local/`, `configs/local_*.yaml`, `output/`, and `runtime/` destinations
are ignored by Git and must not be included in release commits. Keep local
datasets, GeoTIFFs, model weights, credentials, and run artifacts there or in
external storage referenced by `.local`.

## 3. Dataset Release

SatNav-Episodes-v0.1 is available on Kaggle:

```text
https://www.kaggle.com/datasets/07af1ab653c3d8d0518027b41d05dfa677d6a414131b27c4b024b887d74c6a68
```

The release contains episode JSON files, train/evaluation splits,
`scenes_list.yaml`, and dataset documentation. It does not include raw
OpenStreetMap extracts, satellite imagery, map tiles, or simulator GeoTIFF
scenes. Use `applications/map_downloader` with `scenes_list.yaml` to reconstruct
local scene files, then use `applications/trajectory_generation` to generate
offline trajectory data for training.

## 4. SatNav Core and Public API

SatNav is mainly built around `SatNavDataset`, `Env`, `VLNTask`, and `SatSim`. The dataset provides navigation episodes, `Env` connects the task and simulator, and `SatSim` renders RGB observations from GeoTIFF satellite maps.

Use the public environment contract in applications and baseline adapters:

```python
from applications.resources import load_example_task_config
from satnav.core.env import Env

config = load_example_task_config()
env = Env(config)
try:
    observation = env.reset()
    observation, done, info = env.step("STOP")
    metrics = env.get_metrics()
    episode = env.current_episode
    state = env.agent_state
finally:
    env.close()
```

Supported properties include `episodes`, `current_episode`, `simulator`,
`agent_state`, `last_step_info`, `episode_over`, `observation_space`, and
`action_space`. `step()` returns `(observation, done, info)`. Applications
must not reach through `_dataset`, `_task`, or `_sim`. Dataset `scene_id` is a
stable logical name; machine-local resolution is kept in runtime-only
`scene_path` and excluded from normal serialization and benchmark results.

For more details about the module structure and runtime flow, see [SatNav Architecture Overview](doc/simulator/SATNAV_ARCHITECTURE.md).

## 5. Applications

SatNav includes several utility applications for preparing maps, inspecting the simulator, and generating training trajectories:

- [Map Downloader](applications/map_downloader/README.md): downloads Google or Mapbox satellite tiles and exports GeoTIFF scenes for SatNav.
- [Episode Processing](applications/episode_processing/README.md): inspects per-city episode sources and builds canonical train/validation release files.
- [SatSim Viewer](applications/satsim_viewer/README.md): interactively inspects satellite maps and VLN episodes.
- [Trajectory Generation](applications/trajectory_generation/README.md): generates offline trajectory data from episode reference paths.

Note: if you want to train or evaluate models with SatNav episodes, you usually need to first prepare GeoTIFF scenes with Map Downloader and then generate offline trajectory data with Trajectory Generation.

## 6. Examples

`examples/` provides two minimal scripts showing how to follow reference paths and navigation waypoints in SatNav:

- `reference_follower_example.py`: uses `ReferencePathFollower` to follow the reference path of a single episode.
- `satnav_path_follower_example.py`: uses `SatNavPathFollower` to run multiple episodes and output metrics or videos.

For usage details, see [Examples README](examples/README.md).

## 7. Baselines and Unified Evaluation

Classic baselines live under `baselines/classic/`: Random,
ReferenceFollower, Seq2Seq, and CMA. Seq2Seq and CMA use offline imitation
learning; all four methods use the same framework-independent
`satnav.evaluation` rollout and result contract. Importing `satnav` or
`satnav.evaluation` does not import PyTorch.

The neural model documentation remains available here:

- [Seq2Seq](doc/models/SEQ2SEQ_IMPLEMENTATION.md): a lightweight recurrent baseline that encodes the instruction and current RGB observation before predicting navigation actions.
- [CMA](doc/models/CMA_IMPLEMENTATION.md): a recurrent baseline with cross-modal attention that fuses language and visual features before predicting actions.

The complete SatNav-v0.1 data preparation, training, and evaluation workflow
for both models is documented in [Classic Baselines](docs/BASELINE_CLASSIC.md).

External VLMs keep their incompatible model environments outside the core
package. The maintained integrations are
[StreamVLN](docs/BASELINE_STREAMVLN.md),
[NaVILA](docs/BASELINE_NAVILA.md),
[Uni-NaVid](docs/BASELINE_UNINAVID.md), and
[OpenFly](docs/BASELINE_OPENFLY.md). They all use the public `Env` and
`satnav.evaluation.PolicyAdapter` contracts; see the
[VLM baseline overview](baselines/vlm/README.md) for ownership and environment
boundaries. To connect another model, follow the
[model integration guide](docs/MODEL_INTEGRATION.md).

You can quickly run training and evaluation with the bundled tiny example data:

```bash
bash scripts/quickstart_models.sh
```

This script prepares the vocabulary, GloVe embeddings, offline trajectory data, and then trains and evaluates both Seq2Seq and CMA. For the training flow, prerequisites, and output layout, see the [training guide](docs/TRAINING.md).

For canonical SatNav-v0.1 evaluation, first configure the ignored classic
overlay described in Section 2, then run:

```bash
bash scripts/classic/eval.sh random val_seen 8
bash scripts/classic/eval.sh reference_follower val_seen 8
bash scripts/classic/eval_parallel.sh random val_seen 2 0,1 8
```

The launcher defaults to `SATNAV_MAX_STEPS=5`; set `SATNAV_MAX_STEPS=500` for
longer rollout. Seq2Seq and CMA additionally require the matching checkpoint,
vocabulary, and local eval config variables documented in
[Classic Baselines](baselines/classic/README.md). Every run writes rank-local
JSONL, done markers, and `summary.json` under
`output/baselines/classic/<method>/<split>/<steps>steps/<N>rank/`.

Versioned SatNav-v0.1 evaluation settings are provided under
`configs/benchmark/` for `val_seen` and `val_unseen`. Use the five-step smoke
settings to check a rollout pipeline and the 500-step official settings when
reporting benchmark results. See the [evaluation guide](docs/EVALUATION.md) for
the complete protocol and output format.

Canonical online evaluation uses `configs/satnav_eval_task.yaml`: Boundary and
Road success radii are 10 m and the LandmarkSet success radius is 30 m. The
tighter 3 m LandmarkSet radius in the trajectory-generation config is only a
waypoint-arrival tolerance for producing offline expert trajectories; it is
not an evaluation threshold.

## 8. Regression and Release Checks

Run the automated regression suite in the `satnav` environment:

```bash
python -m pytest -q
```

Most tests use fake simulators and temporary output directories, so they do
not require private maps or checkpoints. Targeted real-data and GPU smoke
tests are documented separately and are not part of the default suite.

Before preparing a public release tree, also run:

```bash
bash scripts/check_release_hygiene.sh
```

This checks candidate release files for tracked local-only paths, common
machine-specific values, credential-like values, and placeholder repository
metadata. It checks the current tree; a public release must additionally use a
clean history that never contained private local information.

Ignored `.local` overlays are allowed in a development checkout and are never
read or printed by the normal check. To validate a sanitized release checkout
where no ignored local-only files may exist, run:

```bash
SATNAV_RELEASE_TREE=1 bash scripts/check_release_hygiene.sh
```

## License

SatNav uses separate licenses for code, documentation, episode metadata, and
third-party map content.

- Source code is released under the MIT License. See `LICENSE`.
- Documentation is released under CC BY 4.0 unless otherwise stated.
- SatNav episode JSON files and related benchmark parameter files are released under
  ODbL-1.0 because they may contain information derived from OpenStreetMap.
- The bundled example `applications/resources/map.tif` is a procedurally
  generated synthetic raster dedicated to the public domain under CC0 1.0;
  see [resource provenance](applications/resources/README.md).
- Google Maps, Mapbox, and other third-party satellite or map imagery are not
  included in the SatNav-Episodes release and are not sublicensed by the
  authors.

See `DATA_LICENSE.md` and `NOTICE` for details.

## Acknowledgements

SatNav is inspired by [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). We sincerely thank these projects and their developers for their contributions to embodied AI and the Vision-and-Language Navigation community.

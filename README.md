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
# Recommended: create the tracked base environment.
conda env create -f environments/satnav/conda.yml
conda activate satnav

# Install PyTorch according to your CUDA version.
# The following CUDA 12.1 combination has been tested:
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

cd /path/to/SatNav
pip install -r requirements.txt
pip install -r applications/map_downloader/requirements.txt
pip install -e .
```

The existing manual environment command remains equivalent:

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

Existing environment variables such as `CONDA_INIT`, `CONDA_ENV`,
`CONFIG_PATH`, `OUTPUT_ROOT`, and `CUDA_DEVICES` remain supported. Explicit
environment variables take precedence over the new prefixed defaults.

For real dataset paths, copy the relevant public baseline config to an ignored
`configs/local_*.yaml` file, edit it locally, and select it through
`CONFIG_PATH` or the corresponding component setting. For example:

```bash
cp configs/baselines/seq2seq_offline_train.yaml configs/local_seq2seq_train.yaml
CONFIG_PATH=configs/local_seq2seq_train.yaml \
  bash scripts/seq2seq/train_offline_ddp.sh
```

The `.local/` and `configs/local_*.yaml` destinations are ignored by Git and
must not be included in release commits.

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

## 4. SatNav Core

SatNav is mainly built around `SatNavDataset`, `Env`, `VLNTask`, and `SatSim`. The dataset provides navigation episodes, `Env` connects the task and simulator, and `SatSim` renders RGB observations from GeoTIFF satellite maps.

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

## 7. Baseline Models

SatNav currently provides two VLN baselines:

- [Seq2Seq](doc/models/SEQ2SEQ_IMPLEMENTATION.md): a lightweight recurrent baseline that encodes the instruction and current RGB observation before predicting navigation actions.
- [CMA](doc/models/CMA_IMPLEMENTATION.md): a recurrent baseline with cross-modal attention that fuses language and visual features before predicting actions.

You can quickly run training and evaluation with the bundled tiny example data:

```bash
bash scripts/quickstart_models.sh
```

This script prepares the vocabulary, GloVe embeddings, offline trajectory data, and then trains and evaluates both Seq2Seq and CMA. For detailed steps and default output paths, see [Baseline Model Quickstart](doc/models/QUICKSTART.md).

## 8. Regression and Release Checks

Run the lightweight navigation and trajectory-generator regression suite in
the `satnav` environment:

```bash
python -m unittest discover -s tests -v
```

The suite uses fake simulators and temporary output directories, so it does
not require external maps, checkpoints, or persistent test artifacts.

Before preparing a public release tree, also run:

```bash
bash scripts/check_release_hygiene.sh
```

This checks candidate release files for tracked local-only paths, common
machine-specific values, credential-like values, and placeholder repository
metadata. It checks the current tree; a public release must additionally use a
clean history that never contained private local information.

## License

SatNav uses separate licenses for code, documentation, episode metadata, and
third-party map content.

- Source code is released under the MIT License. See `LICENSE`.
- Documentation is released under CC BY 4.0 unless otherwise stated.
- SatNav episode JSON files and related benchmark metadata are released under
  ODbL-1.0 because they may contain information derived from OpenStreetMap.
- Google Maps, Mapbox, and other third-party satellite or map imagery are not
  included in the SatNav-Episodes release and are not sublicensed by the
  authors.

See `DATA_LICENSE.md` and `NOTICE` for details.

## Acknowledgements

SatNav is inspired by [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). We sincerely thank these projects and their developers for their contributions to embodied AI and the Vision-and-Language Navigation community.

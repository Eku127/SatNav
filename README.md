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
conda create -n satnav python=3.8
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

## 2. SatNav Core

SatNav is mainly built around `SatNavDataset`, `Env`, `VLNTask`, and `SatSim`. The dataset provides navigation episodes, `Env` connects the task and simulator, and `SatSim` renders RGB observations from GeoTIFF satellite maps.

For more details about the module structure and runtime flow, see [SatNav Architecture Overview](doc/simulator/SATNAV_ARCHITECTURE.md).

## 3. Applications

SatNav includes several utility applications for preparing maps, inspecting the simulator, and generating training trajectories:

- [Map Downloader](applications/map_downloader/README.md): downloads Google or Mapbox satellite tiles and exports GeoTIFF scenes for SatNav.
- [SatSim Viewer](applications/satsim_viewer/README.md): interactively inspects satellite maps and VLN episodes.
- [Trajectory Generation](applications/trajectory_generation/README.md): generates offline trajectory data from episode reference paths.

Note: if you want to train or evaluate models with SatNav episodes, you usually need to first prepare GeoTIFF scenes with Map Downloader and then generate offline trajectory data with Trajectory Generation.

## 4. Examples

`examples/` provides two minimal scripts showing how to follow reference paths and navigation waypoints in SatNav:

- `reference_follower_example.py`: uses `ReferencePathFollower` to follow the reference path of a single episode.
- `satnav_path_follower_example.py`: uses `SatNavPathFollower` to run multiple episodes and output metrics or videos.

For usage details, see [Examples README](examples/README.md).

## 5. Baseline Models

SatNav currently provides two VLN baselines:

- [Seq2Seq](doc/models/SEQ2SEQ_IMPLEMENTATION.md): a lightweight recurrent baseline that encodes the instruction and current RGB observation before predicting navigation actions.
- [CMA](doc/models/CMA_IMPLEMENTATION.md): a recurrent baseline with cross-modal attention that fuses language and visual features before predicting actions.

You can quickly run training and evaluation with the bundled tiny example data:

```bash
bash scripts/quickstart_models.sh
```

This script prepares the vocabulary, GloVe embeddings, offline trajectory data, and then trains and evaluates both Seq2Seq and CMA. For detailed steps and default output paths, see [Baseline Model Quickstart](doc/models/QUICKSTART.md).

## Acknowledgements

SatNav is inspired by [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). We sincerely thank these projects and their developers for their contributions to embodied AI and the Vision-and-Language Navigation community.

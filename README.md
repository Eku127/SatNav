# SatNav: Continuous-State Vision-and-Language Navigation via Satellite Maps

SatNav is an open-source platform for vision-and-language navigation in
continuous geographic space. Agents follow language instructions over satellite
maps while moving with longitude, latitude, altitude, and heading.

The platform is built around **SatSim**, a 2D simulator that renders RGB
observations from local GeoTIFF scenes. SatNav also provides dataset tools,
offline trajectory generation, model training, and a shared online evaluation
interface.

Documentation: [English](docs/en-US/README.md) ·
[简体中文](docs/zh-CN/README.md)

## Features

- Continuous navigation in WGS84 geographic coordinates.
- Configurable RGB rendering from local satellite GeoTIFFs.
- Episode loading, scene resolution, task sensors, actions, and navigation
  metrics.
- Map preparation, episode inspection, and offline expert-trajectory generation.
- Framework-independent online evaluation with deterministic episode selection,
  multi-rank sharding, resume, and result aggregation.
- Classic baselines: Random, ReferenceFollower, Seq2Seq, and CMA.
- VLM integrations: StreamVLN, NaVILA, Uni-NaVid, and OpenFly.
- A public `Env` and `PolicyAdapter` contract for integrating new models.

## Quick Start

Clone SatNav and create the core environment:

```bash
git clone https://github.com/Eku127/SatNav.git
cd SatNav

conda env create -f environments/satnav/conda.yml
conda activate satnav
python -m pip install -e .
```

Run the bundled synthetic example:

```bash
python examples/reference_follower_example.py
```

The example includes its own episodes and GeoTIFF scene, so it does not require
the SatNav-v0.1 dataset, downloaded satellite imagery, or a model checkpoint.
See [Installation](docs/en-US/getting-started/INSTALLATION.md) and
[Examples](docs/en-US/getting-started/EXAMPLES.md) for the complete setup and
outputs.

## Dataset and Models

[SatNav-Episodes-v0.1](https://www.kaggle.com/datasets/07af1ab653c3d8d0518027b41d05dfa677d6a414131b27c4b024b887d74c6a68)
provides train, `val_seen`, and `val_unseen` episodes together with the scene
list. Prepare local GeoTIFF scenes separately before running full training or
evaluation.

- [Download episodes](docs/en-US/dataset/DATA_DOWNLOAD.md)
- [Prepare satellite scenes](docs/en-US/applications/MAP_DOWNLOAD.md)
- [Generate offline trajectories](docs/en-US/applications/TRAJECTORY_GENERATION.md)
- [Understand the dataset format](docs/en-US/dataset/DATASET_FORMAT.md)
- [Download released VLM checkpoints](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)

## Baselines

All baselines interact with the public SatNav environment and write results
through the shared evaluation contract.

| Category | Models | Guide |
| --- | --- | --- |
| Classic | Random, ReferenceFollower, Seq2Seq, CMA | [Classic Baselines](docs/en-US/training/CLASSIC.md) |
| VLM | StreamVLN | [StreamVLN](docs/en-US/training/vlm/STREAMVLN.md) |
| VLM | NaVILA | [NaVILA](docs/en-US/training/vlm/NAVILA.md) |
| VLM | Uni-NaVid | [Uni-NaVid](docs/en-US/training/vlm/UNINAVID.md) |
| VLM | OpenFly | [OpenFly](docs/en-US/training/vlm/OPENFLY.md) |

Each VLM uses an isolated environment because its PyTorch, Transformers, and
FlashAttention requirements differ. Released SatNav VLM checkpoints are
available in the
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo).

## Documentation

| Goal | Guide |
| --- | --- |
| Install SatNav | [Installation](docs/en-US/getting-started/INSTALLATION.md) |
| Run the included examples | [Examples](docs/en-US/getting-started/EXAMPLES.md) |
| Use the environment and simulator API | [Core API](docs/en-US/core/CORE_API.md) |
| Download and prepare the dataset | [Dataset](docs/en-US/dataset/DATA_DOWNLOAD.md) |
| Inspect scenes and episodes | [SatSim Viewer](docs/en-US/applications/VIEWER.md) |
| Train a model | [Training](docs/en-US/training/README.md) |
| Evaluate a model | [Evaluation](docs/en-US/evaluation/README.md) |
| Integrate a new model | [Model Integration](docs/en-US/development/MODEL_INTEGRATION.md) |

The complete task-oriented index is available in the
[English documentation](docs/en-US/README.md) and
[Chinese documentation](docs/zh-CN/README.md).

## Repository Structure

```text
satnav/       Core environment, dataset, simulator, task, and evaluation APIs
configs/      Public task and baseline configurations
applications/ Map, episode, trajectory, and viewer tools
baselines/    Classic and VLM integrations
examples/     Runnable examples using bundled synthetic resources
scripts/      Training, evaluation, validation, and utility entry points
docs/         English and Chinese documentation
```

## License

SatNav source code is released under the [MIT License](LICENSE). Dataset,
documentation, map content, and third-party components may use different terms;
see [DATA_LICENSE.md](DATA_LICENSE.md) and [NOTICE](NOTICE).

## Acknowledgements

SatNav's overall code architecture is inspired by
[VLN-CE](https://github.com/jacobkrantz/VLN-CE) and
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab). We thank their
authors and contributors for sharing their work with the community.

We also thank the teams behind
[StreamVLN](https://github.com/InternRobotics/StreamVLN),
[NaVILA](https://github.com/AnjieCheng/NaVILA),
[Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid), and
[OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) for sharing their
methods and implementations. We are especially grateful to the StreamVLN team,
whose training-data processing and trajectory-adaptation design provided
valuable inspiration for SatNav.

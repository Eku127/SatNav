# SatNav Documentation

**[SatNav](https://openreview.net/forum?id=hOEniyN6hl) is accepted at NeurIPS 2026, Evaluations & Datasets Track (Poster).**

**Continuous-State Vision-and-Language Navigation via Satellite Maps**

SatNav provides scene preparation, trajectory generation, model training, and online evaluation for language-guided navigation over satellite maps. Its SatSim environment renders RGB observations as agents move through continuous geographic space.

![SatNav tasks and navigation episode](../assets/readme/overview-final.png)

## Start here

| Your goal | Guide |
| --- | --- |
| Understand the system | [Architecture and navigation loop](concepts/OVERVIEW.md) · [SatSim observations](concepts/SATSIM.md) · [Tasks and metrics](concepts/TASKS_AND_METRICS.md) · [Expert trajectories](concepts/EXPERT_TRAJECTORIES.md) |
| Run your first navigation episode | [Installation](getting-started/INSTALLATION.md) · [Examples](getting-started/EXAMPLES.md) |
| Prepare scenes and episodes | [Dataset](dataset/DATA_DOWNLOAD.md) · [Satellite maps](applications/MAP_DOWNLOAD.md) |
| Train a navigation policy | [Training](training/README.md) |
| Evaluate a model | [Online evaluation](evaluation/README.md) |

## Navigation workflow

![SatNav training and evaluation workflow](../assets/readme/workflow.svg)

Generate expert trajectories for training, then evaluate policies through the same SatSim observation and action interface.

## Resources

- [SatNav Episodes](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1): 118,494 episodes across 59 scenes, with boundary, landmark, and route navigation tasks.
- [Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo): released SatNav baseline checkpoints.
- [SwiftVLN](https://github.com/Eku127/SwiftVLN): SwiftVLN setup on SatNav.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

Installation <getting-started/INSTALLATION>
Examples <getting-started/EXAMPLES>
Workflow guide <README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: System Principles

Architecture and navigation loop <concepts/OVERVIEW>
SatSim observations <concepts/SATSIM>
Tasks and metrics <concepts/TASKS_AND_METRICS>
Expert trajectories <concepts/EXPERT_TRAJECTORIES>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Dataset

Download episodes <dataset/DATA_DOWNLOAD>
Data format <dataset/DATASET_FORMAT>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: SatSim

Prepare satellite maps <applications/MAP_DOWNLOAD>
Viewer <applications/VIEWER>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Training

Training overview <training/README>
Expert trajectories <applications/TRAJECTORY_GENERATION>
Classic baselines <training/CLASSIC>
StreamVLN <training/vlm/STREAMVLN>
NaVILA <training/vlm/NAVILA>
Uni-NaVid <training/vlm/UNINAVID>
OpenFly <training/vlm/OPENFLY>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Evaluation

Online evaluation <evaluation/README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference & Development

Core API <core/CORE_API>
Add a model <development/MODEL_INTEGRATION>
```

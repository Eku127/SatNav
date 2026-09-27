# SatNav Documentation

This page is the main entry point for SatNav documentation. Choose the path
that matches your goal; you do not need to read every document in filename
order.

If this is your first time using SatNav, begin with
[Installation](getting-started/INSTALLATION.md) and run a repository example.
Training and full evaluation additionally require SatNav-v0.1 episodes,
GeoTIFF scenes, and model resources.

## System principles

- [Architecture and navigation loop](concepts/OVERVIEW.md)
- [SatSim observations](concepts/SATSIM.md)
- [Tasks and metrics](concepts/TASKS_AND_METRICS.md)
- [Expert trajectories](concepts/EXPERT_TRAJECTORIES.md)

## 1. Run SatNav for the first time

Follow these documents in order:

1. [Installation](getting-started/INSTALLATION.md): install SatNav Core and any
   optional application or classic-baseline dependencies;
2. [Examples](getting-started/EXAMPLES.md): run a rollout with the bundled
   synthetic scene and two example episodes;
3. [Core API](core/CORE_API.md): learn the `Env`, observation, action, episode,
   and metric interfaces;
4. [Data Format](dataset/DATASET_FORMAT.md): understand real episodes, GeoTIFFs,
   and trajectories.

The repository examples require no SatNav-v0.1 download, satellite imagery,
or model checkpoint.

## 2. Prepare SatNav-v0.1 data

For training or full evaluation, prepare data in this order:

1. [Episode Download](dataset/DATA_DOWNLOAD.md): download train, `val_seen`,
   `val_unseen`, and the scene list;
2. [Satellite Scene Download](applications/MAP_DOWNLOAD.md): request prepared
   GeoTIFFs or generate them with your own API credentials;
3. [SatSim Viewer](applications/VIEWER.md): inspect scenes, episode starts, and
   reference paths;
4. [Trajectory Generation](applications/TRAJECTORY_GENERATION.md): generate
   RGB frames and expert actions for offline training;
5. [Data Format](dataset/DATASET_FORMAT.md): verify the public episode and
   trajectory schemas.

Online evaluation needs episodes, GeoTIFFs, and a model checkpoint; it does not
require offline trajectories. Training Seq2Seq, CMA, or a VLM baseline does.

## 3. Train models

Begin with [Model Training](training/README.md) for the shared workflow,
required inputs, and the choice between Classic and VLM baselines.

### Classic baselines

[Classic Baselines](training/CLASSIC.md) covers Seq2Seq and CMA data
preparation, vocabulary construction, training, checkpoint validation, and
single-/multi-GPU evaluation. Use the tiny example quickstart first when
validating a new environment. Random and ReferenceFollower have no trainable
parameters.

### VLM baselines

Each VLM uses its own Python environment and model resources:

| Baseline | Training and evaluation guide |
| --- | --- |
| StreamVLN | [StreamVLN Baseline](training/vlm/STREAMVLN.md) |
| NaVILA | [NaVILA Baseline](training/vlm/NAVILA.md) |
| Uni-NaVid | [Uni-NaVid Baseline](training/vlm/UNINAVID.md) |
| OpenFly | [OpenFly Baseline](training/vlm/OPENFLY.md) |

Released SatNav checkpoints are available in the
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo).

Do not share PyTorch, Transformers, or FlashAttention environments across VLM
baselines. Store models, datasets, upstream checkouts, and output paths in the
baseline's Git-ignored `.local/env.sh`.

## 4. Evaluate models

[Evaluation](evaluation/README.md) documents the common online rollout,
episode selection, multi-rank sharding, output format, aggregation, error
handling, and resume behavior used by every Classic and VLM baseline.

Recommended evaluation sequence:

1. run a single-GPU smoke with a few episodes and a five-step cap;
2. repeat the same smoke on multiple GPUs to validate sharding and aggregation;
3. run all `val_seen` episodes with a 500-step cap;
4. repeat the full settings on `val_unseen`.

Use the baseline-specific guide for model setup, checkpoints, and launchers.

## 5. Integrate a new model

Read these documents in order:

1. [Core API](core/CORE_API.md): environment, observation, and action contract;
2. [Evaluation](evaluation/README.md): evaluator ownership and result format;
3. [Model Integration](development/MODEL_INTEGRATION.md): implement a
   `PolicyAdapter`, isolated environment, launcher, and local configuration;
4. use a structurally similar [VLM baseline](#vlm-baselines) as a complete
   reference.

New models must use the public `Env` and `PolicyAdapter` interfaces. The
adapter owns model state, frame history, tokenization, processors, and action
queues.

## 6. Find documentation by task

| Goal | Document |
| --- | --- |
| Install SatNav | [Installation](getting-started/INSTALLATION.md) |
| Run repository examples | [Examples](getting-started/EXAMPLES.md) |
| Use the Python API | [Core API](core/CORE_API.md) |
| Understand data fields | [Data Format](dataset/DATASET_FORMAT.md) |
| Download episodes | [Episode Download](dataset/DATA_DOWNLOAD.md) |
| Prepare GeoTIFFs | [Satellite Scene Download](applications/MAP_DOWNLOAD.md) |
| Inspect scenes and episodes | [SatSim Viewer](applications/VIEWER.md) |
| Generate offline trajectories | [Trajectory Generation](applications/TRAJECTORY_GENERATION.md) |
| Choose a training path | [Model Training](training/README.md) |
| Train Seq2Seq or CMA | [Classic Baselines](training/CLASSIC.md) |
| Train or evaluate a VLM | [VLM baselines](#vlm-baselines) |
| Run common online evaluation | [Evaluation](evaluation/README.md) |
| Integrate a new model | [Model Integration](development/MODEL_INTEGRATION.md) |

## 7. Documentation conventions

- Commands run from the SatNav repository root unless stated otherwise.
- `/path/to/...` is a placeholder for a machine-local path.
- Store datasets, models, checkouts, and output paths in `.local/env.sh` or
  another Git-ignored configuration.
- Smoke commands validate the pipeline; they are not performance results.
- Official benchmark runs use the full split, a 500-step cap, and a complete
  checkpoint for the selected baseline.
- Each procedural guide ends with troubleshooting for that workflow.

# Training Models with SatNav

This guide introduces the common SatNav training workflow and uses the repository's tiny example to run an end-to-end training job. The quickstart exports offline trajectories, trains Seq2Seq and CMA, saves checkpoints, and evaluates both policies in SatSim.

Before starting, read [Installation](../getting-started/INSTALLATION.md) and [Dataset Format](../dataset/DATASET_FORMAT.md). If you only need to connect an existing model to online evaluation, go directly to [Model Integration](../development/MODEL_INTEGRATION.md).

## 1. Training workflow

SatNav separates offline training from online evaluation:

```text
Episode + GeoTIFF
        │
        ▼
Trajectory Generation
        │
        ├── annotations.json
        └── images/<episode>/rgb/*.jpg
                    │
                    ▼
             Offline Training
                    │
                    ▼
                Checkpoint
                    │
                    ▼
            Online SatSim Evaluation
```

Training data consists of Episodes, navigation instructions, expert actions, and an RGB observation for each step. Training reads the exported JPEG files and actions directly, so SatSim does not run inside every batch. After training, the checkpoint predicts actions step by step in the online environment, where SatNav computes Success, SPL, Distance to Goal, and other metrics.

The repository provides two categories of trainable baselines:

| Category | Models | Environment | Training entry point |
| --- | --- | --- | --- |
| Classic | Seq2Seq, CMA | SatNav classic environment | `run.py`, `scripts/seq2seq/`, `scripts/cma/` |
| VLM | StreamVLN, NaVILA, Uni-NaVid, OpenFly | One isolated environment per model | `baselines/vlm/<name>/scripts/train.sh` |

Random and ReferenceFollower have no learnable parameters. Classic and VLM baselines can consume the same SatNav trajectory export, while each model owns its sampler, processor, optimizer, and checkpoint format. Because the four VLM dependency stacks are incompatible, use the dedicated environment for the model you select.

## 2. Choose a training path

### 2.1 Classic baselines

Classic baselines are the fastest way to validate the SatNav data, training, and online evaluation pipeline. They also provide lightweight reference methods for new work. Seq2Seq and CMA share the classic environment, trajectory format, and evaluation interface:

| Model | Main architecture | Starting point |
| --- | --- | --- |
| Seq2Seq | Instruction encoder + RGB encoder + recurrent policy | Randomly initialized model with local vocabulary and embeddings |
| CMA | Cross-modal attention + recurrent policy | Randomly initialized model with local vocabulary and embeddings |

For a first run, complete the tiny example in Sections 4–6. It exports example trajectories and performs real training and evaluation for both models. Once SatNav-v0.1 is available, follow [Classic Baselines](CLASSIC.md) for data validation, full training, checkpoint inspection, and single- or multi-GPU evaluation.

Random and ReferenceFollower are evaluation-only policies and are not part of the training workflow.

### 2.2 VLM baselines

VLM baselines support reproducing existing vision-language navigation models and fine-tuning large models on SatNav trajectories. Each integration has its own Python environment, upstream source or runtime implementation, model assets, and launcher:

| Baseline | Model source | Starting point | Full guide |
| --- | --- | --- | --- |
| StreamVLN | Pinned external StreamVLN checkout | Official StreamVLN checkpoint or LLaVA-Video base model | [StreamVLN Baseline](vlm/STREAMVLN.md) |
| NaVILA | Pinned external NaVILA checkout | NaVILA SFT or pretraining checkpoint | [NaVILA Baseline](vlm/NAVILA.md) |
| Uni-NaVid | Pinned external Uni-NaVid checkout | Uni-NaVid checkpoint and EVA weights | [Uni-NaVid Baseline](vlm/UNINAVID.md) |
| OpenFly | Pinned OpenFly runtime bundled with SatNav | Complete Hugging Face checkpoint, or native `.pt` weights with a processor | [OpenFly Baseline](vlm/OPENFLY.md) |

After choosing a VLM, start with its guide; the Classic tiny example is not required. Each guide covers environment setup, model assets, local paths, data validation, smoke training, full training, and online evaluation.

You can keep several VLM environments on the same machine, but do not share PyTorch, Transformers, or FlashAttention installations between them. Store model paths, dataset paths, upstream checkouts, and output locations in the baseline's Git-ignored `.local/env.sh` file.

## 3. Training inputs

A full training run normally requires:

| Input | Purpose | Preparation |
| --- | --- | --- |
| Episode JSON | Instructions, start poses, goals, and reference paths | [Download Episode Data](../dataset/DATA_DOWNLOAD.md) |
| GeoTIFF scenes | Render RGB observations along expert trajectories | [Download Satellite Scenes](../applications/MAP_DOWNLOAD.md) |
| Offline trajectories | `annotations.json`, RGB frames, and actions | [Generate Trajectory Data](../applications/TRAJECTORY_GENERATION.md) |
| Model assets | Vocabulary, embeddings, processor, or pretrained weights | Follow the selected baseline guide |
| Training configuration | Data paths, architecture, and optimization settings | `configs/baselines/` or the VLM-specific configuration |

The basic offline trajectory layout is:

```text
trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

`annotations.json` stores the Episode instruction, action sequence, frame indices, and logical scene information. Each trajectory begins with the initial observation and ends with a `STOP` action. See [Dataset Format — Offline trajectories](../dataset/DATASET_FORMAT.md#6-offline-trajectories) for field definitions.

Episode, scene, trajectory, checkpoint, and output paths in a training configuration must refer to the same experiment data. Put machine-specific paths in a Git-ignored `.local/env.sh` or `configs/local_*.yaml` file rather than a public configuration.

## 4. Tiny example

The repository includes a directly runnable example:

```text
applications/resources/
├── map.tif
├── satnav_example_episodes.json
└── satnav_example_task.yaml
```

It contains two example Episodes and a procedurally generated synthetic GeoTIFF. The quickstart exports offline trajectories for these Episodes and trains models with:

```text
configs/baselines/seq2seq_offline_train.yaml
configs/baselines/cma_offline_train.yaml
```

The tiny example verifies that the following pipeline works in your environment:

- vocabulary and instruction embedding construction;
- SatSim trajectory generation and RGB export;
- offline dataset loading;
- forward pass, backward pass, and optimizer update;
- checkpoint saving and strict reload;
- online Seq2Seq and CMA rollouts with metric output.

This dataset is intentionally small and only validates the training pipeline. Its loss and navigation metrics do not represent performance on the SatNav-v0.1 benchmark.

## 5. Prepare the quickstart environment

Create and activate the classic environment from the repository root:

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic,applications]'
```

If you already followed the [Installation Guide](../getting-started/INSTALLATION.md), do not recreate the environment. For another CUDA version, install matching PyTorch and TorchVision builds.

The quickstart also requires:

- `curl` or `wget` to download GloVe;
- `unzip` to extract `glove.6B.zip`;
- network access to TorchVision weights, or a cached ResNet50 ImageNet checkpoint.

Check the key dependencies:

```bash
python - <<'PY'
import torch
import torchvision
from satnav.core.env import Env
from satnav.training.offline_trainer import OfflineTrainer

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("SatNav training import OK")
PY

command -v unzip
command -v curl || command -v wget
```

## 6. Run the tiny example quickstart

From the repository root, run:

```bash
bash scripts/quickstart_models.sh
```

The script performs these steps in order:

1. builds a vocabulary from the example instructions;
2. downloads or reuses GloVe 6B 50d;
3. creates GloVe embeddings for the example vocabulary;
4. downloads or reuses TorchVision ResNet50 ImageNet weights;
5. exports offline trajectories for two Episodes using the synthetic GeoTIFF;
6. trains and evaluates Seq2Seq;
7. trains and evaluates CMA.

Outputs are written to:

```text
output/quickstart_baselines/
├── artifacts/
│   ├── vocab/
│   │   └── satnav_example_vocab.json
│   ├── glove/
│   └── embeddings/
│       └── satnav_example_glove50d.json.gz
├── trajectory_data/
│   ├── annotations.json
│   ├── summary.json
│   └── images/
├── seq2seq/
│   ├── checkpoints/latest/best.pth
│   └── results/latest/
└── cma/
    ├── checkpoints/latest/best.pth
    └── results/latest/
```

A successful run ends with:

```text
Done. Outputs are under output/quickstart_baselines/
```

Confirm that both checkpoints and evaluation outputs exist:

```bash
test -f output/quickstart_baselines/seq2seq/checkpoints/latest/best.pth
test -f output/quickstart_baselines/cma/checkpoints/latest/best.pth

find output/quickstart_baselines/seq2seq/results/latest -type f
find output/quickstart_baselines/cma/results/latest -type f
```

If `glove.6B.50d.txt` is already available locally, avoid downloading it again:

```bash
LOCAL_GLOVE_TXT=/path/to/glove.6B.50d.txt \
  bash scripts/quickstart_models.sh
```

To run only one model:

```bash
# Run Seq2Seq only
RUN_CMA=0 bash scripts/quickstart_models.sh

# Run CMA only
RUN_SEQ2SEQ=0 bash scripts/quickstart_models.sh
```

To validate data preparation and training without the online evaluation step:

```bash
RUN_EVAL=0 bash scripts/quickstart_models.sh
```

## 7. Troubleshooting

### Why did the GloVe download fail?

Make sure the machine can access the GloVe download URL and has either `curl` or `wget`. If `glove.6B.50d.txt` was downloaded elsewhere, set `LOCAL_GLOVE_TXT` to that file.

### Why did the ResNet50 weight download fail?

The quickstart uses TorchVision's ImageNet-pretrained ResNet50. Make sure the environment can access the PyTorch model download server, or place the weights in the Torch cache for the same user.

### Why is my checkpoint replaced when I rerun the script?

The tiny example uses the fixed `output/quickstart_baselines/` location so it can be rerun as a pipeline check. Move that directory to a separate experiment location before rerunning if you need to keep its output.

### Why are Success or SPL low on the tiny example?

The example has only two Episodes and very few optimizer updates. It validates code, data, checkpoint, and rollout behavior; it is not a model-quality benchmark.

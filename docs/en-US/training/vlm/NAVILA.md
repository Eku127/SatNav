# SatNav NaVILA Baseline

This guide explains how to train NaVILA on SatNav-v0.1 trajectories and run single- or multi-GPU online evaluation in SatSim. NaVILA uses an isolated Python environment and external model source; it does not share an environment with SatNav Core, Classic baselines, or other VLM baselines.

Before starting, read [Installation](../../getting-started/INSTALLATION.md), [Training Models with SatNav](../README.md), and [Evaluation](../../evaluation/README.md).

## 1. Model and SatNav interface

NaVILA generates text actions from a navigation instruction, the current RGB observation, and visual history. The SatNav adapter maps model output to four primitive actions:

| Model output | SatNav action | Environment behavior |
| --- | --- | --- |
| `stop` | `STOP` | End the current Episode |
| `forward` | `MOVE_FORWARD` | Move forward 10 m |
| `left` | `TURN_LEFT` | Turn left 15° |
| `right` | `TURN_RIGHT` | Turn right 15° |

The default `compact` action format requires exactly one action word. The `sentence` format preserves NaVILA's natural-language output. When a sentence specifies a longer distance or larger turn, the adapter expands it into a sequence of SatNav primitive actions.

Inference uses eight frames of observation history. The adapter pads an incomplete history with black frames; for a longer history, it samples uniformly while always retaining the current frame.

SatNav supports upstream revision [`AnjieCheng/NaVILA@76b98f233dd0fff05dfcd69435eec6740febff9d`](https://github.com/AnjieCheng/NaVILA/commit/76b98f233dd0fff05dfcd69435eec6740febff9d). Before using the source or weights, read [UPSTREAM](../../../../baselines/vlm/navila/UPSTREAM.md), [NOTICE](../../../../baselines/vlm/navila/NOTICE), and the [upstream license](../../../../baselines/vlm/navila/LICENSE.upstream).

## 2. Create an isolated environment

From the SatNav repository root:

```bash
conda env create -f baselines/vlm/navila/environment/conda.yml
conda activate satnav-navila
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the FlashAttention 2.5.8 wheel built for Python 3.10, PyTorch 2.3, CUDA 12.x, and the old CXX11 ABI:

```bash
export NAVILA_FLASH_ATTN_WHEEL="${NAVILA_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
python -m pip install "$NAVILA_FLASH_ATTN_WHEEL"
```

If this wheel is incompatible with your platform, obtain a local wheel that exactly matches Python, PyTorch, CUDA, and the ABI, then point `NAVILA_FLASH_ATTN_WHEEL` to it.

Install the remaining dependencies and SatNav:

```bash
python -m pip install -r baselines/vlm/navila/requirements.txt
python -m pip install -e .
```

## 3. Prepare upstream source

Clone NaVILA outside the SatNav repository and check out the supported revision:

```bash
git clone https://github.com/AnjieCheng/NaVILA.git /path/to/NaVILA
git -C /path/to/NaVILA checkout --detach \
  76b98f233dd0fff05dfcd69435eec6740febff9d

python -m pip install --no-deps -e /path/to/NaVILA
bash baselines/vlm/navila/scripts/patch_environment.sh
```

`patch_environment.sh` installs the Transformers and DeepSpeed compatibility files supplied by the pinned upstream revision into the current NaVILA environment. It does not modify the upstream checkout. Do not run it in another VLM environment.

Check the source revision, environment, and entry points:

```bash
git -C /path/to/NaVILA rev-parse HEAD
git -C /path/to/NaVILA status --short

python - <<'PY'
import accelerate
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m pip check
python -m baselines.vlm.navila.dataset --help
python -m baselines.vlm.navila.trainer --help
python -m baselines.vlm.navila.evaluate --help
python -m baselines.vlm.navila.checkpoint --help
command -v python
```

`rev-parse` must print the revision above, and `status --short` must print nothing. Record the result of `command -v python`; use it later as `NAVILA_PYTHON`.

## 4. Prepare SatNav data

### 4.1 Episodes, GeoTIFF scenes, and trajectories

Follow [Download Episode Data](../../dataset/DATA_DOWNLOAD.md), [Download Satellite Scenes](../../applications/MAP_DOWNLOAD.md), and [Generate Trajectory Data](../../applications/TRAJECTORY_GENERATION.md). NaVILA reads JPEG files directly from the trajectory export:

```text
trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            └── ...
```

Set the data paths:

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/scenes
export SATNAV_TRAJECTORY_ROOT=/path/to/trajectory_data
```

First validate the Episode files and 59 GeoTIFF scenes:

```bash
bash scripts/validation/data_validation.sh
```

### 4.2 Validate trajectories

Check annotations, actions, frame counts, and image decoding:

```bash
python -m baselines.vlm.navila.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --expected-episodes 105164 \
  --strict-frames \
  --decode-samples 64
```

The launcher validates the complete train split again before full training. Missing images, out-of-root paths, invalid actions, and frame-count mismatches stop training instead of silently dropping samples.

## 5. Download model assets

Choose a model directory:

```bash
export NAVILA_MODEL_ROOT=/path/to/navila-models
mkdir -p "$NAVILA_MODEL_ROOT"
```

### 5.1 Released SatNav checkpoints

The [SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
provides two NaVILA checkpoints:

- [scratch](https://huggingface.co/Eku127/navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4), trained from the NaVILA pretraining checkpoint;
- [continue](https://huggingface.co/Eku127/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4), trained from the NaVILA SFT checkpoint.

Download either checkpoint with the baseline downloader:

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo Eku127/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4 \
  --model-root "$NAVILA_MODEL_ROOT"
```

Pass the downloaded directory to `--model-path` for evaluation. These releases
contain inference artifacts rather than optimizer and scheduler state.

### 5.2 Training starting points

`continue` training uses the NaVILA SFT checkpoint:

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo a8cheng/navila-llama3-8b-8f \
  --model-root "$NAVILA_MODEL_ROOT"
```

`scratch` training starts from the NaVILA pretraining checkpoint:

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
  --model-root "$NAVILA_MODEL_ROOT"
```

If a Hugging Face repository requires authorization, log in from the current environment first. Add `--endpoint` to the download command when using a mirror.

The resulting layout is:

```text
navila-models/
├── navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4/
├── navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4/
├── navila-llama3-8b-8f/
└── navila-siglip-llama3-8b-v1.5-pretrain/
```

Both directories must contain the complete model configuration, tokenizer, vision tower, multimodal projector, and all weights. A single weight shard or adapter is not sufficient.

## 6. Configure local paths

Copy the local template:

```bash
mkdir -p baselines/vlm/navila/.local
cp baselines/vlm/navila/local.env.example \
  baselines/vlm/navila/.local/env.sh
```

Fill `baselines/vlm/navila/.local/env.sh` with the paths prepared above:

```bash
export NAVILA_REPO="${NAVILA_REPO:-/path/to/NaVILA}"
export NAVILA_PYTHON="${NAVILA_PYTHON:-/path/to/satnav-navila/bin/python}"
export NAVILA_MODEL_ROOT="${NAVILA_MODEL_ROOT:-/path/to/navila-models}"
export NAVILA_PRETRAIN_MODEL="${NAVILA_PRETRAIN_MODEL:-${NAVILA_MODEL_ROOT}/navila-siglip-llama3-8b-v1.5-pretrain}"
export NAVILA_SFT_MODEL="${NAVILA_SFT_MODEL:-${NAVILA_MODEL_ROOT}/navila-llama3-8b-8f}"

export SATNAV_NAVILA_TRAIN_DATA="${SATNAV_NAVILA_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_NAVILA_EVAL_EPISODES="${SATNAV_NAVILA_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_NAVILA_SCENES_DIR="${SATNAV_NAVILA_SCENES_DIR:-/path/to/scenes}"
export SATNAV_NAVILA_OUTPUT="${SATNAV_NAVILA_OUTPUT:-output/baselines/vlm/navila}"
```

Variable sources:

| Variable | Source |
| --- | --- |
| `NAVILA_REPO` | NaVILA checkout from Section 3 |
| `NAVILA_PYTHON` | Output of `command -v python` in Section 3 |
| `NAVILA_MODEL_ROOT` | Model directory selected in Section 5 |
| `NAVILA_PRETRAIN_MODEL` | Downloaded pretraining checkpoint |
| `NAVILA_SFT_MODEL` | Downloaded SFT checkpoint |
| `SATNAV_NAVILA_TRAIN_DATA` | Validated `trajectory_data/` from Section 4 |
| `SATNAV_NAVILA_EVAL_EPISODES` | SatNav-v0.1 evaluation Episode path template |
| `SATNAV_NAVILA_SCENES_DIR` | Validated GeoTIFF directory from Section 4 |
| `SATNAV_NAVILA_OUTPUT` | Training and evaluation output root |

Launchers load this file automatically. An environment variable set explicitly in the shell takes precedence over `.local/env.sh`. Do not commit local data, models, upstream source, output directories, or environment paths.

To use the variables directly in the current shell:

```bash
source baselines/vlm/navila/.local/env.sh
```

## 7. Training configuration and checkpoint validation

Before training, confirm that the SFT checkpoint loads completely:

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path "$NAVILA_SFT_MODEL" \
  --navila-repo "$NAVILA_REPO" \
  --device cuda:0
```

For `scratch`, repeat the check with `$NAVILA_PRETRAIN_MODEL`. Missing, incompatible, or unloadable model components cause an explicit error.

Defaults are in `baselines/vlm/navila/configs/train.yaml`:

| Parameter | Default |
| --- | ---: |
| Video frames | 8 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | `3e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 8 |

Training uses DeepSpeed ZeRO-2 from `baselines/vlm/navila/configs/zero2.json`. By default, it updates the vision tower, multimodal projector, and language model. A bounded smoke run can use `--train-components projector` to validate only the projector update path.

In training commands, `--max-steps` means optimizer steps. In evaluation commands, the same name means the maximum primitive actions per Episode.

## 8. Smoke training

Run two optimizer steps to validate data loading, parameter updates, and complete checkpoint saving:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_SFT_MODEL" \
  --output-dir output/baselines/vlm/navila/train/continue-smoke \
  --max-steps 2 \
  --save-steps 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --warmup-ratio 0 \
  --dataloader-workers 0 \
  --train-components projector \
  --max-samples 16
```

With eight GPUs, `--max-samples 16` provides two complete global microbatches. If you change the GPU count or batch size, keep the sample limit at least `GPU count × per-device batch size`.

After training, confirm that the output reloads and the projector parameters changed from the starting model:

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path output/baselines/vlm/navila/train/continue-smoke \
  --compare-model "$NAVILA_SFT_MODEL" \
  --navila-repo "$NAVILA_REPO" \
  --device cuda:0
```

## 9. Full training

Continue from the NaVILA SFT checkpoint:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_SFT_MODEL" \
  --output-dir output/baselines/vlm/navila/train/continue-v0-1
```

Start from the NaVILA pretraining checkpoint:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh scratch \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_PRETRAIN_MODEL" \
  --output-dir output/baselines/vlm/navila/train/scratch-v0-1
```

If GPU memory is insufficient, reduce `--batch-size` first and increase `--gradient-accumulation` to retain the desired effective batch size. The current launcher has no training-resume option; use an independent output directory for each full run.

## 10. Configure online evaluation

The training output directory is a complete checkpoint. Set the checkpoint to evaluate:

```bash
export NAVILA_CHECKPOINT=/path/to/trained-navila-checkpoint
```

Check Episode, scene, and output path resolution:

```bash
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run validates configuration, data selection, and the output location without loading the model or interacting with the environment.

## 11. Single-GPU smoke evaluation

Select four Episodes with at most five steps each:

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/5steps-1rank \
  --fail-on-episode-error
```

Result layout:

```text
<output-dir>/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

Confirm that `summary.json` has `status` equal to `complete`, `error_episode_count` equal to 0, and includes `distance_to_goal`, `success`, `oracle_success`, `spl`, and `path_length`.

## 12. Multi-GPU evaluation

Each rank reads the complete Episode file. The shared evaluator handles deterministic sorting and stride sharding. A two-GPU smoke run is:

```bash
CUDA_DEVICES=0,1 \
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

After every rank exits successfully, the launcher writes the aggregated `summary.json`.

## 13. Full evaluation

Evaluate the complete `val_seen` split with a 500-step limit:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

After completion, change the split and output directory to `val_unseen` and run again. `{split}` in `SATNAV_NAVILA_EVAL_EPISODES` is replaced automatically.

To continue an interrupted evaluation, rerun with exactly the same parameters and output directory and add `--resume`. Resume skips Episodes already written by the current rank. Use a new output directory after changing the checkpoint, data, seed, GPU count, or evaluation parameters.

## 14. Troubleshooting

### Why does the launcher report a NaVILA revision mismatch?

Make sure `NAVILA_REPO` points to the correct checkout, then restore the supported revision:

```bash
git -C "$NAVILA_REPO" checkout --detach \
  76b98f233dd0fff05dfcd69435eec6740febff9d
git -C "$NAVILA_REPO" status --short
```

`status --short` must print nothing. Do not store models, logs, or other local files in this checkout.

### Why can FlashAttention not be imported?

The wheel must match Python, PyTorch, CUDA, and the CXX11 ABI. In a clean environment, install PyTorch and FlashAttention before the remaining dependencies. Do not reuse another VLM environment.

### Why did the environment patch fail?

Confirm that the environment uses the pinned Transformers and DeepSpeed versions and that `NAVILA_REPO` is at the supported revision. Run `patch_environment.sh` only after dependency installation.

### Why does checkpoint loading fail?

Use a complete NaVILA checkpoint, not an isolated projector, adapter, or partial weight set. The model directory must include tokenizer, vision tower, multimodal projector, and language-model files.

### Why does trajectory validation report a frame-count mismatch?

Check that each `video` path in `annotations.json` resolves under the current `images/` directory and that trajectory generation completed. Do not combine annotations and images from different exports.

### Why is model loading delayed after training starts?

The launcher validates the complete SatNav-v0.1 trajectory before loading the model. This can take time on network or slow storage; keep annotations and JPEG files on fast local storage when possible.

### Why were some Episodes skipped after restarting evaluation?

With `--resume`, the evaluator skips Episodes already completed in the output directory. Reuse that directory only when continuing exactly the same evaluation; otherwise, choose a new `--output-dir`.

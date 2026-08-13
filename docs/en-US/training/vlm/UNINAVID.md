# SatNav Uni-NaVid Baseline

This guide explains how to train Uni-NaVid on SatNav-v0.1 trajectories and run single- or multi-GPU online evaluation in SatSim. Uni-NaVid uses an isolated Python environment and external model source; it does not share an environment with SatNav Core, Classic baselines, or other VLM baselines.

Before starting, read [Installation](../../getting-started/INSTALLATION.md), [Training Models with SatNav](../README.md), and [Evaluation](../../evaluation/README.md).

## 1. Model and SatNav interface

Uni-NaVid generates up to four text actions from the navigation instruction and a stream of RGB observations. The SatNav adapter maps its output to four primitive actions:

| Model output | SatNav action | Environment behavior |
| --- | --- | --- |
| `stop` | `STOP` | End the current Episode |
| `forward` | `MOVE_FORWARD` | Move forward 10 m |
| `left` | `TURN_LEFT` | Turn left 15° |
| `right` | `TURN_RIGHT` | Turn right 15° |

Generated actions enter a queue and execute one at a time through the SatNav interaction interface. During evaluation, the adapter continuously maintains new observations and the navigation feature cache, then resets all model state at the start of each Episode.

SatNav supports upstream revision [`jzhzhang/Uni-NaVid@79ef5ea3fea14c205342d1ab070563d84c7a966a`](https://github.com/jzhzhang/Uni-NaVid/commit/79ef5ea3fea14c205342d1ab070563d84c7a966a). Before using the source or weights, read [UPSTREAM](../../../../baselines/vlm/uninavid/UPSTREAM.md), [NOTICE](../../../../baselines/vlm/uninavid/NOTICE), and the [upstream license](../../../../baselines/vlm/uninavid/LICENSE.upstream).

## 2. Create an isolated environment

From the SatNav repository root:

```bash
conda env create -f baselines/vlm/uninavid/environment/conda.yml
conda activate satnav-uninavid
export PYTHONNOUSERSITE=1

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install a FlashAttention wheel compatible with Python 3.9, PyTorch 2.5, CUDA, and the machine ABI:

```bash
export UNINAVID_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl
test -f "$UNINAVID_FLASH_ATTN_WHEEL"
python -m pip install "$UNINAVID_FLASH_ATTN_WHEEL"
```

Install the remaining dependencies and SatNav:

```bash
python -m pip install -r baselines/vlm/uninavid/requirements.txt
python -m pip install -e .
python -m pip check
```

The Uni-NaVid adapter reads JPEG files directly from SatNav trajectories, so images do not need to be converted to MP4 and Decord is not required. The upstream source is loaded by path in later steps; do not install it as a Python package.

Check the key packages and entry points:

```bash
python - <<'PY'
import accelerate
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m baselines.vlm.uninavid.dataset --help
python -m baselines.vlm.uninavid.trainer --help
python -m baselines.vlm.uninavid.evaluate --help
python -m baselines.vlm.uninavid.checkpoint --help
command -v python
```

Record the output of `command -v python`; use it later as `UNINAVID_PYTHON`.

## 3. Prepare upstream source

Clone Uni-NaVid outside the SatNav repository and check out the supported revision:

```bash
git clone https://github.com/jzhzhang/Uni-NaVid.git /path/to/Uni-NaVid
git -C /path/to/Uni-NaVid checkout --detach \
  79ef5ea3fea14c205342d1ab070563d84c7a966a
```

Confirm the revision and clean worktree:

```bash
git -C /path/to/Uni-NaVid rev-parse HEAD
git -C /path/to/Uni-NaVid status --short
```

Training and evaluation launchers verify both conditions. Do not modify this checkout or switch it to another Uni-NaVid revision while reusing existing outputs.

## 4. Prepare SatNav data

### 4.1 Episodes, GeoTIFF scenes, and trajectories

Follow [Download Episode Data](../../dataset/DATA_DOWNLOAD.md), [Download Satellite Scenes](../../applications/MAP_DOWNLOAD.md), and [Generate Trajectory Data](../../applications/TRAJECTORY_GENERATION.md). Uni-NaVid reads JPEG files directly from the trajectory export:

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

Check Uni-NaVid's four-action windows and action/frame alignment:

```bash
python -m baselines.vlm.uninavid.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --expected-episodes 105164 \
  --expected-samples 1399366 \
  --strict-frames \
  --decode-samples 32
```

The launcher validates the complete train split again before full training. Missing images, out-of-root paths, invalid actions, and insufficient frame counts stop training rather than silently dropping samples.

## 5. Download model assets

Choose a model directory and set the upstream path:

```bash
export UNINAVID_REPO=/path/to/Uni-NaVid
export UNINAVID_MODEL_ROOT=/path/to/uninavid-models
mkdir -p "$UNINAVID_MODEL_ROOT"
```

### 5.1 Released SatNav checkpoints

The [SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
provides two Uni-NaVid checkpoints:

- [scratch](https://huggingface.co/Eku127/uninavid-satnav-scratch-1ep-lr1e-5), trained from Vicuna 7B v1.5;
- [continue](https://huggingface.co/Eku127/uninavid-satnav-continue-1ep-lr1e-5), trained from the Uni-NaVid checkpoint.

Download either checkpoint into `UNINAVID_MODEL_ROOT`:

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  Eku127/uninavid-satnav-continue-1ep-lr1e-5 \
  --local-dir "$UNINAVID_MODEL_ROOT/uninavid-satnav-continue-1ep-lr1e-5"
```

Pass the downloaded directory to `--model-path` for evaluation. The EVA weight
and processor described below are still required. These releases contain
inference artifacts rather than optimizer and scheduler state.

The released checkpoints achieve the following reference results on the full
splits with a 500-step cap.

| Checkpoint | Split | Episodes | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | `val_seen` | 4,574 | 174.68 | 60.43 | 25.12 | 24.81 |
| Scratch | `val_unseen` | 8,756 | 228.46 | 49.94 | 20.36 | 20.00 |
| Continue | `val_seen` | 4,574 | 87.11 | 68.17 | 49.69 | 49.15 |
| Continue | `val_unseen` | 8,756 | 149.85 | 55.85 | 36.72 | 36.29 |

### 5.2 Training starting point and shared assets

Download the complete Uni-NaVid checkpoint and EVA weights:

```bash
bash baselines/vlm/uninavid/scripts/download.sh \
  --source-dir "$UNINAVID_REPO" \
  --model-root "$UNINAVID_MODEL_ROOT" \
  --model-revision main
```

If the Hugging Face repository requires authorization, log in from the current environment and rerun the command. The downloader reuses completed files and checks both the upstream source revision and EVA file.

The resulting layout is:

```text
uninavid-models/
├── eva_vit_g.pth
├── uninavid-satnav-scratch-1ep-lr1e-5/
├── uninavid-satnav-continue-1ep-lr1e-5/
└── huggingface/
    └── Uni-NaVid/
        └── uninavid-7b-full-224-video-fps-1-grid-2/
```

The image processor remains in the upstream source:

```text
Uni-NaVid/uninavid/processor/clip-patch14-224/
```

## 6. Configure local paths

Copy the local template:

```bash
mkdir -p baselines/vlm/uninavid/.local
cp baselines/vlm/uninavid/local.env.example \
  baselines/vlm/uninavid/.local/env.sh
```

Fill `baselines/vlm/uninavid/.local/env.sh` with the paths prepared above:

```bash
export UNINAVID_REPO="${UNINAVID_REPO:-/path/to/Uni-NaVid}"
export UNINAVID_MODEL_ROOT="${UNINAVID_MODEL_ROOT:-/path/to/uninavid-models}"
export UNINAVID_MODEL="${UNINAVID_MODEL:-/path/to/uninavid-models/huggingface/Uni-NaVid/uninavid-7b-full-224-video-fps-1-grid-2}"
export UNINAVID_EVA="${UNINAVID_EVA:-/path/to/uninavid-models/eva_vit_g.pth}"
export UNINAVID_PROCESSOR="${UNINAVID_PROCESSOR:-${UNINAVID_REPO}/uninavid/processor/clip-patch14-224}"
export UNINAVID_PYTHON="${UNINAVID_PYTHON:-/path/to/satnav-uninavid/bin/python}"
export UNINAVID_FLASH_ATTN_WHEEL="${UNINAVID_FLASH_ATTN_WHEEL:-/path/to/flash_attn.whl}"

export SATNAV_UNINAVID_TRAIN_DATA="${SATNAV_UNINAVID_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_UNINAVID_EVAL_EPISODES="${SATNAV_UNINAVID_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_UNINAVID_SCENES_DIR="${SATNAV_UNINAVID_SCENES_DIR:-/path/to/scenes}"
export SATNAV_UNINAVID_OUTPUT="${SATNAV_UNINAVID_OUTPUT:-output/baselines/vlm/uninavid}"
```

Variable sources:

| Variable | Source |
| --- | --- |
| `UNINAVID_REPO` | Uni-NaVid checkout from Section 3 |
| `UNINAVID_MODEL_ROOT` | Model directory selected in Section 5 |
| `UNINAVID_MODEL` | Downloaded complete Uni-NaVid checkpoint |
| `UNINAVID_EVA` | Downloaded `eva_vit_g.pth` |
| `UNINAVID_PROCESSOR` | `clip-patch14-224/` in the upstream source |
| `UNINAVID_PYTHON` | Output of `command -v python` in Section 2 |
| `UNINAVID_FLASH_ATTN_WHEEL` | FlashAttention wheel installed in Section 2 |
| `SATNAV_UNINAVID_TRAIN_DATA` | Validated `trajectory_data/` from Section 4 |
| `SATNAV_UNINAVID_EVAL_EPISODES` | SatNav-v0.1 evaluation Episode path template |
| `SATNAV_UNINAVID_SCENES_DIR` | Validated GeoTIFF directory from Section 4 |
| `SATNAV_UNINAVID_OUTPUT` | Training and evaluation output root |

Launchers load this file automatically. An environment variable set explicitly in the shell takes precedence over `.local/env.sh`. Do not commit local data, models, upstream source, output directories, or environment paths.

To use the variables directly in the current shell:

```bash
source baselines/vlm/uninavid/.local/env.sh
```

## 7. Training configuration and checkpoint validation

Before training, confirm that the complete checkpoint loads:

```bash
python -m baselines.vlm.uninavid.checkpoint \
  --model-path "$UNINAVID_MODEL" \
  --uninavid-repo "$UNINAVID_REPO" \
  --eva-path "$UNINAVID_EVA" \
  --processor-path "$UNINAVID_PROCESSOR" \
  --device cuda:0
```

This command checks the checkpoint, tokenizer, vision tower, and FlashAttention. Missing, incompatible, or shape-mismatched weights cause an explicit error.

Defaults are in `baselines/vlm/uninavid/configs/train.yaml`:

| Parameter | Default |
| --- | ---: |
| Action window | 4 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | `1e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 2 |

Training uses DeepSpeed ZeRO-1 from `baselines/vlm/uninavid/configs/zero1.json`. It updates the language model and multimodal projector while keeping the vision tower frozen. At completion, the launcher reloads the full output to confirm that it can be evaluated.

In training commands, `--max-steps` means optimizer steps. In evaluation commands, the same name means the maximum primitive actions per Episode.

## 8. Smoke training

Run two optimizer steps to validate data loading, model updates, and checkpoint saving:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/smoke \
  --max-steps 2 \
  --save-steps 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --warmup-ratio 0 \
  --dataloader-workers 0 \
  --max-samples 16 \
  --disable-augmentation
```

With eight GPUs, `--max-samples 16` provides two complete global microbatches. If you change the GPU count or batch size, keep the sample limit at least `GPU count × per-device batch size`.

After a successful run, the output directory contains a directly evaluable model, training logs, and `checkpoint-*`. The launcher also confirms that the language model and projector changed while the frozen vision tower did not.

## 9. Full training and resume

Train on the complete SatNav-v0.1 trajectory export:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/v0-1
```

The default is one epoch. If GPU memory is insufficient, reduce `--batch-size` and use `--gradient-accumulation` to adjust the effective batch size.

If training is interrupted and `checkpoint-*` exists in the output directory, rerun with exactly the same parameters and output directory and add `--resume`:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/v0-1 \
  --resume
```

Resume is only for the same training run. Use a new output directory after changing the model, data, configuration, GPU count, or sampling parameters.

## 10. Configure online evaluation

The training output directory is a complete checkpoint. Set the checkpoint to evaluate:

```bash
export UNINAVID_CHECKPOINT=/path/to/trained-uninavid-checkpoint
```

Check Episode, scene, and output path resolution:

```bash
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
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
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/5steps-1rank \
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
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

After every rank exits successfully, the launcher writes the aggregated `summary.json`.

## 13. Full evaluation

Evaluate the complete `val_seen` split with a 500-step limit:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

After completion, change the split and output directory to `val_unseen` and run again. `{split}` in `SATNAV_UNINAVID_EVAL_EPISODES` is replaced automatically.

To continue an interrupted evaluation, rerun with exactly the same parameters and output directory and add `--resume`. Resume skips Episodes already written by the current rank. Use a new output directory after changing the checkpoint, data, seed, GPU count, or evaluation parameters.

## 14. Troubleshooting

### Why does the launcher report a Uni-NaVid revision mismatch or dirty worktree?

Make sure `UNINAVID_REPO` points to the correct checkout, then restore the supported revision:

```bash
git -C "$UNINAVID_REPO" checkout --detach \
  79ef5ea3fea14c205342d1ab070563d84c7a966a
git -C "$UNINAVID_REPO" status --short
```

`status --short` must print nothing. Do not store models, logs, or other local files in this checkout.

### Why did the model download fail?

Confirm that the machine can access Hugging Face and whether the model repository requires login or acceptance of usage terms. After authorization, rerun the same command to reuse files already downloaded.

### Why can it not find EVA or the processor?

`UNINAVID_EVA` must point to the downloaded `eva_vit_g.pth`. `UNINAVID_PROCESSOR` must point to `clip-patch14-224/` containing `preprocessor_config.json` in the pinned upstream checkout.

### Why does checkpoint loading fail?

Use a complete Uni-NaVid checkpoint, not a standalone adapter or partial weights. The model directory must include the configuration, tokenizer, weight index, and every weight shard.

### Why does trajectory validation report too few frames?

Check that each `video` path in `annotations.json` resolves under the current `images/` directory and that trajectory generation completed. Do not combine annotations and images from different exports.

### Why is model loading delayed after training starts?

The launcher validates the complete SatNav-v0.1 trajectory before loading the model. This can take time on network or slow storage; keep annotations and JPEG files on fast local storage when possible.

### Why was resume rejected?

Confirm that `checkpoint-*` exists in the output directory and that the model, data, configuration, GPU count, and sampling parameters exactly match the initial run. Use a new output directory for a changed experiment.

### Why were some Episodes skipped after restarting evaluation?

With `--resume`, the evaluator skips Episodes already completed in the output directory. Reuse that directory only when continuing exactly the same evaluation; otherwise, choose a new `--output-dir`.

# SatNav StreamVLN Baseline

This guide explains how to train StreamVLN on SatNav-v0.1 trajectories and run single- or multi-GPU online evaluation in SatSim. StreamVLN uses an isolated Python environment and external model source; it does not share an environment with SatNav Core, Classic baselines, or other VLM baselines.

Before starting, read [Installation](../../getting-started/INSTALLATION.md), [Training Models with SatNav](../README.md), and [Evaluation](../../evaluation/README.md).

## 1. Model and SatNav interface

StreamVLN generates symbolic actions from a navigation instruction, the current RGB observation, and visual history. The SatNav adapter maps model output to four primitive actions:

| Model output | SatNav action | Environment behavior |
| --- | --- | --- |
| `STOP` | `STOP` | End the current Episode |
| `↑` | `MOVE_FORWARD` | Move forward 10 m |
| `←` | `TURN_LEFT` | Turn left 15° |
| `→` | `TURN_RIGHT` | Turn right 15° |

The model can generate an action chunk. The adapter queues these actions and executes them one at a time through the SatNav interaction interface. By default, training builds a visual window from 32 observations, retains eight history samples, and predicts four future actions.

SatNav supports upstream revision [`Eku127/StreamVLN@60476e81f4c01b29f1a51a7469f1cb4addbc1d62`](https://github.com/Eku127/StreamVLN/commit/60476e81f4c01b29f1a51a7469f1cb4addbc1d62). Before using the source or weights, read [UPSTREAM](../../../../baselines/vlm/streamvln/UPSTREAM.md) and [NOTICE](../../../../baselines/vlm/streamvln/NOTICE).

## 2. Create an isolated environment

From the SatNav repository root:

```bash
conda env create -f baselines/vlm/streamvln/environment/conda.yml
conda activate satnav-streamvln

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install a FlashAttention wheel compatible with Python 3.9, PyTorch 2.5, CUDA, and the machine ABI:

```bash
export STREAMVLN_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl
test -f "$STREAMVLN_FLASH_ATTN_WHEEL"
python -m pip install "$STREAMVLN_FLASH_ATTN_WHEEL"
```

Install the remaining dependencies and SatNav:

```bash
python -m pip install -r baselines/vlm/streamvln/requirements.txt
python baselines/vlm/streamvln/scripts/normalize_decord_wheel.py
python -m pip install -e .
python -m pip check
```

`normalize_decord_wheel.py` fixes a known metadata issue in the official Decord 0.6.0 wheel; it does not alter video decoding behavior.

Check the key packages and entry points:

```bash
python - <<'PY'
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m baselines.vlm.streamvln.dataset --help
python -m baselines.vlm.streamvln.trainer --help
python -m baselines.vlm.streamvln.evaluate --help
command -v python
```

Record the output of `command -v python`; use it later as `STREAMVLN_PYTHON`.

## 3. Prepare upstream source

Clone StreamVLN outside the SatNav repository and check out the supported revision:

```bash
git clone https://github.com/Eku127/StreamVLN.git /path/to/StreamVLN
git -C /path/to/StreamVLN checkout \
  60476e81f4c01b29f1a51a7469f1cb4addbc1d62
```

Training and evaluation launchers verify this revision. Do not switch the same checkout to another StreamVLN version and continue using existing outputs.

## 4. Prepare SatNav data

### 4.1 Episodes, GeoTIFF scenes, and trajectories

Follow [Download Episode Data](../../dataset/DATA_DOWNLOAD.md), [Download Satellite Scenes](../../applications/MAP_DOWNLOAD.md), and [Generate Trajectory Data](../../applications/TRAJECTORY_GENERATION.md). StreamVLN uses the same trajectory export as the other baselines:

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

Validate the complete train split:

```bash
mkdir -p output/baselines/vlm/streamvln

python scripts/validation/validate_trajectory_output.py \
  --annotations "$SATNAV_TRAJECTORY_ROOT/annotations.json" \
  --output-root "$SATNAV_TRAJECTORY_ROOT" \
  --source-episodes "$SATNAV_DATA_ROOT/episodes/train/all_episodes.json" \
  --generation-config applications/episode_processing/configs/trajectory_generation.yaml \
  --scenes-dir "$SATNAV_SCENES_DIR" \
  --expected-count 105164 \
  --decode-images \
  --report output/baselines/vlm/streamvln/data_validation.json
```

Then check StreamVLN's 32-frame segmentation and action/frame alignment:

```bash
python -m baselines.vlm.streamvln.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --num-frames 32 \
  --strict-frames
```

The validator reports the usable Episode and training sample counts. Resolve missing images, out-of-root paths, invalid actions, or insufficient frame counts before full training.

## 5. Download model assets

Choose a model directory:

```bash
export STREAMVLN_MODEL_ROOT=/path/to/streamvln-models
mkdir -p "$STREAMVLN_MODEL_ROOT"
```

`continue` training starts from the official StreamVLN checkpoint:

```bash
bash baselines/vlm/streamvln/scripts/download.sh \
  --model-root "$STREAMVLN_MODEL_ROOT"
```

`scratch` training needs the LLaVA-Video base model. This directory can also supply a tokenizer for a checkpoint that does not include one. Both training and evaluation require a local SigLIP vision tower:

```bash
bash baselines/vlm/streamvln/scripts/download.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2 \
  --model-root "$STREAMVLN_MODEL_ROOT"

bash baselines/vlm/streamvln/scripts/download.sh \
  --repo google/siglip-so400m-patch14-384 \
  --model-root "$STREAMVLN_MODEL_ROOT"
```

Downloads go to `STREAMVLN_MODEL_ROOT` by default. Add `--endpoint https://hf-mirror.com` to use a Hugging Face mirror, or install ModelScope and add `--source modelscope`.

The resulting layout is usually:

```text
streamvln-models/
├── StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/
├── LLaVA-Video-7B-Qwen2/
└── siglip-so400m-patch14-384/
```

- `LLaVA-Video-7B-Qwen2/` is the base model for scratch training and can provide a missing tokenizer;
- `siglip-so400m-patch14-384/` is the vision tower used during training and evaluation;
- `StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/` is the starting point for continue training.

## 6. Configure local paths

After preparing the environment, upstream source, SatNav data, and model assets, copy the local template:

```bash
mkdir -p baselines/vlm/streamvln/.local
cp baselines/vlm/streamvln/local.env.example \
  baselines/vlm/streamvln/.local/env.sh
```

Fill `baselines/vlm/streamvln/.local/env.sh` with the paths prepared above:

```bash
export STREAMVLN_REPO="${STREAMVLN_REPO:-/path/to/StreamVLN}"
export STREAMVLN_PYTHON="${STREAMVLN_PYTHON:-/path/to/satnav-streamvln/bin/python}"
export STREAMVLN_MODEL_ROOT="${STREAMVLN_MODEL_ROOT:-/path/to/streamvln-models}"
export STREAMVLN_TOKENIZER_PATH="${STREAMVLN_TOKENIZER_PATH:-/path/to/streamvln-models/LLaVA-Video-7B-Qwen2}"
export STREAMVLN_VISION_TOWER="${STREAMVLN_VISION_TOWER:-/path/to/streamvln-models/siglip-so400m-patch14-384}"

export SATNAV_STREAMVLN_TRAIN_DATA="${SATNAV_STREAMVLN_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_STREAMVLN_EVAL_EPISODES="${SATNAV_STREAMVLN_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_STREAMVLN_SCENES_DIR="${SATNAV_STREAMVLN_SCENES_DIR:-/path/to/scenes}"
export SATNAV_STREAMVLN_OUTPUT="${SATNAV_STREAMVLN_OUTPUT:-output/baselines/vlm/streamvln}"
```

Variable sources:

| Variable | Source |
| --- | --- |
| `STREAMVLN_REPO` | StreamVLN checkout from Section 3 |
| `STREAMVLN_PYTHON` | Output of `command -v python` in Section 2 |
| `STREAMVLN_MODEL_ROOT` | Model directory selected in Section 5 |
| `STREAMVLN_TOKENIZER_PATH` | Downloaded `LLaVA-Video-7B-Qwen2/` |
| `STREAMVLN_VISION_TOWER` | Downloaded `siglip-so400m-patch14-384/` |
| `SATNAV_STREAMVLN_TRAIN_DATA` | Validated `trajectory_data/` from Section 4 |
| `SATNAV_STREAMVLN_EVAL_EPISODES` | SatNav-v0.1 evaluation Episode path template |
| `SATNAV_STREAMVLN_SCENES_DIR` | Validated GeoTIFF directory from Section 4 |
| `SATNAV_STREAMVLN_OUTPUT` | Training and evaluation output root |

Launchers load this file automatically. An environment variable set explicitly in the shell takes precedence over `.local/env.sh`. Do not commit local data, models, upstream source, output directories, or environment paths.

To use the variables directly in the current shell:

```bash
source baselines/vlm/streamvln/.local/env.sh
```

## 7. Training configuration

Defaults are in `baselines/vlm/streamvln/configs/train.yaml`:

| Parameter | Default |
| --- | ---: |
| Frames per window | 32 |
| History samples | 8 |
| Future actions | 4 |
| Epochs | 1 |
| Per-device batch size | 3 |
| Gradient accumulation | 2 |
| Learning rate | `2e-5` |
| Vision tower learning rate | `5e-6` |

Training uses DeepSpeed ZeRO-2 from `baselines/vlm/streamvln/configs/zero2.json` and updates the vision tower, multimodal projector, and language model. Override common parameters through `train.sh` instead of editing the public configuration.

In training commands, `--max-steps` means optimizer steps. In evaluation commands, the same name means the maximum primitive actions per Episode.

## 8. Smoke training

Use one GPU and one optimizer step to validate model loading, data loading, and checkpoint writing:

```bash
CUDA_DEVICES=0 \
SATNAV_MAX_EPISODES=1 \
SATNAV_MAX_SAMPLES=1 \
bash baselines/vlm/streamvln/scripts/train.sh continue \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --max-steps 1 \
  --dataloader-workers 0 \
  --no-data-augmentation \
  --no-torch-compile \
  --output-dir output/baselines/vlm/streamvln/train/continue-smoke
```

`SATNAV_MAX_SAMPLES` must be at least `GPU count × per-device batch size`; otherwise, `dataloader_drop_last=true` cannot form a complete batch.

After the smoke run, confirm that the output contains a model configuration, weights, and trainer state. Use the directory containing `config.json` and model weights as the later `--model-path`:

```bash
find output/baselines/vlm/streamvln/train/continue-smoke \
  -maxdepth 2 -name config.json -print
```

## 9. Full training

Continue from the official StreamVLN checkpoint:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/train.sh continue \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/train/continue-v0-1
```

Start from the LLaVA-Video base model:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/train.sh scratch \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/train/scratch-v0-1
```

If GPU memory is insufficient, reduce `--batch-size` first and use `--gradient-accumulation` to retain the desired effective batch size. The current launcher has no separate training-resume option; use an independent output directory for each run.

## 10. Configure online evaluation

Set the checkpoint to evaluate:

```bash
export STREAMVLN_CHECKPOINT=/path/to/trained-streamvln-checkpoint
```

Check Episode, scene, tokenizer, and output path resolution:

```bash
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run validates configuration, data selection, and the output location without loading the model or interacting with the environment.

## 11. Single-GPU smoke evaluation

Select five Episodes with at most five steps each:

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/5steps-1rank \
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

Each rank reads the complete Episode file. The shared evaluator performs deterministic sorting and stride sharding, so the data does not need to be split in advance. A two-GPU smoke run is:

```bash
CUDA_DEVICES=0,1 \
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

After all ranks exit successfully, the launcher writes the aggregated `summary.json`.

## 13. Full evaluation

Evaluate the full `val_seen` split with a 500-step limit:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

After completion, change the split and output directory to `val_unseen` and run again. `{split}` in `SATNAV_STREAMVLN_EVAL_EPISODES` is replaced automatically.

To continue an interrupted evaluation, rerun with exactly the same parameters and output directory and add `--resume`. Resume skips Episodes already written by the current rank. Use a new output directory after changing the checkpoint, data, seed, GPU count, or evaluation parameters.

## 14. Troubleshooting

### Why does the launcher report a StreamVLN revision mismatch?

Make sure `STREAMVLN_REPO` points to the intended checkout, then run:

```bash
git -C "$STREAMVLN_REPO" checkout \
  60476e81f4c01b29f1a51a7469f1cb4addbc1d62
```

### Why can it not find the tokenizer?

Set `STREAMVLN_TOKENIZER_PATH` to a directory containing `tokenizer_config.json`, or pass `--tokenizer-path` explicitly. A directory containing only model weight shards is not sufficient.

### Why does vision-tower loading fail?

Make sure `STREAMVLN_VISION_TOWER` points to a complete local SigLIP model directory and matches the vision tower used for training.

### Why does trajectory validation report too few frames?

Check that each `video` path in `annotations.json` resolves under the current `images/` directory and that trajectory generation completed. Do not combine annotations and images from different exports.

### Why did smoke training perform no optimizer step?

When limiting samples with `SATNAV_MAX_SAMPLES`, the value must be at least `GPU count × per-device batch size`. Also confirm that the selected trajectory contains at least four actions after action normalization.

### Why is there no output for a long time after training starts?

The first launch loads a large model, initializes DeepSpeed, and may compile the model. For environment or data diagnosis, run a bounded smoke with `--dataloader-workers 0 --no-data-augmentation --no-torch-compile`.

### Why were some Episodes skipped after restarting evaluation?

With `--resume`, the evaluator skips Episodes already completed in the output directory. Reuse that directory only when continuing exactly the same evaluation; otherwise, choose a new `--output-dir`.

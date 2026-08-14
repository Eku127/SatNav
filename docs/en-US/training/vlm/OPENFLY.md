# SatNav OpenFly Baseline

This guide explains how to train OpenFly on SatNav-v0.1 trajectories and run single- or multi-GPU online evaluation in SatSim. OpenFly uses an isolated Python environment. Its adapted model runtime is included in SatNav, so no external OpenFly package is required, and the environment is not shared with SatNav Core, Classic baselines, or other VLM baselines.

Before starting, read [Installation](../../getting-started/INSTALLATION.md), [Training Models with SatNav](../README.md), and [Evaluation](../../evaluation/README.md).

## 1. Model and SatNav interface

OpenFly predicts the next action from the navigation instruction, current RGB observation, two preceding observations, and action history. The SatNav adapter maps model output to four primitive actions:

| Model output | SatNav action | Environment behavior |
| --- | --- | --- |
| `stop` | `STOP` | End the current Episode |
| `forward` | `MOVE_FORWARD` | Move forward 10 m |
| `left` | `TURN_LEFT` | Turn left 15° |
| `right` | `TURN_RIGHT` | Turn right 15° |

Inference always uses exactly three frames in the order current, previous, and two steps back. At the beginning of an Episode, the adapter repeats the available frame when history is incomplete. The prompt retains at most the latest 16 actions.

OpenFly supports two checkpoint action formats:

- `compact`: generate `stop`, `forward`, `left`, or `right` directly;
- `original`: generate OpenFly's original eight-dimensional action token, then map it to the same four SatNav primitive actions.

Evaluation defaults to `auto`, which reads the action format from the checkpoint. Do not force a checkpoint to use the other format during training or evaluation.

The bundled runtime is based on [`SHAILAB-IPEC/OpenFly-Platform@c075075497a7122bad82f5b76b9be926ad5a81b3`](https://github.com/SHAILAB-IPEC/OpenFly-Platform/commit/c075075497a7122bad82f5b76b9be926ad5a81b3) and follows the pinned SwiftVLN adapter's three-frame and prompt behavior. Before using the source or weights, read [UPSTREAM](../../../../baselines/vlm/openfly/UPSTREAM.md), [NOTICE](../../../../baselines/vlm/openfly/NOTICE), and the [upstream license](../../../../baselines/vlm/openfly/LICENSE.upstream).

## 2. Create an isolated environment

Run the environment bootstrap from the SatNav repository root:

```bash
bash baselines/vlm/openfly/scripts/bootstrap_env.sh satnav-openfly
conda activate satnav-openfly
export PYTHONNOUSERSITE=1
```

The script creates a Python 3.10 environment, installs PyTorch 2.3.0 with the CUDA 12.1 wheel, installs the pinned OpenFly dependencies and SatNav, and validates dependency versions. Do not train or evaluate inside an existing VILA, OpenFly, or other VLM environment.

Check the key packages and entry points:

```bash
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
python -m baselines.vlm.openfly.dataset --help
python -m baselines.vlm.openfly.trainer --help
python -m baselines.vlm.openfly.evaluate --help
python -m baselines.vlm.openfly.checkpoint --help
command -v python
```

Record the output of `command -v python`; use it later as `OPENFLY_PYTHON`.

## 3. OpenFly source

Training and evaluation use the pinned implementation distributed under `baselines/vlm/openfly/`. Do not clone or install OpenFly-Platform and do not add another OpenFly checkout to `PYTHONPATH`.

For source comparison only, you can prepare a read-only checkout outside SatNav:

```bash
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git \
  /path/to/OpenFly-Platform
git -C /path/to/OpenFly-Platform checkout --detach \
  c075075497a7122bad82f5b76b9be926ad5a81b3
```

This checkout is not a training or evaluation input. Normal use of the OpenFly baseline can skip these clone commands.

## 4. Prepare SatNav data

### 4.1 Episodes, GeoTIFF scenes, and trajectories

Follow [Download Episode Data](../../dataset/DATA_DOWNLOAD.md), [Download Satellite Scenes](../../applications/MAP_DOWNLOAD.md), and [Generate Trajectory Data](../../applications/TRAJECTORY_GENERATION.md). OpenFly training reads both the train Episode file and trajectory JPEG files:

```text
SatNav-v0.1/
└── episodes/
    └── train/
        └── all_episodes.json

trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            └── ...
```

The Episode file and trajectories must come from the same dataset version. OpenFly matches them by scene, trajectory ID, and instruction, and reads `trajectory_type` from the Episode.

Set the data paths:

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/scenes
export SATNAV_TRAJECTORY_ROOT=/path/to/trajectory_data
export SATNAV_OPENFLY_TRAIN_EPISODES="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
```

First validate the Episode files and 59 GeoTIFF scenes:

```bash
bash scripts/validation/data_validation.sh
```

### 4.2 Validate trajectories

Check Episode matching, actions, frame counts, and image files over the complete train split:

```bash
mkdir -p output/baselines/vlm/openfly

bash baselines/vlm/openfly/scripts/validate_data.sh \
  "$SATNAV_TRAJECTORY_ROOT" \
  output/baselines/vlm/openfly/data_validation.json
```

The launcher validates the complete train split again before training. An unmatched Episode, missing or extra image, out-of-root path, invalid action, or frame-count mismatch stops training rather than silently dropping samples.

## 5. Prepare a model

SatNav does not store OpenFly weights in the source repository. Use a released
SatNav checkpoint, a model from OpenFly upstream, or an existing training run.

### 5.1 Released SatNav checkpoints

The [SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
provides two OpenFly checkpoints:

- [scratch](https://huggingface.co/Eku127/openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5), trained from OpenVLA/OpenFly processor assets;
- [continue](https://huggingface.co/Eku127/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5), trained from the OpenFly Agent checkpoint.

Download either checkpoint into a local model directory:

```bash
export OPENFLY_MODEL_ROOT=/path/to/openfly-models
python -m huggingface_hub.commands.huggingface_cli download \
  Eku127/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 \
  --local-dir "$OPENFLY_MODEL_ROOT/openfly-satnav-continue"
```

Pass the downloaded directory to `--model-path` for evaluation. These releases
contain inference artifacts rather than optimizer and scheduler state.

The released checkpoints achieve the following reference results with a
500-step cap.

| Checkpoint | Split | Episodes | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | `val_seen` | 4,601 | 166.88 | 33.47 | 13.21 | 13.03 |
| Scratch | `val_unseen` | 8,756 | 195.91 | 32.17 | 11.73 | 11.62 |
| Continue | `val_seen` | 4,601 | 163.07 | 38.08 | 21.10 | 20.99 |
| Continue | `val_unseen` | 8,756 | 196.90 | 34.88 | 17.12 | 16.91 |

### 5.2 Continue from a complete Hugging Face checkpoint

`continue` accepts a complete local OpenFly checkpoint in Hugging Face format:

```text
openfly-hf-checkpoint/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
├── model.safetensors
└── ...
```

Weights may instead be stored as several `model-*.safetensors` files with `model.safetensors.index.json`. The directory must include the processor, tokenizer, and every weight shard; a standalone adapter or partial shard set is not sufficient.

### 5.3 Scratch from a native checkpoint and processor

`scratch` accepts a native OpenFly run directory or one `.pt` checkpoint:

```text
openfly-native-run/
└── checkpoints/
    ├── step-10000-epoch-1-loss=...pt
    └── ...
```

When given a run directory, the launcher selects the `.pt` with the highest step under `checkpoints/`. It also requires a matching processor and tokenizer directory:

```text
openfly-processor/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

Training converts the native model to the current OpenFly Hugging Face runtime. Reserve additional disk space for this conversion. The original `.pt` and processor directory are not modified.

## 6. Configure local paths

Copy the local template:

```bash
mkdir -p baselines/vlm/openfly/.local
cp baselines/vlm/openfly/local.env.example \
  baselines/vlm/openfly/.local/env.sh
```

Fill `baselines/vlm/openfly/.local/env.sh` with the paths prepared above:

```bash
export OPENFLY_PYTHON="${OPENFLY_PYTHON:-/path/to/satnav-openfly/bin/python}"

export OPENFLY_CONTINUE_MODEL="${OPENFLY_CONTINUE_MODEL:-/path/to/openfly-hf-checkpoint}"
export OPENFLY_NATIVE_RUN="${OPENFLY_NATIVE_RUN:-/path/to/openfly-native-run}"
export OPENFLY_PROCESSOR_PATH="${OPENFLY_PROCESSOR_PATH:-/path/to/openfly-processor}"
export OPENFLY_NATIVE_HF_CACHE_DIR="${OPENFLY_NATIVE_HF_CACHE_DIR:-/path/to/openfly-native-hf-cache}"

export SATNAV_OPENFLY_TRAIN_DATA="${SATNAV_OPENFLY_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_OPENFLY_TRAIN_EPISODES="${SATNAV_OPENFLY_TRAIN_EPISODES:-/path/to/SatNav-v0.1/episodes/train/all_episodes.json}"
export SATNAV_OPENFLY_EVAL_EPISODES="${SATNAV_OPENFLY_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_OPENFLY_SCENES_DIR="${SATNAV_OPENFLY_SCENES_DIR:-/path/to/scenes}"
export SATNAV_OPENFLY_OUTPUT="${SATNAV_OPENFLY_OUTPUT:-output/baselines/vlm/openfly}"
```

When using only `continue`, leave `OPENFLY_NATIVE_RUN`, `OPENFLY_PROCESSOR_PATH`, and `OPENFLY_NATIVE_HF_CACHE_DIR` unset. When using only `scratch`, leave `OPENFLY_CONTINUE_MODEL` unset.

Variable sources:

| Variable | Source |
| --- | --- |
| `OPENFLY_PYTHON` | Output of `command -v python` in Section 2 |
| `OPENFLY_CONTINUE_MODEL` | Released or locally prepared complete Hugging Face checkpoint from Section 5.1 or 5.2 |
| `OPENFLY_NATIVE_RUN` | Native run or `.pt` from Section 5.3 |
| `OPENFLY_PROCESSOR_PATH` | Processor/tokenizer directory from Section 5.3 |
| `OPENFLY_NATIVE_HF_CACHE_DIR` | Local cache used for native model conversion |
| `SATNAV_OPENFLY_TRAIN_DATA` | Validated `trajectory_data/` from Section 4 |
| `SATNAV_OPENFLY_TRAIN_EPISODES` | Train Episode file corresponding to the trajectories |
| `SATNAV_OPENFLY_EVAL_EPISODES` | SatNav-v0.1 evaluation Episode path template |
| `SATNAV_OPENFLY_SCENES_DIR` | Validated GeoTIFF directory from Section 4 |
| `SATNAV_OPENFLY_OUTPUT` | Training and evaluation output root |

Launchers load this file automatically. An environment variable set explicitly in the shell takes precedence over `.local/env.sh`. Do not commit local data, models, output directories, caches, or environment paths.

To use the variables directly in the current shell:

```bash
source baselines/vlm/openfly/.local/env.sh
```

## 7. Training configuration and checkpoint validation

Before `continue` training, confirm that the complete checkpoint loads:

```bash
python -m baselines.vlm.openfly.checkpoint \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --device cuda:0 \
  --dtype bfloat16
```

This checks the configuration, processor, tokenizer, and all model weights. A native `.pt` used for `scratch` is validated and converted when training starts.

Defaults are in `baselines/vlm/openfly/configs/train.yaml`:

| Parameter | Default |
| --- | ---: |
| RGB frames per sample | 3 |
| Action format | `compact` |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `2e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 4 |

Training uses DeepSpeed ZeRO-2 from `baselines/vlm/openfly/configs/zero2.json` and updates the vision backbone, language model, and projector. At completion, the launcher reloads the output and confirms that all three model components changed.

In training commands, `--max-steps` means optimizer steps. In evaluation commands, the same name means the maximum primitive actions per Episode.

## 8. Smoke training

Run one optimizer step to validate data loading, model updates, and checkpoint saving:

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-smoke \
  --max-episodes 2 \
  --max-samples 4 \
  --max-steps 1 \
  --gpus 1
```

`--max-episodes` and `--max-samples` limit only the bounded subset sent to the trainer; preflight validation still covers the complete train split. After success, the output contains a directly evaluable model and `checkpoint-*`, and the launcher performs strict reload and parameter-update checks.

## 9. Full training and resume

Continue from a complete Hugging Face checkpoint:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-v0-1 \
  --gpus 8
```

Start from a native OpenFly checkpoint:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend scratch \
  --model-path "$OPENFLY_NATIVE_RUN" \
  --processor-path "$OPENFLY_PROCESSOR_PATH" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/scratch-v0-1 \
  --gpus 8
```

If GPU memory is insufficient, reduce `--batch-size` first and use `--gradient-accumulation` to adjust the effective batch size.

If training is interrupted and a complete `checkpoint-*` exists in the output directory, rerun with exactly the same parameters and output directory and add `--resume`:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-v0-1 \
  --gpus 8 \
  --resume
```

Resume is only for the same training run. Use a new output directory after changing the model, data, action format, configuration, GPU count, or sampling parameters.

## 10. Configure online evaluation

The training output directory is a complete Hugging Face checkpoint. Set the checkpoint to evaluate:

```bash
export OPENFLY_CHECKPOINT=/path/to/trained-openfly-checkpoint
```

Check checkpoint, Episode, scene, and output path resolution:

```bash
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run checks the checkpoint configuration and evaluation data selection without loading model weights or interacting with the environment.

## 11. Single-GPU smoke evaluation

Select four Episodes with at most five steps each:

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/5steps-1rank \
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
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

Multi-GPU evaluation requires an explicit `--output-dir`. After every rank exits successfully, the launcher writes the aggregated `summary.json`.

## 13. Full evaluation

Evaluate the complete `val_seen` split with a 500-step limit:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

After completion, change the split and output directory to `val_unseen` and run again. `{split}` in `SATNAV_OPENFLY_EVAL_EPISODES` is replaced automatically.

To continue an interrupted evaluation, rerun with exactly the same parameters and output directory and add `--resume`. Resume skips Episodes already written by the current rank. Use a new output directory after changing the checkpoint, data, seed, GPU count, or evaluation parameters.

## 14. Troubleshooting

### Why can it not find the OpenFly Python environment?

Make sure `OPENFLY_PYTHON` points to the executable created in Section 2:

```bash
"$OPENFLY_PYTHON" -c "import torch, transformers; print(torch.__version__)"
```

Do not point it to the SatNav Core, VILA, or another VLM environment.

### Why does a continue checkpoint fail to load?

The model directory must contain the complete configuration, processor, tokenizer, and all safetensors weights. OpenFly does not accept PyTorch `.bin`, a standalone adapter, missing shards, or a directory containing only optimizer state.

### Why can scratch training not find the processor?

Use `--processor-path` or `OPENFLY_PROCESSOR_PATH` to select the complete processor matching the native model. It must include at least the model configuration, image processor, and tokenizer files.

### Why does the action format not match?

`compact` and `original` have different outputs and training targets. Keep the default `auto` during evaluation. If `--action-format` is explicit, it must match the format declared by the checkpoint.

### Why can trajectory validation not match an Episode?

Make sure `SATNAV_OPENFLY_TRAIN_EPISODES` and the trajectories come from the same SatNav-v0.1 train release and that instructions were not rewritten. Do not mix Episodes and trajectory exports from different versions.

### Why is model loading delayed after training starts?

The launcher validates the complete SatNav-v0.1 trajectory before loading the model. This can take time on network or slow storage; keep Episodes, annotations, and JPEG files on fast local storage when possible.

### Why was training resume rejected?

Confirm that a complete `checkpoint-*` exists and that the backend, model, data, action format, configuration, GPU count, and sampling parameters exactly match the initial run. Use a new output directory for a changed experiment.

### Why were some Episodes skipped after restarting evaluation?

With `--resume`, the evaluator skips Episodes already completed in the output directory. Reuse that directory only when continuing exactly the same evaluation; otherwise, choose a new `--output-dir`.

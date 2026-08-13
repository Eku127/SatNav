# SatNav Classic Baselines

This guide covers SatNav-v0.1 data preparation, training, and evaluation for Seq2Seq and CMA. Both models share the Episode data, GeoTIFF scenes, offline trajectories, vocabulary, and GloVe embeddings, while using different model configurations and checkpoints.

For a first run on the two bundled example Episodes, start with [Training Models with SatNav](README.md). The rest of this guide targets the complete SatNav-v0.1 dataset.

## 1. Model overview

Seq2Seq and CMA use offline imitation learning. Training reads exported RGB frames, navigation instructions, and expert actions. During evaluation, the model predicts one action at a time from the current observation in SatSim.

| Model | Instruction encoding | Visual features | Navigation state |
| --- | --- | --- | --- |
| Seq2Seq | Unidirectional LSTM using the final instruction state | Global ResNet50 feature | One-layer GRU |
| CMA | Bidirectional LSTM retaining every token | Spatial ResNet50 features | Two GRUs with cross-modal attention |

Both models receive:

- the current RGB observation;
- the tokenized instruction;
- the previous SatNav action;
- an Episode continuation mask.

The output is one of four SatNav primitive actions: `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, or `TURN_RIGHT`.

## 2. Environment setup

Create the classic environment from the SatNav repository root:

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic,applications]'
```

For another CUDA version, install PyTorch and TorchVision builds that match the machine driver. Confirm that the training entry points import correctly:

```bash
python -c "from satnav.training.offline_trainer import OfflineTrainer; print('Classic training import OK')"
python -m baselines.classic --help
```

Training and evaluation scripts use the repository's `configs/` and `scripts/`, so run them from the source checkout and keep the editable installation.

## 3. Prepare SatNav-v0.1 data

### 3.1 Episodes and GeoTIFF scenes

Follow [Download Episode Data](../dataset/DATA_DOWNLOAD.md) and [Download Satellite Scenes](../applications/MAP_DOWNLOAD.md), then set:

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/satnav-scenes
export SATNAV_TRAIN_EPISODES_PATH="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
export SATNAV_TRAJECTORY_DIR="$SATNAV_DATA_ROOT/trajectory_data"
```

The standard dataset has this layout:

```text
SatNav-v0.1/
└── episodes/
    ├── train/all_episodes.json
    └── eval/
        ├── val_seen/all_episodes.json
        └── val_unseen/all_episodes.json
```

`SATNAV_SCENES_DIR` must contain the 59 GeoTIFF files referenced by logical Episode `scene_id` values. Validate the configuration:

```bash
bash scripts/validation/data_validation.sh
```

A complete dataset reports 105,164 train Episodes, 4,574 `val_seen` Episodes, 8,756 `val_unseen` Episodes, and 59 GeoTIFF scenes.

### 3.2 Generate offline trajectories

Seq2Seq and CMA use the same trajectory export. Generate the complete train split with the parallel entry point:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir "$SATNAV_TRAJECTORY_DIR"
```

The complete export is approximately 233 GB; reserve at least 250 GB of free space. If generation is interrupted, rerun the same command with the same configuration and output directory to continue.

The output layout is:

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            └── ...
```

A complete run reports:

```text
Success (incl. cached): 105164
Discarded (max steps): 0
Failed: 0
Generated annotations: 105164 / 105164 episodes
```

See [Generate Trajectory Data](../applications/TRAJECTORY_GENERATION.md) for production settings, worker tuning, and continuation behavior.

### 3.3 Build the vocabulary and GloVe embeddings

Both models must use a vocabulary built from the same train Episodes. Choose a shared artifact directory:

```bash
export SATNAV_CLASSIC_ARTIFACTS="$PWD/output/baselines/classic/artifacts"
export SATNAV_VOCAB_PATH="$SATNAV_CLASSIC_ARTIFACTS/satnav_v0_1_vocab.json"
export SATNAV_EMBEDDING_PATH="$SATNAV_CLASSIC_ARTIFACTS/satnav_v0_1_glove50d.json.gz"
mkdir -p "$SATNAV_CLASSIC_ARTIFACTS"
```

Build the vocabulary:

```bash
python -m satnav.utils.build_vocab \
  --dataset "$SATNAV_TRAIN_EPISODES_PATH" \
  --output "$SATNAV_VOCAB_PATH"
```

Prepare the GloVe 6B 50d text file, then create embeddings whose rows match the vocabulary:

```bash
export SATNAV_GLOVE_TXT=/path/to/glove.6B.50d.txt

python -m satnav.utils.build_glove_embeddings \
  --vocab "$SATNAV_VOCAB_PATH" \
  --glove "$SATNAV_GLOVE_TXT" \
  --output "$SATNAV_EMBEDDING_PATH" \
  --embedding-dim 50
```

Read the vocabulary size:

```bash
python - "$SATNAV_VOCAB_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    vocabulary = json.load(handle)
print(vocabulary["vocab_size"])
PY
```

Use this value for `MODEL.INSTRUCTION_ENCODER.vocab_size` in all four local configurations. Training, checkpoint loading, and evaluation must keep using the same vocabulary.

## 4. Create local configurations

The public YAML files target the tiny example. Copy them to Git-ignored local configurations for full training:

```bash
cp configs/baselines/seq2seq_offline_train.yaml configs/local_seq2seq_train.yaml
cp configs/baselines/seq2seq_eval.yaml configs/local_seq2seq_eval.yaml
cp configs/baselines/cma_offline_train.yaml configs/local_cma_train.yaml
cp configs/baselines/cma_eval.yaml configs/local_cma_eval.yaml
```

### 4.1 Training configurations

Set these fields in `configs/local_seq2seq_train.yaml` and `configs/local_cma_train.yaml`:

| Field | Value |
| --- | --- |
| `BASE_TASK_CONFIG_PATH` | `configs/satnav_task.yaml` |
| `DATASET.SPLIT` | `train` |
| `DATASET.DATA_PATH` | Train `all_episodes.json` |
| `DATASET.SCENES_DIR` | GeoTIFF scene directory |
| `DATASET.vocab_file` | File referenced by `SATNAV_VOCAB_PATH` |
| `IL.OFFLINE.annotations_path` | `trajectory_data/annotations.json` |
| `IL.OFFLINE.images_root` | `trajectory_data/images` |
| `MODEL.INSTRUCTION_ENCODER.embedding_file` | File referenced by `SATNAV_EMBEDDING_PATH` |
| `MODEL.INSTRUCTION_ENCODER.vocab_size` | `vocab_size` from the vocabulary JSON |

Production trajectories store 448 × 448 RGB images. The default `IL.OFFLINE.rgb_size: 224` resizes them to the training input size and can remain unchanged.

Keep the model-specific fields consistent:

| Field | Seq2Seq | CMA |
| --- | --- | --- |
| `MODEL.policy_name` | `seq2seq` | `cma` |
| `MODEL.INSTRUCTION_ENCODER.bidirectional` | `false` | `true` |
| `MODEL.INSTRUCTION_ENCODER.final_state_only` | `true` | `false` |

### 4.2 Evaluation configurations

In `configs/local_seq2seq_eval.yaml` and `configs/local_cma_eval.yaml`:

- set `BASE_TASK_CONFIG_PATH` to `configs/satnav_eval_task.yaml`;
- keep the same `MODEL` structure used for training;
- use the same vocabulary and embeddings;
- set `MODEL.INSTRUCTION_ENCODER.vocab_size` to the actual vocabulary size.

Evaluation launchers override Episode and scene paths for the selected split. Do not change hidden sizes, encoder direction, embedding size, or backbone before loading an existing checkpoint.

### 4.3 Local environment variables

Copy the shared, training, and evaluation templates:

```bash
mkdir -p .local \
  scripts/seq2seq/.local \
  scripts/cma/.local \
  baselines/classic/.local

cp local.env.example .local/env.sh
cp scripts/seq2seq/local.env.example scripts/seq2seq/.local/env.sh
cp scripts/cma/local.env.example scripts/cma/.local/env.sh
cp baselines/classic/local.env.example baselines/classic/.local/env.sh
```

Set Seq2Seq training variables in `scripts/seq2seq/.local/env.sh`:

```bash
export SEQ2SEQ_TRAIN_CONFIG_PATH="${SEQ2SEQ_TRAIN_CONFIG_PATH:-configs/local_seq2seq_train.yaml}"
export SEQ2SEQ_EVAL_CONFIG_PATH="${SEQ2SEQ_EVAL_CONFIG_PATH:-configs/local_seq2seq_eval.yaml}"
export SEQ2SEQ_OUTPUT_ROOT="${SEQ2SEQ_OUTPUT_ROOT:-output/seq2seq_offline}"
export SEQ2SEQ_CUDA_DEVICES="${SEQ2SEQ_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
```

Set CMA training variables in `scripts/cma/.local/env.sh`:

```bash
export CMA_TRAIN_CONFIG_PATH="${CMA_TRAIN_CONFIG_PATH:-configs/local_cma_train.yaml}"
export CMA_EVAL_CONFIG_PATH="${CMA_EVAL_CONFIG_PATH:-configs/local_cma_eval.yaml}"
export CMA_OUTPUT_ROOT="${CMA_OUTPUT_ROOT:-output/cma}"
export CMA_CUDA_DEVICES="${CMA_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
```

Set shared evaluation variables in `baselines/classic/.local/env.sh`:

```bash
export SATNAV_DATA_ROOT="${SATNAV_DATA_ROOT:-/path/to/SatNav-v0.1}"
export SATNAV_SCENES_DIR="${SATNAV_SCENES_DIR:-/path/to/satnav-scenes}"
export SATNAV_VOCAB_PATH="${SATNAV_VOCAB_PATH:-/path/to/satnav_v0_1_vocab.json}"

export SATNAV_SEQ2SEQ_EVAL_CONFIG="${SATNAV_SEQ2SEQ_EVAL_CONFIG:-configs/local_seq2seq_eval.yaml}"
export SATNAV_SEQ2SEQ_CHECKPOINT="${SATNAV_SEQ2SEQ_CHECKPOINT:-/path/to/seq2seq/best.pth}"

export SATNAV_CMA_EVAL_CONFIG="${SATNAV_CMA_EVAL_CONFIG:-configs/local_cma_eval.yaml}"
export SATNAV_CMA_CHECKPOINT="${SATNAV_CMA_CHECKPOINT:-/path/to/cma/best.pth}"
```

## 5. Train Seq2Seq

### 5.1 Single-GPU smoke training

Use one GPU and one epoch to validate data loading, optimizer updates, and checkpoint saving:

```bash
CONFIG_PATH=configs/local_seq2seq_train.yaml \
CUDA_DEVICES=0 \
GPUS_PER_NODE=1 \
NUM_EPOCHS=1 \
PER_GPU_BATCH_SIZE=1 \
NUM_WORKERS=0 \
USE_SWANLAB=false \
SWANLAB_EXP_NAME=seq2seq-v0-1-smoke \
bash scripts/seq2seq/train_offline_ddp.sh
```

The launcher uses `torchrun` even with one GPU, keeping the configuration override path identical between single- and multi-GPU training.

### 5.2 Multi-GPU training

After the smoke run succeeds, increase the GPU count, batch size, worker count, and epoch count:

```bash
CONFIG_PATH=configs/local_seq2seq_train.yaml \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_NODE=8 \
NUM_EPOCHS=10 \
PER_GPU_BATCH_SIZE=8 \
NUM_WORKERS=8 \
USE_SWANLAB=false \
SWANLAB_EXP_NAME=seq2seq-v0-1 \
bash scripts/seq2seq/train_offline_ddp.sh
```

The effective Seq2Seq batch size is `PER_GPU_BATCH_SIZE × GPUS_PER_NODE`. Multi-GPU training supports a single node; `GPUS_PER_NODE` must equal the number of entries in `CUDA_DEVICES`.

Default output layout:

```text
output/seq2seq_offline/
├── checkpoints/
│   ├── seq2seq-v0-1/
│   │   └── best.pth
│   └── latest -> seq2seq-v0-1
└── logs/
    └── seq2seq-v0-1.log
```

## 6. Train CMA

### 6.1 Single-GPU smoke training

```bash
CONFIG_PATH=configs/local_cma_train.yaml \
CUDA_DEVICES=0 \
GPUS_PER_NODE=1 \
NUM_EPOCHS=1 \
PER_GPU_BATCH_SIZE=1 \
NUM_WORKERS=0 \
USE_SWANLAB=false \
SWANLAB_EXP_NAME=cma-v0-1-smoke \
bash scripts/cma/train_ddp.sh
```

### 6.2 Multi-GPU training

```bash
CONFIG_PATH=configs/local_cma_train.yaml \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_NODE=8 \
NUM_EPOCHS=10 \
PER_GPU_BATCH_SIZE=4 \
NUM_WORKERS=4 \
USE_SWANLAB=false \
SWANLAB_EXP_NAME=cma-v0-1 \
bash scripts/cma/train_ddp.sh
```

CMA uses spatial visual features and two recurrent states, so it usually needs more GPU memory than Seq2Seq. If training runs out of memory, reduce `PER_GPU_BATCH_SIZE` first, then reduce `NUM_WORKERS`.

Default output layout:

```text
output/cma/
├── checkpoints/
│   ├── cma-v0-1/
│   │   └── best.pth
│   └── latest -> cma-v0-1
└── logs/
    └── cma-v0-1.log
```

## 7. Checkpoints and continued training

Both Seq2Seq and CMA `best.pth` files contain:

```text
config, epoch, loss, optim_state, state_dict, step_id
```

Confirm that training completed at least one optimizer step:

```bash
python - /path/to/best.pth <<'PY'
import sys
from baselines.classic.common.checkpoints import read_training_checkpoint

checkpoint = read_training_checkpoint(sys.argv[1])
print("epoch", checkpoint["epoch"])
print("step_id", checkpoint["step_id"])
print("loss", checkpoint["loss"])
PY
```

To continue training, set the following in the corresponding local training configuration:

```yaml
IL:
  load_from_ckpt: true
  ckpt_to_load: /path/to/previous/best.pth
  epochs: 20
```

`epochs` is the target total epoch count, not the number of additional epochs. The trainer restores the model, optimizer, completed epoch count, and `step_id`, then writes the new run to the experiment directory created by the launcher.

## 8. Configure shared evaluation

Update the checkpoints in `baselines/classic/.local/env.sh`:

```bash
export SATNAV_SEQ2SEQ_CHECKPOINT=/path/to/seq2seq-v0-1/best.pth
export SATNAV_CMA_CHECKPOINT=/path/to/cma-v0-1/best.pth
```

Check that the local evaluation configurations resolve correctly:

```bash
python -m baselines.classic \
  --method seq2seq \
  --config configs/local_seq2seq_eval.yaml \
  --split val_seen \
  --print-config

python -m baselines.classic \
  --method cma \
  --config configs/local_cma_eval.yaml \
  --split val_seen \
  --print-config
```

Seq2Seq and CMA checkpoints are loaded strictly.

## 9. Smoke evaluation

A useful smoke run selects eight Episodes and allows at most five steps per Episode. Use a separate output directory for each checkpoint:

```bash
SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/seq2seq-v0-1/val_seen/5steps/1rank \
bash scripts/classic/eval.sh seq2seq val_seen 8

SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/cma-v0-1/val_seen/5steps/1rank \
bash scripts/classic/eval.sh cma val_seen 8
```

Result layout:

```text
<output-dir>/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

Inspect the summary:

```bash
python - output/baselines/classic/seq2seq-v0-1/val_seen/5steps/1rank/summary.json <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(json.dumps(summary, indent=2, sort_keys=True))
assert summary["status"] == "complete"
assert summary["error_episode_count"] == 0
PY
```

CMA uses the same result format; only the summary path changes.

## 10. Multi-GPU evaluation

Online evaluation shards Episodes across ranks, with an independent model process on one GPU per rank. A two-GPU smoke run is:

```bash
SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/seq2seq-v0-1/val_seen/5steps/2rank \
bash scripts/classic/eval_parallel.sh seq2seq val_seen 2 0,1 8

SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/cma-v0-1/val_seen/5steps/2rank \
bash scripts/classic/eval_parallel.sh cma val_seen 2 0,1 8
```

`eval_parallel.sh` waits for all ranks and then runs the shared aggregator. `limit` applies to the globally sorted Episode list before stride sharding, so each rank processes four Episodes in this example.

## 11. Full evaluation

For official evaluation, set `SATNAV_MAX_STEPS=500` and `limit=-1`:

```bash
SATNAV_MAX_STEPS=500 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/seq2seq-v0-1/val_seen/500steps/8rank \
bash scripts/classic/eval_parallel.sh \
  seq2seq val_seen 8 0,1,2,3,4,5,6,7 -1

SATNAV_MAX_STEPS=500 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/cma-v0-1/val_seen/500steps/8rank \
bash scripts/classic/eval_parallel.sh \
  cma val_seen 8 0,1,2,3,4,5,6,7 -1
```

After `val_seen`, change both the split and output directory to `val_unseen` and run again. See [Evaluation — SatNav-v0.1 evaluation settings](../evaluation/README.md#satnav-v01-evaluation-settings) for the standard Episode counts and rollout parameters.

Evaluation launchers enable resume by default. To continue an interrupted run, rerun with exactly the same data, checkpoint, seed, `world_size`, maximum steps, and output directory. Use a new output directory when evaluating another checkpoint or changing any run parameter.

A complete result must satisfy:

- `summary.json` has `status` equal to `complete`;
- `error_episode_count` is 0;
- `unique_record_count` equals the Episode count for the split;
- `metrics` includes `distance_to_goal`, `success`, `oracle_success`, `spl`, and `path_length`.

See [Evaluation](../evaluation/README.md) for result fields, Episode sharding, and resume rules.

## 12. Troubleshooting

### Why can training not find an RGB frame?

Make sure `IL.OFFLINE.annotations_path` points to the final `annotations.json` and `IL.OFFLINE.images_root` points to `images/` from the same trajectory export. Do not mix annotations and images from different exports.

### Why does the vocabulary size not match?

The training configuration, evaluation configuration, embedding file, and `SATNAV_VOCAB_PATH` must all come from the same vocabulary build. Check `vocab_size` in the JSON file and update both model configurations.

### Why does checkpoint loading fail?

Use the evaluation configuration for the correct model. Seq2Seq and CMA have different encoders, recurrent states, and parameter names, so their checkpoints are not interchangeable. A checkpoint also cannot be loaded directly after changing the same model's hidden size, backbone, or instruction encoder.

### Why do all multi-GPU training processes use the same GPU?

Launch through `train_offline_ddp.sh` or `train_ddp.sh`, and make `GPUS_PER_NODE` equal the number of GPUs in `CUDA_DEVICES`. Do not manually assign the same `LOCAL_RANK` to several processes.

### Why did a repeated evaluation run execute no Episodes?

Evaluation resumes by default and skips Episodes already present in the output directory. This is expected when continuing the same run. After changing the checkpoint or evaluation settings, choose a new `SATNAV_CLASSIC_RUN_OUTPUT`.

### Why are Success and SPL low in a smoke run?

A five-step smoke run only checks checkpoint loading, model inference, environment interaction, and result writing. Report metrics from the complete 500-step `val_seen` and `val_unseen` evaluations.

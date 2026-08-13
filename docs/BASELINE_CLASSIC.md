# SatNav Classic Baselines

本文介绍如何使用 SatNav-v0.1 数据准备、训练和评测 Seq2Seq 与 CMA。两个模型共享 Episode、
GeoTIFF、离线 trajectory、vocabulary 和 GloVe embedding，但使用不同的模型配置和
checkpoint。

如果只需要运行仓库内置的两个示例 Episode，请先阅读[模型训练](TRAINING.md)。本文以下
内容面向完整 SatNav-v0.1 数据。

## 1. 模型概览

Seq2Seq 和 CMA 都使用离线模仿学习。训练阶段读取预先生成的 RGB frame、导航指令和
expert action；评测阶段在 SatSim 中根据当前 observation 逐步预测动作。

| 模型 | 指令编码 | 视觉特征 | 导航状态 |
| --- | --- | --- | --- |
| Seq2Seq | 单向 LSTM，使用最终指令状态 | ResNet50 全局特征 | 单层 GRU |
| CMA | 双向 LSTM，保留全部 token | ResNet50 空间特征 | 两层 GRU 与跨模态注意力 |

两个模型都接收：

- 当前 RGB observation；
- tokenized instruction；
- 上一个 SatNav action；
- Episode continuation mask。

模型输出四个 SatNav primitive action 之一：`STOP`、`MOVE_FORWARD`、`TURN_LEFT` 或
`TURN_RIGHT`。模型结构的详细说明参阅 [Seq2Seq Model Structure](../doc/models/SEQ2SEQ_IMPLEMENTATION.md)
和 [CMA Model Structure](../doc/models/CMA_IMPLEMENTATION.md)。

## 2. 准备环境

在 SatNav 仓库根目录创建 classic 环境：

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic,applications]'
```

如果使用其他 CUDA 版本，请安装与本机驱动匹配的 PyTorch 和 TorchVision。确认训练入口
可以导入：

```bash
python -c "from satnav.training.offline_trainer import OfflineTrainer; print('Classic training import OK')"
python -m baselines.classic --help
```

训练和评测脚本依赖仓库中的 `configs/` 与 `scripts/`，因此应从源码目录运行，并保留
editable install。

## 3. 准备 SatNav-v0.1 数据

### 3.1 Episode 与 GeoTIFF

按照 [Episode 数据下载](DATA_DOWNLOAD.md)和[卫星场景下载](APPLICATION_MAP_DOWNLOAD.md)
准备数据，然后在当前终端设置：

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/satnav-scenes
export SATNAV_TRAIN_EPISODES_PATH="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
export SATNAV_TRAJECTORY_DIR="$SATNAV_DATA_ROOT/trajectory_data"
```

标准数据应包含：

```text
SatNav-v0.1/
└── episodes/
    ├── train/all_episodes.json
    └── eval/
        ├── val_seen/all_episodes.json
        └── val_unseen/all_episodes.json
```

`SATNAV_SCENES_DIR` 应包含与 Episode logical `scene_id` 对应的 59 个 GeoTIFF。运行数据配置
检查：

```bash
bash scripts/validation/data_validation.sh
```

完成时应显示 train 105,164、val_seen 4,574、val_unseen 8,756 个 Episode，以及 59 个
GeoTIFF scene。

### 3.2 生成离线 Trajectory

Seq2Seq 和 CMA 使用同一份 trajectory export。完整 train split 使用并行生成入口：

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir "$SATNAV_TRAJECTORY_DIR"
```

完整数据约占 233 GB，建议至少预留 250 GB 可用空间。生成中断后，使用相同配置和输出目录
重新执行命令即可继续。

输出结构为：

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

生成完成后应显示：

```text
Success (incl. cached): 105164
Discarded (max steps): 0
Failed: 0
Generated annotations: 105164 / 105164 episodes
```

完整的生产配置、worker 调整和续跑说明参阅[轨迹数据生成](APPLICATION_TRAJ_GENERATION.md)。

### 3.3 构建 Vocabulary 与 GloVe Embedding

Seq2Seq 和 CMA 应使用从同一份 train Episode 构建的 vocabulary。先选择公共产物目录：

```bash
export SATNAV_CLASSIC_ARTIFACTS="$PWD/output/baselines/classic/artifacts"
export SATNAV_VOCAB_PATH="$SATNAV_CLASSIC_ARTIFACTS/satnav_v0_1_vocab.json"
export SATNAV_EMBEDDING_PATH="$SATNAV_CLASSIC_ARTIFACTS/satnav_v0_1_glove50d.json.gz"
mkdir -p "$SATNAV_CLASSIC_ARTIFACTS"
```

构建 vocabulary：

```bash
python -m satnav.utils.build_vocab \
  --dataset "$SATNAV_TRAIN_EPISODES_PATH" \
  --output "$SATNAV_VOCAB_PATH"
```

准备 GloVe 6B 50d 文本文件，然后构建与 vocabulary 行号一致的 embedding：

```bash
export SATNAV_GLOVE_TXT=/path/to/glove.6B.50d.txt

python -m satnav.utils.build_glove_embeddings \
  --vocab "$SATNAV_VOCAB_PATH" \
  --glove "$SATNAV_GLOVE_TXT" \
  --output "$SATNAV_EMBEDDING_PATH" \
  --embedding-dim 50
```

查看 vocabulary 大小：

```bash
python - "$SATNAV_VOCAB_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    vocabulary = json.load(handle)
print(vocabulary["vocab_size"])
PY
```

记录该数值，后续四个本地配置中的 `MODEL.INSTRUCTION_ENCODER.vocab_size` 必须使用相同
大小。训练、checkpoint 和评测不能更换为另一份 vocabulary。

## 4. 创建本地配置

公共 YAML 默认指向仓库 tiny example。正式训练时先复制为 Git ignored 的本地配置：

```bash
cp configs/baselines/seq2seq_offline_train.yaml configs/local_seq2seq_train.yaml
cp configs/baselines/seq2seq_eval.yaml configs/local_seq2seq_eval.yaml
cp configs/baselines/cma_offline_train.yaml configs/local_cma_train.yaml
cp configs/baselines/cma_eval.yaml configs/local_cma_eval.yaml
```

### 4.1 训练配置

在 `configs/local_seq2seq_train.yaml` 和 `configs/local_cma_train.yaml` 中设置：

| 字段 | 值 |
| --- | --- |
| `BASE_TASK_CONFIG_PATH` | `configs/satnav_task.yaml` |
| `DATASET.SPLIT` | `train` |
| `DATASET.DATA_PATH` | train `all_episodes.json` |
| `DATASET.SCENES_DIR` | GeoTIFF scene 目录 |
| `DATASET.vocab_file` | `SATNAV_VOCAB_PATH` 对应文件 |
| `IL.OFFLINE.annotations_path` | `trajectory_data/annotations.json` |
| `IL.OFFLINE.images_root` | `trajectory_data/images` |
| `MODEL.INSTRUCTION_ENCODER.embedding_file` | `SATNAV_EMBEDDING_PATH` 对应文件 |
| `MODEL.INSTRUCTION_ENCODER.vocab_size` | vocabulary JSON 中的 `vocab_size` |

生产 trajectory 保存 448 × 448 RGB。Classic 默认 `IL.OFFLINE.rgb_size: 224` 会在加载时将
图像缩放为模型训练尺寸，可以保留该设置。

模型专属字段必须保持一致：

| 字段 | Seq2Seq | CMA |
| --- | --- | --- |
| `MODEL.policy_name` | `seq2seq` | `cma` |
| `MODEL.INSTRUCTION_ENCODER.bidirectional` | `false` | `true` |
| `MODEL.INSTRUCTION_ENCODER.final_state_only` | `true` | `false` |

### 4.2 评测配置

在 `configs/local_seq2seq_eval.yaml` 和 `configs/local_cma_eval.yaml` 中：

- 将 `BASE_TASK_CONFIG_PATH` 设置为 `configs/satnav_eval_task.yaml`；
- 使用与训练相同的 `MODEL` 结构；
- 使用同一份 vocabulary 和 embedding；
- 将 `MODEL.INSTRUCTION_ENCODER.vocab_size` 设置为实际 vocabulary 大小。

评测启动脚本会根据 split 覆盖 Episode 和 scene 路径。不要修改模型 hidden size、encoder
方向、embedding size 或 backbone 后再加载已有 checkpoint。

### 4.3 本地环境变量

复制共享、训练和评测模板：

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

在 `scripts/seq2seq/.local/env.sh` 中设置 Seq2Seq 训练参数：

```bash
export SEQ2SEQ_TRAIN_CONFIG_PATH="${SEQ2SEQ_TRAIN_CONFIG_PATH:-configs/local_seq2seq_train.yaml}"
export SEQ2SEQ_EVAL_CONFIG_PATH="${SEQ2SEQ_EVAL_CONFIG_PATH:-configs/local_seq2seq_eval.yaml}"
export SEQ2SEQ_OUTPUT_ROOT="${SEQ2SEQ_OUTPUT_ROOT:-output/seq2seq_offline}"
export SEQ2SEQ_CUDA_DEVICES="${SEQ2SEQ_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
```

在 `scripts/cma/.local/env.sh` 中设置 CMA 训练参数：

```bash
export CMA_TRAIN_CONFIG_PATH="${CMA_TRAIN_CONFIG_PATH:-configs/local_cma_train.yaml}"
export CMA_EVAL_CONFIG_PATH="${CMA_EVAL_CONFIG_PATH:-configs/local_cma_eval.yaml}"
export CMA_OUTPUT_ROOT="${CMA_OUTPUT_ROOT:-output/cma}"
export CMA_CUDA_DEVICES="${CMA_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
```

在 `baselines/classic/.local/env.sh` 中设置统一评测参数：

```bash
export SATNAV_DATA_ROOT="${SATNAV_DATA_ROOT:-/path/to/SatNav-v0.1}"
export SATNAV_SCENES_DIR="${SATNAV_SCENES_DIR:-/path/to/satnav-scenes}"
export SATNAV_VOCAB_PATH="${SATNAV_VOCAB_PATH:-/path/to/satnav_v0_1_vocab.json}"

export SATNAV_SEQ2SEQ_EVAL_CONFIG="${SATNAV_SEQ2SEQ_EVAL_CONFIG:-configs/local_seq2seq_eval.yaml}"
export SATNAV_SEQ2SEQ_CHECKPOINT="${SATNAV_SEQ2SEQ_CHECKPOINT:-/path/to/seq2seq/best.pth}"

export SATNAV_CMA_EVAL_CONFIG="${SATNAV_CMA_EVAL_CONFIG:-configs/local_cma_eval.yaml}"
export SATNAV_CMA_CHECKPOINT="${SATNAV_CMA_CHECKPOINT:-/path/to/cma/best.pth}"
```


## 5. 训练 Seq2Seq

### 5.1 单 GPU Smoke

先使用单 GPU 和一个 epoch 确认数据、optimizer 与 checkpoint 链路：

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

即使只使用一张 GPU，launcher 也通过 `torchrun` 启动相同训练入口，使单卡和多卡使用一致
的配置覆盖方式。

### 5.2 多 GPU 训练

Smoke 完成后再增加 GPU、batch size、worker 和 epoch：

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

Seq2Seq 的有效 batch size 为 `PER_GPU_BATCH_SIZE × GPUS_PER_NODE`。多卡训练仅支持单节点；
`GPUS_PER_NODE` 必须与 `CUDA_DEVICES` 中的 GPU 数量一致。

默认输出结构为：

```text
output/seq2seq_offline/
├── checkpoints/
│   ├── seq2seq-v0-1/
│   │   └── best.pth
│   └── latest -> seq2seq-v0-1
└── logs/
    └── seq2seq-v0-1.log
```

## 6. 训练 CMA

### 6.1 单 GPU Smoke

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

### 6.2 多 GPU 训练

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

CMA 使用空间视觉特征与两层 recurrent state，通常比 Seq2Seq 占用更多显存。发生 OOM 时，
先减小 `PER_GPU_BATCH_SIZE`，再降低 `NUM_WORKERS`。

默认输出结构为：

```text
output/cma/
├── checkpoints/
│   ├── cma-v0-1/
│   │   └── best.pth
│   └── latest -> cma-v0-1
└── logs/
    └── cma-v0-1.log
```

## 7. Checkpoint 与继续训练

Seq2Seq 和 CMA 的 `best.pth` 都包含：

```text
config, epoch, loss, optim_state, state_dict, step_id
```

训练后可以检查 checkpoint 是否至少完成一个 optimizer step：

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

继续训练时，在相应的 local train config 中设置：

```yaml
IL:
  load_from_ckpt: true
  ckpt_to_load: /path/to/previous/best.pth
  epochs: 20
```

`epochs` 表示目标总 epoch 数，不是额外增加的 epoch 数。Trainer 会恢复模型、optimizer、
已完成 epoch 和 `step_id`，并将本次结果写入 launcher 创建的新实验目录。

## 8. 配置统一评测

评测前更新 `baselines/classic/.local/env.sh` 中的 checkpoint：

```bash
export SATNAV_SEQ2SEQ_CHECKPOINT=/path/to/seq2seq-v0-1/best.pth
export SATNAV_CMA_CHECKPOINT=/path/to/cma-v0-1/best.pth
```

检查本地 eval config 能否解析：

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

Seq2Seq 和 CMA checkpoint 会严格加载。

## 9. Smoke 评测

Smoke 建议选择 8 个 Episode，每个 Episode 最多运行 5 步。为每个 checkpoint 设置独立
输出目录：

```bash
SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/seq2seq-v0-1/val_seen/5steps/1rank \
bash scripts/classic/eval.sh seq2seq val_seen 8

SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/cma-v0-1/val_seen/5steps/1rank \
bash scripts/classic/eval.sh cma val_seen 8
```

结果结构为：

```text
<output-dir>/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

检查汇总状态：

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

CMA 使用相同的结果格式，只需替换 summary 路径。

## 10. 多 GPU 评测

在线评测按 Episode 分片，每个 rank 在一张 GPU 上加载独立模型进程。两张 GPU 的 smoke：

```bash
SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/seq2seq-v0-1/val_seen/5steps/2rank \
bash scripts/classic/eval_parallel.sh seq2seq val_seen 2 0,1 8

SATNAV_MAX_STEPS=5 \
SATNAV_CLASSIC_RUN_OUTPUT=output/baselines/classic/cma-v0-1/val_seen/5steps/2rank \
bash scripts/classic/eval_parallel.sh cma val_seen 2 0,1 8
```

`eval_parallel.sh` 会等待所有 rank 完成，然后自动运行统一聚合器。`limit` 先作用于全局
Episode 排序结果，再进行 stride sharding；上例的两个 rank 各处理 4 个 Episode。

## 11. 完整评测

正式评测将 `SATNAV_MAX_STEPS` 设置为 500，并令 `limit=-1`：

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

`val_seen` 完成后，将 split 和输出目录分别改为 `val_unseen` 再运行一次。标准 Episode 数和
评测参数参阅 [Evaluation - SatNav-v0.1 标准设置](EVALUATION.md#satnav-v01-标准设置)。

评测脚本默认启用 resume。同一运行中断后，使用完全相同的数据、checkpoint、seed、
`world_size`、最大步数和输出目录重新执行即可。评测另一个 checkpoint 或修改运行参数时，
必须使用新的输出目录。

完整结果应满足：

- `summary.json` 的 `status` 为 `complete`；
- `error_episode_count` 为 0；
- `unique_record_count` 等于当前 split 的 Episode 数；
- `metrics` 包含 `distance_to_goal`、`success`、`oracle_success`、`spl` 和 `path_length`。

结果字段、Episode 分片和 resume 规则参阅[统一评测](EVALUATION.md)。

## 12. 常见问题

### 为什么训练时找不到 RGB frame？

确认 `IL.OFFLINE.annotations_path` 指向最终 `annotations.json`，`IL.OFFLINE.images_root`
指向同一 trajectory export 的 `images/`。不要组合来自不同生成批次的 annotation 和图像。

### 为什么 vocabulary size 不匹配？

训练 config、eval config、embedding 和 `SATNAV_VOCAB_PATH` 必须来自同一次 vocabulary
构建。检查 JSON 中的 `vocab_size`，并同步更新两个模型配置中的
`MODEL.INSTRUCTION_ENCODER.vocab_size`。

### 为什么 checkpoint 无法加载？

确认选择了正确模型的 eval config。Seq2Seq 与 CMA 的 encoder、recurrent state 和参数名
不同，不能互换 checkpoint；同一模型修改 hidden size、backbone 或 instruction encoder 后
也不能直接加载旧 checkpoint。

### 为什么多 GPU 训练的所有进程都在同一张卡上？

通过 `train_offline_ddp.sh` 或 `train_ddp.sh` 启动训练，并确保 `GPUS_PER_NODE` 等于
`CUDA_DEVICES` 中的 GPU 数量。不要直接为多个进程手工设置相同的 `LOCAL_RANK`。

### 为什么评测重新运行后没有执行 Episode？

评测默认 resume，并会跳过输出目录中已经存在的 Episode。继续同一次运行时这是正常行为；
如果更换了 checkpoint 或评测参数，应指定新的 `SATNAV_CLASSIC_RUN_OUTPUT`。

### 为什么 Smoke 的 Success 和 SPL 很低？

5-step smoke 只用于检查 checkpoint 加载、模型推理、环境交互和结果写入。正式指标应来自
500-step 完整 `val_seen` 和 `val_unseen` 评测。

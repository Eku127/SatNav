# SatNav VLN 训练指南

本文档介绍如何使用 SatNav 统一配置系统训练视觉-语言导航（VLN）模型。

## 目录

1. [快速开始](#快速开始)
2. [配置系统](#配置系统)
3. [训练流程](#训练流程)
4. [评估模型](#评估模型)
5. [使用 W&B 监控](#使用-wb-监控)
6. [高级用法](#高级用法)
7. [故障排除](#故障排除)

## 快速开始

### 前置条件

确保已安装所有依赖：

```bash
conda activate satnav
pip install -r requirements.txt
```

### 准备数据

1. **准备数据集**: 确保有 SatNav 格式的数据集（包含 `reference_path`）

2. **生成词汇表**（如果数据集中没有）:

```bash
python -m satnav.utils.build_vocab \
  --dataset tests/test_data/satnav_dataset_complex.json \
  --output data/vocab.json
```

3. **（可选）生成 GloVe embeddings**:

```bash
python -m satnav.utils.build_glove_embeddings \
  --vocab data/vocab.json \
  --glove glove.6B.50d.txt \
  --output data/embeddings.json.gz
```

### 训练模型

使用统一入口点 `run.py` 训练 Seq2Seq 模型：

```bash
# 基本训练
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train

# 自定义参数训练
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.epochs 20 \
    IL.batch_size 8 \
    IL.lr 1e-4 \
    WANDB.run_name my-experiment
```

### 评估模型

```bash
# 评估最佳模型
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval

# 评估特定 split
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.SPLIT val_unseen

# 评估特定 checkpoint
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.CKPT_PATH data/checkpoints/seq2seq/ckpt.5.pth
```

## 配置系统

### 统一配置架构

SatNav 采用 VLN-CE 风格的统一配置系统：

- **单一入口点**: `run.py` 处理 train/eval/inference 所有模式
- **统一配置文件**: 一个 YAML 文件包含所有配置
- **配置块分离**: IL（训练）、EVAL（评估）、INFERENCE（推理）
- **命令行覆盖**: 支持通过命令行参数覆盖任何配置项

### 配置文件结构

实验配置文件（如 `configs/baselines/seq2seq.yaml`）包含以下部分：

```yaml
# ===== 基础配置 =====
BASE_TASK_CONFIG_PATH: configs/debug_vln_task.yaml  # 任务配置路径
TRAINER_NAME: recollect_trainer                     # Trainer 名称
CHECKPOINT_FOLDER: data/checkpoints/seq2seq         # Checkpoint 目录
RESULTS_DIR: data/results/seq2seq                   # 结果目录
TORCH_GPU_ID: 0                                     # GPU ID
NUM_PROCESSES: 1                                    # 进程数

# ===== 训练配置 (IL) =====
IL:
  lr: 2.5e-4              # 学习率
  batch_size: 5           # Batch size
  epochs: 10              # 训练轮数
  load_from_ckpt: false   # 是否从 checkpoint 加载
  ckpt_to_load: data/checkpoints/seq2seq/best.pth  # Checkpoint 路径
  
  RECOLLECT_TRAINER:
    preload_size: 5       # 预加载的 episode 数量
    max_traj_len: 500     # 最大轨迹长度

# ===== 评估配置 (EVAL) =====
EVAL:
  SPLIT: val_seen         # 评估数据集 split
  EPISODE_COUNT: -1       # 评估 episode 数量（-1 表示全部）
  SAVE_RESULTS: true      # 是否保存结果到 JSON
  CKPT_PATH: data/checkpoints/seq2seq/best.pth  # 评估的 checkpoint

# ===== 推理配置 (INFERENCE) =====
INFERENCE:
  SPLIT: test
  CKPT_PATH: data/checkpoints/seq2seq/best.pth
  PREDICTIONS_FILE: data/predictions/seq2seq_test.json

# ===== 模型配置 (MODEL) =====
MODEL:
  policy_name: seq2seq
  # ... 模型架构配置

# ===== W&B 配置 =====
WANDB:
  project: satnav-vln
  run_name: seq2seq-baseline
  mode: online            # online, offline, 或 disabled
```

### 配置加载顺序

配置按以下顺序合并（后者覆盖前者）：

1. **Task Config**: `BASE_TASK_CONFIG_PATH` 指定的任务配置
2. **Experiment Config**: 实验配置文件（`--exp-config` 参数）
3. **Command Line Opts**: 命令行参数覆盖

### 参考配置文件

- **默认配置模板**: `configs/default.yaml`（参考文档）
- **实验配置**: `configs/baselines/seq2seq.yaml`
- **任务配置**: `configs/debug_vln_task.yaml`

## 训练流程

### 1. RecollectTrainer 工作流程

RecollectTrainer 实时从环境收集训练数据：

1. **GT 轨迹提取**: 使用 `ReferencePathFollower` 将 `reference_path` 转换为离散动作序列
2. **数据收集**: 在环境中执行 GT 动作，收集 observations
3. **Instruction Tokenization**: 将自然语言指令转换为 token indices
4. **Batch 训练**: 使用 Teacher Forcing 训练策略网络
5. **Best Model 保存**: 跟踪并保存最佳模型

### 2. 训练输出

训练过程中会输出以下信息：

```
================================================================================
SatNav VLN - TRAIN Mode
================================================================================
Experiment config: configs/baselines/seq2seq.yaml
Loading task config: configs/debug_vln_task.yaml
Trainer: recollect_trainer
================================================================================

Creating environment...
Loading vocabulary...
Vocabulary size: 50
Extracting GT trajectories from reference paths...
Extracted 10 trajectories

Initializing policy...
Loading policy: seq2seq
Agent parameters: 25,609,651. Trainable: 2,047,396

Training for 10 epochs
Epoch 1/10: 100%|████████| 11/11 [00:07<00:00, 1.38it/s, loss=1.18, batch_time=0.09s]

Epoch 1 completed:
  Average Loss: 1.3019
  Epoch Time: 7.48s
  Number of batches: 11
  New best loss: 1.3019, saving checkpoint...
```

### 3. Checkpoint 管理

- **自动保存**: 每个 epoch 后，如果 loss 降低，自动保存为 `best.pth`
- **手动保存**: 可以在配置中指定保存策略
- **加载**: 设置 `IL.load_from_ckpt: true` 和 `IL.ckpt_to_load`

## 评估模型

### 基本评估

```bash
# 使用默认配置评估
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval
```

输出示例：

```
================================================================================
SatNav VLN - EVAL Mode
================================================================================
Starting Evaluation
================================================================================

Evaluating checkpoint: data/checkpoints/seq2seq/best.pth

Evaluation configuration:
  Split: val_seen
  Episode count: all
  Save results: True

Checkpoint loaded successfully:
  Epoch: 2
  Step: 6
  Loss: 1.1313
  Parameters: 25,609,651
```

### 评估不同 Split

```bash
# 评估 val_unseen
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.SPLIT val_unseen

# 评估特定数量的 episodes
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.SPLIT val_unseen \
    EVAL.EPISODE_COUNT 100
```

**重要说明**：`EVAL.SPLIT` 会在评估时**自动同步**到 `DATASET.SPLIT`，确保加载正确的数据集文件。你**不需要**手动修改 task config (`debug_vln_task.yaml`) 中的 `DATASET.SPLIT`。

评估时会看到类似输出：
```
Synchronizing EVAL.SPLIT (val_unseen) to DATASET.SPLIT
  Updated DATASET.SPLIT to: val_unseen
```

数据集路径中的 `{split}` 占位符会根据同步后的 `DATASET.SPLIT` 自动替换。

### 评估特定 Checkpoint

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.CKPT_PATH data/checkpoints/seq2seq/ckpt.5.pth
```

### 评估结果

评估结果会保存到 `RESULTS_DIR` 目录（如果 `EVAL.SAVE_RESULTS: true`）：

**文件位置**: `data/results/seq2seq/eval_ckpt_{checkpoint_index}_{split}.json`

**JSON格式**:
```json
{
  "spl": 0.4523,
  "success": 0.5200,
  "distance_to_goal": 5.2341,
  "path_length": 125.34,
  "steps_taken": 45.2,
  "num_episodes": 50,
  "split": "val_seen",
  "checkpoint_index": 0
}
```

**指标说明**:
- `spl`: Success weighted by Path Length（成功率加权路径长度）
- `success`: 成功率（到达目标并执行STOP）
- `distance_to_goal`: 最终位置到目标的平均距离（米）
- `path_length`: 平均路径长度（米）
- `steps_taken`: 平均步数
- `num_episodes`: 评估的episode数量
- `split`: 数据集split
- `checkpoint_index`: Checkpoint索引

### 视频生成（可选）

启用视频生成：

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    'VIDEO_OPTION' '["disk"]'
```

视频将保存到 `VIDEO_DIR` 目录（默认: `data/videos/seq2seq/`），文件名格式: `episode_{episode_id}_ckpt_{checkpoint_index}.mp4`

**注意**: 视频生成需要在 `TASK.MEASUREMENTS` 中启用 `TOP_DOWN_MAP`。

## 使用 W&B 监控

### 1. 初始化 W&B

首次使用需要登录：

```bash
wandb login
```

### 2. 配置 W&B

在配置文件中设置：

```yaml
WANDB:
  project: "satnav-vln"        # 项目名称
  run_name: "seq2seq-exp1"     # 实验名称
  entity: null                 # 团队名称（可选）
  mode: "online"               # online: 实时上传, offline: 本地保存, disabled: 关闭
```

或通过命令行覆盖：

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    WANDB.run_name my-experiment \
    WANDB.mode online
```

### 3. 查看训练指标

训练期间，以下指标会自动记录到 W&B:

- `train/loss`: 每个 batch 的损失
- `train/epoch`: 当前 epoch
- `train/learning_rate`: 学习率
- `epoch/loss`: Epoch 平均损失
- `epoch/time`: Epoch 训练时间
- `epoch/number`: Epoch 编号

访问 https://wandb.ai/[your-username]/satnav-vln 查看实时指标。

### 4. 保存 Artifacts

最佳模型会自动上传到 W&B（如果启用）。

### 5. 离线模式

如果没有网络连接：

```yaml
WANDB:
  mode: "offline"
```

训练完成后，使用 `wandb sync` 上传日志。

### 6. 禁用 W&B

完全禁用 W&B 日志：

```yaml
WANDB:
  mode: "disabled"
```

或通过命令行：

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    WANDB.mode disabled
```

## 高级用法

### 命令行覆盖配置

可以通过命令行覆盖配置文件中的任何参数：

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.lr 1e-4 \
    IL.batch_size 10 \
    IL.epochs 20 \
    WANDB.run_name my-custom-experiment \
    TORCH_GPU_ID 1
```

**格式**: `KEY1 VALUE1 KEY2 VALUE2 ...`

**支持嵌套**: `PARENT.CHILD VALUE`

### 从 Checkpoint 继续训练

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.load_from_ckpt true \
    IL.ckpt_to_load data/checkpoints/seq2seq/best.pth
```

### 使用不同的 Task Config

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    BASE_TASK_CONFIG_PATH configs/vln_task.yaml
```

### 调试模式（小规模训练）

快速测试训练流程：

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.epochs 1 \
    IL.batch_size 2 \
    IL.RECOLLECT_TRAINER.preload_size 2
```

### Trainer Registry 系统

SatNav 使用 registry 系统管理 trainer：

```python
from satnav.training import get_trainer, register_trainer

# 获取 trainer
trainer_class = get_trainer("recollect_trainer")

# 注册自定义 trainer
@register_trainer("my_trainer")
class MyTrainer(BaseILTrainer):
    def train(self):
        # 自定义训练逻辑
        pass
```

## 故障排除

### 问题 1: CUDA Out of Memory

**症状**: `torch.cuda.OutOfMemoryError`

**解决方案**: 减小 batch size 或 preload_size

```bash
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.batch_size 2 \
    IL.RECOLLECT_TRAINER.preload_size 3
```

### 问题 2: 找不到词汇表

**错误信息**: `Vocabulary file not found` 或 `KeyError: 'instruction_vocab'`

**解决方案**: 确保数据集包含 `instruction_vocab` 或生成词汇表文件：

```bash
python -m satnav.utils.build_vocab \
    --dataset your_dataset.json \
    --output data/vocab.json
```

### 问题 3: Reference Path 为空

**错误信息**: `Episode X has invalid reference_path`

**解决方案**: 检查数据集中的 `reference_path` 字段，确保至少有 2 个 waypoints。

### 问题 4: Trainer 未注册

**错误信息**: `Trainer 'xxx' is not registered`

**解决方案**: 检查 trainer 名称拼写，或确保 trainer 类已导入：

```python
# 在 satnav/training/__init__.py 中
from satnav.training.recollect_trainer import RecollectTrainer
```

### 问题 5: 配置覆盖不生效

**症状**: 命令行参数似乎被忽略

**解决方案**: 检查参数格式，确保使用空格分隔：

```bash
# ✅ 正确
python run.py ... IL.lr 1e-4 IL.batch_size 8

# ❌ 错误
python run.py ... IL.lr=1e-4 IL.batch_size=8
```

### 问题 6: W&B 连接超时

**解决方案**: 使用离线模式：

```bash
python run.py ... WANDB.mode offline
```

### 问题 7: GPU 不可用

如果没有 GPU，训练器会自动使用 CPU，但速度较慢。可以显式指定：

```bash
python run.py ... TORCH_GPU_ID -1  # 使用 CPU
```

## 配置系统迁移指南

### 从旧系统迁移

如果你使用过旧的训练命令：

```bash
# ❌ 旧方式（已弃用）
python -m satnav.training.recollect_trainer \
    --config configs/training/recollect_trainer.yaml \
    --model-config configs/baselines/seq2seq.yaml
```

迁移到新系统：

```bash
# ✅ 新方式
python run.py \
    --exp-config configs/baselines/seq2seq.yaml \
    --run-type train
```

### 配置文件迁移

1. **合并配置**: 将 training config 和 model config 合并到一个文件
2. **添加配置块**: 确保有 IL、EVAL、INFERENCE 配置块
3. **更新路径**: 使用 `BASE_TASK_CONFIG_PATH` 指定 task config

参考 `configs/baselines/seq2seq.yaml` 和 `configs/default.yaml`。

## 参考资料

- [Seq2Seq 模型文档](../models/SEQ2SEQ_IMPLEMENTATION.md)
- [CMA 模型文档](../models/CMA_IMPLEMENTATION.md)
- [Embedding 生成指南](../EMBEDDING_GUIDE.md)
- [默认配置模板](../../configs/default.yaml)
- [VLN-CE 参考实现](https://github.com/jacobkrantz/VLN-CE)

## 下一步

- **完整评估实现**: 添加环境 rollout 和 metrics 计算
- **Inference 模式**: 实现 predictions 生成
- **DAgger 训练**: 探索 dataset aggregation 训练策略
- **模型集成**: 添加 ensemble 支持
- **分布式训练**: 支持多 GPU 训练

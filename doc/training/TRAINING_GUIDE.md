# SatNav VLN 训练指南

本文档介绍如何使用 SatNav 训练框架训练视觉-语言导航（VLN）模型。

## 目录

1. [快速开始](#快速开始)
2. [配置说明](#配置说明)
3. [训练流程](#训练流程)
4. [使用 W&B 监控](#使用-wb-监控)
5. [高级用法](#高级用法)
6. [故障排除](#故障排除)

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

使用 RecollectTrainer 训练 Seq2Seq 模型：

```bash
python -m satnav.training.recollect_trainer \
  --config configs/training/recollect_trainer.yaml \
  --model-config configs/baselines/seq2seq.yaml
```

**说明：** 任务配置（DATASET, SIMULATOR, TASK, ENVIRONMENT）会根据训练配置中的 `TASK_CONFIG_PATH` 字段自动加载。

## 配置说明

### 训练配置 (configs/training/recollect_trainer.yaml)

```yaml
IL:
  lr: 2.5e-4              # 学习率
  batch_size: 5           # Batch size
  epochs: 10              # 训练轮数
  load_from_ckpt: false   # 是否从 checkpoint 加载
  ckpt_to_load: ""        # Checkpoint 路径

CHECKPOINT_FOLDER: "data/checkpoints/seq2seq"  # Checkpoint 保存路径

WANDB:
  project: "satnav-vln"           # W&B 项目名称
  run_name: "seq2seq-baseline"    # W&B 运行名称
  entity: null                    # W&B 团队名称（可选）
  mode: "online"                  # online, offline, 或 disabled

TORCH_GPU_ID: 0           # GPU ID
```

### RecollectTrainer 配置

```yaml
IL:
  RECOLLECT_TRAINER:
    preload_size: 10      # 预加载的 episode 数量
    max_traj_len: 500     # 最大轨迹长度
```

### 模型配置 (configs/baselines/seq2seq.yaml)

包含模型架构配置，如编码器参数、隐藏层大小等。详见 [SEQ2SEQ_BASELINE.md](../models/SEQ2SEQ_BASELINE.md)。

### 数据集配置 (configs/vln_task.yaml)

包含数据集路径、scene 配置、simulator 参数等。

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
Creating environment...
Loading vocabulary...
Vocabulary size: 50
Extracting GT trajectories from reference paths...
Extracted 10 trajectories
Initializing policy...
Loading policy: seq2seq
Agent parameters: 1234567. Trainable: 1234567
Training for 10 epochs
Epoch 1/10: 100%|████████| loss: 2.3456
Epoch 1 completed:
  Average Loss: 2.3456
  Epoch Time: 45.23s
  New best loss: 2.3456, saving checkpoint...
```

### 3. Checkpoint 管理

- **保存**: 每个 epoch 后，如果 loss 降低，自动保存为 `best.pth`
- **加载**: 设置 `IL.load_from_ckpt: true` 和 `IL.ckpt_to_load: path/to/ckpt.pth`

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
  mode: "online"               # online: 实时上传, offline: 本地保存, disabled: 关闭
```

### 3. 查看训练指标

训练期间，以下指标会自动记录到 W&B:

- `train/loss`: 每个 batch 的损失
- `train/epoch`: 当前 epoch
- `train/learning_rate`: 学习率
- `epoch/loss`: Epoch 平均损失
- `epoch/time`: Epoch 训练时间

访问 https://wandb.ai/[your-username]/satnav-vln 查看实时指标。

### 4. 保存 Artifacts

最佳模型会自动上传到 W&B：

```python
wandb.save('data/checkpoints/seq2seq/best.pth')
```

## 高级用法

### 命令行覆盖配置

可以通过命令行覆盖配置参数：

```bash
python -m satnav.training.recollect_trainer \
  --config configs/training/recollect_trainer.yaml \
  --model-config configs/baselines/seq2seq.yaml \
  --opts \
    IL.lr=1e-4 \
    IL.batch_size=10 \
    WANDB.run_name=my-experiment
```

### 从 Checkpoint 继续训练

```bash
python -m satnav.training.recollect_trainer \
  --config configs/training/recollect_trainer.yaml \
  --model-config configs/baselines/seq2seq.yaml \
  --opts \
    IL.load_from_ckpt=true \
    IL.ckpt_to_load=data/checkpoints/seq2seq/best.pth
```

### 离线模式（无网络）

如果没有网络连接，可以使用离线模式：

```yaml
WANDB:
  mode: "offline"
```

训练完成后，使用 `wandb sync` 上传日志。

### 禁用 W&B

完全禁用 W&B 日志：

```yaml
WANDB:
  mode: "disabled"
```

## 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**: 减小 batch size

```yaml
IL:
  batch_size: 2  # 从 5 减小到 2
```

### 问题 2: 找不到词汇表

**错误信息**: `Vocabulary file not found`

**解决方案**: 确保数据集包含 `instruction_vocab` 或生成词汇表文件

### 问题 3: Reference Path 为空

**错误信息**: `Episode X has invalid reference_path`

**解决方案**: 检查数据集中的 `reference_path` 字段，确保至少有 2 个waypoints

### 问题 4: W&B 连接超时

**解决方案**: 使用离线模式或禁用 W&B：

```bash
--opts WANDB.mode=offline
```

### 问题 5: GPU 不可用

如果没有 GPU，训练器会自动使用 CPU，但速度较慢。

## 参考资料

- [Seq2Seq 模型文档](../models/SEQ2SEQ_BASELINE.md)
- [Embedding 生成指南](../EMBEDDING_GUIDE.md)
- [VLN-CE 参考实现](https://github.com/jacobkrantz/VLN-CE)

## 下一步

- 探索 DAgger 训练（未来实现）
- 实现评估流程
- 添加模型集成


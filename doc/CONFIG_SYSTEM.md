# SatNav 统一配置系统

本文档介绍 SatNav 的统一配置系统架构和使用方法。

## 概述

SatNav 采用 VLN-CE 风格的统一配置系统，提供：

- **单一入口点**: `run.py` 处理 train/eval/inference 所有模式
- **统一配置**: 一个 YAML 文件包含训练、评估、推理的所有配置
- **灵活覆盖**: 支持通过命令行参数覆盖任何配置项
- **Trainer Registry**: 基于注册表的 trainer 管理系统

## 架构设计

### 配置层次结构

```
┌─────────────────────────────────────┐
│     Command Line Overrides          │  ← 最高优先级
├─────────────────────────────────────┤
│     Experiment Config               │
│  (configs/baselines/seq2seq.yaml)   │
├─────────────────────────────────────┤
│     Task Config                     │
│  (configs/debug_vln_task.yaml)      │
└─────────────────────────────────────┘
       ↓ 合并 (OmegaConf.merge)
┌─────────────────────────────────────┐
│     Final Unified Config            │
└─────────────────────────────────────┘
```

### 入口点系统

```
run.py
  ├── --run-type train    → trainer.train()
  ├── --run-type eval     → trainer.eval()
  └── --run-type inference → trainer.inference()
```

### Trainer Registry

```python
# 注册 trainer
@register_trainer("recollect_trainer")
class RecollectTrainer(BaseILTrainer):
    ...

# 通过配置获取 trainer
trainer_class = get_trainer(config.TRAINER_NAME)
trainer = trainer_class(config)
```

## 配置文件结构

### 1. Experiment Config (实验配置)

位置: `configs/baselines/seq2seq.yaml`

包含所有模式的完整配置：

```yaml
# 基础设置
BASE_TASK_CONFIG_PATH: configs/debug_vln_task.yaml
TRAINER_NAME: recollect_trainer
CHECKPOINT_FOLDER: data/checkpoints/seq2seq
RESULTS_DIR: data/results/seq2seq
TORCH_GPU_ID: 0

# 训练配置
IL:
  lr: 2.5e-4
  batch_size: 5
  epochs: 10
  load_from_ckpt: false
  ckpt_to_load: data/checkpoints/seq2seq/best.pth
  RECOLLECT_TRAINER:
    preload_size: 5
    max_traj_len: 500

# 评估配置
EVAL:
  SPLIT: val_seen
  EPISODE_COUNT: -1
  SAVE_RESULTS: true
  CKPT_PATH: data/checkpoints/seq2seq/best.pth

# 推理配置
INFERENCE:
  SPLIT: test
  CKPT_PATH: data/checkpoints/seq2seq/best.pth
  PREDICTIONS_FILE: data/predictions/seq2seq_test.json

# 模型配置
MODEL:
  policy_name: seq2seq
  INSTRUCTION_ENCODER:
    # ... 指令编码器配置
  RGB_ENCODER:
    # ... 视觉编码器配置
  SEQ2SEQ:
    # ... Seq2Seq 模型特定配置

# W&B 配置
WANDB:
  project: satnav-vln
  run_name: seq2seq-baseline
  mode: online
```

### 2. Task Config (任务配置)

位置: `configs/debug_vln_task.yaml`

包含环境、任务、数据集配置：

```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  FORWARD_STEP_SIZE: 10
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 512
    HEIGHT: 512
    HFOV: 90

TASK:
  TYPE: VLN
  SUCCESS_DISTANCE: 10.0
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, SPL, PATH_LENGTH]

DATASET:
  SPLIT: train
  DATA_PATH: data/debug_data/{split}/{split}.json
  SCENES_DIR: data/scene_datasets/
```

### 3. Default Config (默认配置)

位置: `configs/default.yaml`

提供所有配置选项的参考文档和默认值。

## 使用方法

### 基本用法

```bash
# 训练
python run.py \
    --exp-config configs/baselines/seq2seq.yaml \
    --run-type train

# 评估
python run.py \
    --exp-config configs/baselines/seq2seq.yaml \
    --run-type eval

# 推理
python run.py \
    --exp-config configs/baselines/seq2seq.yaml \
    --run-type inference
```

### 命令行覆盖

可以覆盖任何配置项：

```bash
python run.py \
    --exp-config configs/baselines/seq2seq.yaml \
    --run-type train \
    IL.lr 1e-4 \
    IL.batch_size 10 \
    WANDB.run_name my-experiment
```

**格式规则**:
- 参数以空格分隔: `KEY1 VALUE1 KEY2 VALUE2`
- 支持嵌套路径: `PARENT.CHILD.SUBCHILD VALUE`
- 布尔值: `true` 或 `false` (小写)
- 数字: 直接写数值
- 字符串: 直接写文本（如包含空格需引号）

### 配置块说明

#### IL (Imitation Learning / Training)

训练相关的配置：

```yaml
IL:
  lr: 2.5e-4                # 学习率
  batch_size: 5             # Batch size
  epochs: 10                # 训练轮数
  load_from_ckpt: false     # 是否加载 checkpoint
  ckpt_to_load: ""          # Checkpoint 路径
  
  # Trainer 特定配置
  RECOLLECT_TRAINER:
    preload_size: 5         # 预加载 episodes 数量
    max_traj_len: 500       # 最大轨迹长度
```

#### EVAL (Evaluation)

评估相关的配置：

```yaml
EVAL:
  SPLIT: val_seen           # 评估 split (train/val_seen/val_unseen/test)
  EPISODE_COUNT: -1         # 评估 episode 数量 (-1 表示全部)
  SAVE_RESULTS: true        # 是否保存结果到 JSON
  CKPT_PATH: ""             # 评估的 checkpoint 路径
```

**重要说明**：`EVAL.SPLIT` 会在评估时**自动同步**到 `DATASET.SPLIT`，确保加载正确的数据集文件。这是参考 VLN-CE 的设计，评估时不需要手动修改 task config 中的 `DATASET.SPLIT`。

数据集路径使用 `{split}` 占位符（如 `data/debug_data/{split}/{split}.json`），会根据同步后的 `DATASET.SPLIT` 自动替换。

#### INFERENCE (Inference)

推理相关的配置：

```yaml
INFERENCE:
  SPLIT: test               # 推理 split
  CKPT_PATH: ""             # 推理的 checkpoint 路径
  PREDICTIONS_FILE: ""      # 预测结果保存路径
```

#### MODEL

模型架构配置：

```yaml
MODEL:
  policy_name: seq2seq      # 模型名称（从 registry 查找）
  
  INSTRUCTION_ENCODER:      # 指令编码器
    vocab_size: null
    embedding_size: 50
    hidden_size: 128
    # ...
  
  RGB_ENCODER:              # 视觉编码器
    cnn_type: TorchVisionResNet50
    output_size: 256
    trainable: false
    # ...
  
  SEQ2SEQ:                  # Seq2Seq 模型特定配置
    hidden_size: 512
    rnn_type: GRU
    use_prev_action: true
```

#### WANDB

W&B 日志配置：

```yaml
WANDB:
  project: "satnav-vln"     # W&B 项目名称
  run_name: "experiment"    # 运行名称
  entity: null              # 团队名称（可选）
  mode: "online"            # online/offline/disabled
```

## 配置加载流程

### 内部实现

```python
# 1. 加载实验配置
config = OmegaConf.load(args.exp_config)

# 2. 加载并合并 task 配置
if "BASE_TASK_CONFIG_PATH" in config:
    task_config = OmegaConf.load(config.BASE_TASK_CONFIG_PATH)
    config = OmegaConf.merge(task_config, config)

# 3. 应用命令行覆盖
if args.opts:
    override_config = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(config, override_config)

# 4. 从 registry 获取 trainer
trainer_class = get_trainer(config.TRAINER_NAME)

# 5. 实例化并运行
trainer = trainer_class(config)
if args.run_type == "train":
    trainer.train()
elif args.run_type == "eval":
    trainer.eval()
```

### 配置访问

在代码中访问配置：

```python
# 直接访问
learning_rate = self.config.IL.lr
batch_size = self.config.IL.batch_size

# 使用 OmegaConf.select() (推荐，支持默认值)
from omegaconf import OmegaConf

split = OmegaConf.select(self.config, 'EVAL.SPLIT', default='val_seen')
episode_count = OmegaConf.select(self.config, 'EVAL.EPISODE_COUNT', default=-1)
```

## Registry 系统

### Trainer Registry

注册 trainer：

```python
from satnav.training.registry import register_trainer

@register_trainer("my_trainer")
class MyTrainer(BaseILTrainer):
    def train(self):
        # 训练逻辑
        pass
    
    def eval(self):
        # 评估逻辑
        pass
```

使用 trainer：

```python
from satnav.training import get_trainer

# 通过名称获取
trainer_class = get_trainer("my_trainer")
trainer = trainer_class(config)
trainer.train()

# 列出所有已注册的 trainer
from satnav.training.registry import list_trainers
print(list_trainers())  # ['recollect_trainer', 'my_trainer']
```

### Model Registry

类似的 registry 系统用于模型：

```python
from satnav.models import ModelRegistry

# 注册模型
@ModelRegistry.register_model("my_model")
class MyModel(ILPolicy):
    ...

# 使用模型
model_class = ModelRegistry.get_model("my_model")
model = model_class.from_config(config, obs_space, act_space)
```

## 最佳实践

### 1. 组织实验配置

为不同的实验创建独立的配置文件：

```
configs/
├── baselines/
│   ├── seq2seq.yaml              # 基本 Seq2Seq
│   ├── seq2seq_large.yaml        # 大模型变体
│   └── seq2seq_finetune.yaml     # 微调配置
├── default.yaml                  # 默认配置参考
└── debug_vln_task.yaml           # 调试任务配置
```

### 2. 使用命令行覆盖进行快速实验

```bash
# 快速测试不同学习率
for lr in 1e-4 5e-4 1e-3; do
    python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
        IL.lr $lr \
        WANDB.run_name seq2seq-lr-$lr
done
```

### 3. 版本控制配置

- ✅ 提交实验配置文件到 git
- ✅ 在配置中记录实验描述
- ✅ 使用有意义的 checkpoint 和结果目录名称

### 4. 配置验证

在训练前验证配置：

```bash
# 查看最终合并的配置
python -c "
from omegaconf import OmegaConf
config = OmegaConf.load('configs/baselines/seq2seq.yaml')
task = OmegaConf.load(config.BASE_TASK_CONFIG_PATH)
final = OmegaConf.merge(task, config)
print(OmegaConf.to_yaml(final))
" | less
```

## 与 VLN-CE 的对比

| 特性 | SatNav | VLN-CE |
|------|--------|--------|
| 入口点 | `run.py` | `run.py` |
| 配置格式 | YAML + OmegaConf | YAML + yacs CN |
| Registry | ✅ Trainer + Model | ✅ baseline_registry |
| 配置块 | IL/EVAL/INFERENCE | IL/EVAL/INFERENCE |
| 命令行覆盖 | ✅ `KEY VALUE` | ✅ `opts` list |
| Task Config | ✅ 自动加载 | ✅ 自动加载 |

**主要区别**:
- SatNav 使用 OmegaConf（更现代，支持 Hydra）
- VLN-CE 使用 yacs（Detectron2 风格）

## 故障排除

### 配置解析错误

**问题**: `Error parsing config overrides`

**解决**: 检查参数格式，使用空格分隔：

```bash
# ✅ 正确
IL.lr 1e-4 IL.batch_size 8

# ❌ 错误
IL.lr=1e-4 IL.batch_size=8
```

### Trainer 未找到

**问题**: `Trainer 'xxx' is not registered`

**解决**: 
1. 检查拼写
2. 确保 trainer 已导入（在 `satnav/training/__init__.py`）
3. 检查 decorator: `@register_trainer("name")`

### 配置覆盖不生效

**问题**: 命令行参数被忽略

**解决**: 
1. 确保参数在配置块中存在
2. 使用正确的嵌套路径 (e.g., `IL.lr` 而不是 `lr`)
3. 检查参数类型匹配

## 参考资料

- [Training Guide](training/TRAINING_GUIDE.md) - 训练指南
- [Default Config](../configs/default.yaml) - 默认配置模板
- [VLN-CE Config](https://github.com/jacobkrantz/VLN-CE) - VLN-CE 参考
- [OmegaConf Docs](https://omegaconf.readthedocs.io/) - OmegaConf 文档


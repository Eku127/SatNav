# SatNav: 连续状态 VLN 评测平台

## 项目简介

SatNav 是一个用于评测连续状态下视觉语言导航（VLN）的测试平台。该平台设计简洁，专注于最小功能实现，不依赖 Habitat 组件，使用自定义的 **satsim 卫星地图仿真器**。

**特点**：
- 使用卫星地图作为场景（而非3D室内场景）
- 位置使用地理坐标（经度、纬度、高度）
- 旋转使用roll角度（航向角）
- 图像通过从卫星地图crop生成

## 设计特点

- ✅ **最小化实现**：只实现核心功能，代码简洁
- ✅ **无 Registry**：简化设计，直接导入使用
- ✅ **单一任务**：专注于 VLN 任务
- ✅ **独立实现**：不依赖 Habitat 任何组件
- ✅ **接口预留**：satsim 仿真器接口完整但待实现

## 项目结构

```
satnav/
├── satnav/                    # 主包
│   ├── core/                 # 核心组件
│   │   ├── env.py           # 环境类
│   │   ├── simulator.py     # 仿真器接口
│   │   ├── episode.py        # Episode数据定义
│   │   └── config.py         # 配置系统
│   ├── task/                 # VLN任务
│   │   ├── vln_task.py       # VLN任务类
│   │   ├── sensors.py        # 传感器
│   │   ├── actions.py        # 动作定义
│   │   └── measures.py       # 评价指标
│   ├── dataset/              # 数据集
│   │   └── r2r_dataset.py    # R2R数据集加载器
│   └── sims/                 # 仿真器包装
│       └── satsim_wrapper.py # satsim包装（接口待实现）
├── configs/                   # 配置文件
│   └── vln_task.yaml         # VLN任务配置
├── examples/                  # 示例代码
│   └── example_vln.py        # 使用示例
├── doc/                      # 文档目录
│   └── refactor/             # 设计文档
│       ├── ARCHITECTURE.md
│       ├── ARCHITECTURE_DIAGRAM.md
│       ├── CAMERA_MODEL.md
│       ├── CODE_STRUCTURE.md
│       ├── DESIGN_EVALUATION.md
│       └── DESIGN_SUMMARY.md
├── setup.py                  # 安装脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 核心组件

### 1. 环境类 (Env)
连接仿真器、任务和数据集，提供统一的 `reset()` 和 `step()` 接口。

### 2. 仿真器接口 (SatSimWrapper)
定义仿真器抽象接口，包装 satsim 卫星地图仿真器。所有方法当前为 `NotImplementedError`，等待 satsim 实现。

**特点**：
- 位置：经度、纬度、高度（用xyz坐标系统表示）
- 旋转：roll角度（航向角，0-360度，0表示正北）
- 图像：从卫星地图crop生成，而非3D渲染

### 3. VLN 任务 (VLNTask)
定义 VLN 任务逻辑，管理传感器（RGB、Instruction）和评价指标（SPL、Success、DistanceToGoal）。

**注意**：仅使用RGB图像作为视觉观测，卫星地图不提供深度信息。

### 4. 数据集 (R2RDataset)
加载 R2R 格式的 VLN 数据集（JSON.gz 文件）。

### 5. 配置系统
加载和管理 YAML 配置文件，格式类似 VLN-CE。

## 快速开始

### 1. 环境安装

#### 方法一：使用 Conda（推荐）

```bash
# 创建conda环境（Python 3.8+）
conda create -n satnav python=3.8
conda activate satnav

# 安装依赖
cd /path/to/satnav
pip install -r requirements.txt

# 或者以开发模式安装
pip install -e .
```

#### 方法二：使用 pip + venv

```bash
# 创建虚拟环境
python3 -m venv satnav_env
source satnav_env/bin/activate  # Linux/Mac
# 或
satnav_env\Scripts\activate  # Windows

# 安装依赖
cd /path/to/satnav
pip install -r requirements.txt

# 或者以开发模式安装
pip install -e .
```

#### 依赖说明

项目依赖（见 `requirements.txt`）：
- `numpy>=1.19.0` - 数值计算
- `omegaconf>=2.1.0` - 配置管理
- `pyyaml>=5.4.0` - YAML文件解析
- `attrs>=21.0.0` - 数据类支持

**注意**：本项目不依赖 `gym`，简化实现。

### 2. 验证安装

```bash
# 验证导入
python3 -c "from satnav.core import VLNEpisode, InstructionData, NavigationGoal; print('✓ Installation successful')"
```

### 3. 运行测试

```bash
# 确保在conda环境中
conda activate satnav

# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_config.py      # 配置加载测试
pytest tests/test_utils.py       # 工具函数测试（测地距离）
pytest tests/test_dataset.py     # 数据集加载测试

# 运行测试并显示覆盖率
pytest --cov=satnav --cov-report=html

# 运行测试（详细输出）
pytest -v

# 运行测试并显示覆盖率报告
pytest --cov=satnav --cov-report=term-missing
```

### 4. 准备数据

将 R2R 数据集放在 `data/datasets/R2R/` 目录下，格式如下：
```
data/datasets/R2R/
├── train/
│   └── train.json.gz
├── val_seen/
│   └── val_seen.json.gz
└── val_unseen/
    └── val_unseen.json.gz
```

### 5. 配置环境

编辑 `configs/vln_task.yaml`，设置数据集路径和场景路径。

### 6. 运行示例

```python
from satnav.core import Env
from satnav.core.config import load_config
from satnav.dataset import R2RDataset

# 加载配置
config = load_config("configs/vln_task.yaml")

# 创建数据集
dataset = R2RDataset(config.DATASET)

# 创建环境
env = Env(config, dataset)

# 运行一个 episode
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, info = env.step(action)

# 获取评价指标
metrics = env.get_metrics()
print(f"Success: {metrics['success']}")
print(f"SPL: {metrics['spl']}")
```

## 配置格式

配置文件格式类似 VLN-CE：

```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  FORWARD_STEP_SIZE: 0.25
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 224
    HEIGHT: 224
    # Horizontal Field of View in degrees. This simulates a drone's camera.
    # The ground area captured in the image is calculated based on HFOV and the agent's current altitude.
    HFOV: 90

TASK:
  TYPE: VLN-v0
  SUCCESS_DISTANCE: 3.0
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, SPL, PATH_LENGTH]

DATASET:
  # TYPE: Dataset type identifier (kept for compatibility, not used in current implementation)
  # Note: This field is kept for compatibility with habitat-lab/VLN-CE config format.
  # Since satnav doesn't use a registry system, this field is informational only and
  # doesn't affect dataset loading. The actual dataset class is always SatNavDataset.
  TYPE: SatNav
  SPLIT: train
  DATA_PATH: data/datasets/R2R/{split}/{split}.json.gz
  SCENES_DIR: data/scene_datasets/
```

**配置字段说明**:
- `DATASET.TYPE`: 数据集类型标识符。保留此字段是为了与 habitat-lab/VLN-CE 的配置格式兼容。由于 satnav 不使用 registry 系统，此字段仅为信息性，不影响实际的数据集加载。实际使用的数据集类始终是 `SatNavDataset`。
- `DATASET.SPLIT`: 数据集划分（train/val/test）
- `DATASET.DATA_PATH`: 数据集文件路径，支持 `{split}` 占位符
- `DATASET.SCENES_DIR`: 场景数据目录（可选）

## 评价指标

- **Success**: 是否到达目标（距离 < success_distance）
- **SPL**: Success weighted by Path Length
- **DistanceToGoal**: 到目标的距离
- **PathLength**: 已走路径长度
- **NDTW**: Normalized Dynamic Time Warping（可选）

## satsim 接口（待实现）

所有仿真器方法当前为 `NotImplementedError`，需要实现：

- `reset(scene_id)`: 重置仿真器，加载卫星地图场景
- `step(action)`: 执行动作
- `get_agent_state()`: 获取智能体状态（位置：经纬度+高度，旋转：roll角度）
- `set_agent_state(position, rotation)`: 设置智能体状态
  - `position`: [longitude, latitude, altitude]
  - `rotation`: roll角度（0-360度，0表示正北）
- `get_observations()`: 获取传感器观测（RGB从卫星地图crop）
- `geodesic_distance(pos_a, pos_b)`: 计算测地距离（地球表面的最短路径，考虑地球曲率）
- `is_navigable(position)`: 检查位置是否可导航
- `sample_navigable_point()`: 采样可导航点

## 开发状态

### ✅ 已完成
- 架构设计
- 文档编写
- Phase 1: 项目基础结构
- Phase 2.1: Episode数据定义（`core/episode.py`）

### ⏳ 进行中
- Phase 2: 核心数据结构实现
- Phase 3-10: 其他组件实现

### 📋 待实现
- satsim 集成
- 完整测试

## 与 VLN-CE 和 Habitat-Lab 的对比

| 特性 | Habitat-Lab | VLN-CE | SatNav |
|------|-------------|--------|--------|
| **定位** | 通用框架 | VLN专用实现 | VLN测试平台 |
| **复杂度** | 高 | 中 | 低 |
| **Registry** | ✅ | ✅ | ❌ |
| **依赖** | Habitat-Sim | Habitat-Sim | satsim（自定义） |
| **任务** | 多种 | VLN | VLN |
| **基线模型** | 可选 | ✅ | ❌ |

## 详细文档

详细的设计文档位于 `doc/refactor/` 目录：

- [ARCHITECTURE.md](doc/refactor/ARCHITECTURE.md) - 详细的架构设计
- [CAMERA_MODEL.md](doc/refactor/CAMERA_MODEL.md) - 相机模型说明（HFOV、视野计算、与地图API对接）
- [CODE_STRUCTURE.md](doc/refactor/CODE_STRUCTURE.md) - 代码结构说明
- [ARCHITECTURE_DIAGRAM.md](doc/refactor/ARCHITECTURE_DIAGRAM.md) - 架构图
- [DESIGN_EVALUATION.md](doc/refactor/DESIGN_EVALUATION.md) - 设计评估
- [DESIGN_SUMMARY.md](doc/refactor/DESIGN_SUMMARY.md) - 设计总结

## 许可证

MIT License


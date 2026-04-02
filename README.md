# SatNav: 连续状态 VLN 评测平台

## 项目简介

SatNav 是一个用于评测连续状态下视觉语言导航（VLN）的测试平台。该平台使用卫星地图作为场景，支持基于地理坐标（经度、纬度、高度）的连续空间导航，并提供**双仿真器架构**：

- **SatSim**：2D 卫星地图仿真器（高速渲染，依赖本地 GeoTIFF 文件）
- **AerialSim**：3D 航拍仿真器（使用 Google 3D Tiles，真实感渲染）

**核心特点**：
- 🌍 使用卫星地图作为场景（而非3D室内场景）
- 📍 位置使用地理坐标（经度、纬度、高度）
- 🧭 旋转使用航向角（heading，0度表示正北）
- 🖼️ 支持 2D 卫星图像和 3D 航拍渲染两种模式
- 🔄 仿真器可通过配置无缝切换，代码无需修改
- 🚀 最小化实现，不依赖 Habitat 组件

---

## 1. 环境安装

### 1.1 使用 Conda（推荐）

```bash
# 创建conda环境（Python 3.8+）
conda create -n satnav python=3.8
conda activate satnav

# 安装依赖
cd /path/to/SatNav
pip install -r requirements.txt

# 或者以开发模式安装
pip install -e .
```

### 1.2 使用 pip + venv

```bash
# 创建虚拟环境
python3 -m venv satnav_env
source satnav_env/bin/activate  # Linux/Mac
# 或
satnav_env\Scripts\activate  # Windows

# 安装依赖
cd /path/to/SatNav
pip install -r requirements.txt

# 或者以开发模式安装
pip install -e .
```

---

## 2. SatSim 仿真器工作原理

SatSim 是 SatNav 的核心仿真引擎，负责加载卫星地图、执行动作、生成观测。它采用**双坐标系统**设计，在内部使用 EPSG:3857（Web Mercator）进行高效计算，对外暴露 WGS84（经纬度）接口。

### 2.1 坐标系统

SatSim 使用两套坐标系统：

**外部接口（WGS84 - EPSG:4326）**：
- 输入/输出位置：`[longitude, latitude, altitude]`
- 经度/纬度：度数（-180 到 180，-90 到 90）
- 高度：米

**内部处理（Web Mercator - EPSG:3857）**：
- 位置：`[x, y]` 米为单位
- 优势：在局部区域内可以近似为平面，便于计算距离和移动

**坐标转换**：
- 使用 `pyproj` 库进行精确的坐标转换
- 所有转换在 `GeoUtils` 类中统一管理
- 转换发生在 SatSim 的边界层，对上层透明

### 2.2 场景加载

**场景文件格式**：
- 支持 GeoTIFF（.tif）格式的卫星地图
- 自动检测坐标系，如果不是 EPSG:3857，使用 `rasterio.WarpedVRT` 进行重投影
- 场景路径：`{SCENES_DIR}/{scene_id}.tif`

**加载策略**：
- **延迟加载**：场景在首次访问时加载（`reset()` 时）
- **LRU缓存**：使用 `@lru_cache` 缓存最近使用的5个场景
- **自动重投影**：非 EPSG:3857 的 TIF 文件自动重投影到 EPSG:3857

### 2.3 动作执行

SatSim 支持四种离散动作：

1. **MOVE_FORWARD**：向前移动 `FORWARD_STEP_SIZE` 米（默认 0.25m）
   - 在 Web Mercator 坐标系中计算新位置
   - 使用 `GeoUtils.move_in_mercator()` 根据当前航向角移动
   - 边界检查：如果新位置超出地图范围，则保持当前位置不变

2. **TURN_LEFT**：向左旋转 `TURN_ANGLE` 度（默认 15°）
   - 航向角减少，范围 [0, 360)

3. **TURN_RIGHT**：向右旋转 `TURN_ANGLE` 度（默认 15°）
   - 航向角增加，范围 [0, 360)

4. **STOP**：停止动作，不改变位置或旋转

### 2.4 图像渲染

SatSim 使用 `SatelliteCamera` 类生成 RGB 观测：

**渲染流程**：
1. **计算视野范围**：根据位置、高度、HFOV 和旋转角度计算地面覆盖范围
   - 水平半宽：`half_x_m = altitude * tan(HFOV / 2)`（HFOV控制水平方向）
   - 垂直半宽：`half_y_m = half_x_m / aspect_ratio`
   - 考虑旋转角度，计算四个角点的位置

2. **裁剪卫星地图**：使用 `rasterio.windows.from_bounds()` 从 TIF 文件中裁剪对应区域

3. **旋转图像**：使用 `scipy.ndimage.rotate()` 根据航向角旋转图像

4. **缩放图像**：使用 `cv2.resize()` 缩放到目标尺寸（默认 224x224）

5. **返回 RGB 数组**：形状为 `(H, W, 3)` 的 uint8 数组

**边界检查**：
- 如果相机视野超出地图边界，会抛出 `ValueError`
- 确保所有渲染的图像都完全在地图范围内

---

## 3. SatNav 架构

SatNav 采用分层架构，通过工厂模式支持多种仿真器后端：

```
Env → VLNTask → Simulator (抽象接口)
                    ↓
            create_simulator() 工厂函数
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   SatSimWrapper          AerialSimWrapper
        ↓                       ↓
     SatSim                 AerialSim
   (2D 卫星图)            (3D Google Tiles)
```

- **Env（环境层）**：管理 Episode 迭代，提供 `reset()` 和 `step(action)` 接口
- **VLNTask（任务层）**：管理传感器（RGB、Instruction）和评价指标（Success、SPL、DistanceToGoal等）
- **Simulator（抽象接口）**：定义仿真器统一接口（`reset`、`step`、`get_agent_state` 等）
- **SatSimWrapper / AerialSimWrapper（适配器层）**：实现 Simulator 接口
- **SatSim / AerialSim（仿真器层）**：场景加载、动作执行、图像渲染

### 3.1 仿真器切换

通过配置文件中的 `SIMULATOR.TYPE` 字段即可切换仿真器，**无需修改任何代码**：

```yaml
SIMULATOR:
  TYPE: satsim    # 使用 2D 卫星图仿真器
  # TYPE: aerialsim  # 使用 3D 航拍仿真器
```

| 特性 | SatSim | AerialSim |
|------|--------|-----------|
| 渲染速度 | ~1ms/帧 | ~5-6s/帧 |
| 图像类型 | 2D 卫星俯视图 | 3D 航拍渲染 |
| 数据依赖 | 本地 GeoTIFF 文件 | 网络 + Google API Key |
| 覆盖范围 | 已下载的区域 | 全球（有 3D 数据的区域） |

---

## 4. 配置和数据集

### 4.1 配置文件格式

SatNav 使用 YAML 格式的配置文件，示例：`configs/vln_task.yaml`

```yaml
ENVIRONMENT:
  # 每个 episode 的最大步数
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  # 仿真器类型: "satsim" (2D卫星图) 或 "aerialsim" (3D航拍)
  TYPE: satsim
  # 向前移动的步长（米）
  FORWARD_STEP_SIZE: 10
  # 转向角度（度）
  TURN_ANGLE: 15
  # RGB 传感器配置
  RGB_SENSOR:
    WIDTH: 448      # 图像宽度（像素）
    HEIGHT: 448     # 图像高度（像素）
    HFOV: 90        # 水平视野角度（度）
  # AerialSim 专用配置（仅当 TYPE: aerialsim 时使用）
  AERIAL:
    API_KEY: "YOUR_GOOGLE_API_KEY"  # Google 3D Tiles API key
    BROWSER: chrome                  # 浏览器: chrome/edge/firefox
    HEADLESS: true                   # 无头模式（服务器环境推荐）

TASK:
  TYPE: VLN
  # 成功距离阈值（米）
  SUCCESS_DISTANCE: 3.0
  # 允许的动作列表
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  # 启用的评价指标
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, SPL, PATH_LENGTH]

DATASET:
  TYPE: SatNav      # 仅用于标识，不影响实际加载
  SPLIT: train      # 数据集划分：train/val_seen/val_unseen/test
  # 数据集文件路径（支持 {split} 占位符）
  DATA_PATH: data/datasets/satnav/{split}/{split}.json.gz
  # 场景数据目录（TIF 文件所在目录）
  SCENES_DIR: data/scene_datasets/
```

### 4.2 数据集格式

SatNav 使用 JSON 格式的数据集文件（支持 `.json` 和 `.json.gz`），格式如下：

```json
{
  "instruction_vocab": {
    "word_list": ["go", "to", "the", "destination", ...]
  },
  "episodes": [
    {
      "episode_id": "example_001",
      "trajectory_id": "traj_001",
      "scene_id": "map",
      "start_position": [114.064413, 22.543496, 100.0],
      "start_rotation": 0.0,
      "goals": [
        {
          "position": [114.070413, 22.543496, 100.0]
        }
      ],
      "reference_path": [
        [114.064413, 22.543496, 100.0],
        [114.067413, 22.543496, 100.0],
        [114.070413, 22.543496, 100.0]
      ],
      "instruction": {
        "instruction_text": "Go east to the destination"
      }
    }
  ]
}
```

**字段说明**：
- `episode_id`: Episode 唯一标识符
- `scene_id`: 场景标识符，对应 `{SCENES_DIR}/{scene_id}.tif` 文件
- `start_position`: 起始位置 `[longitude, latitude, altitude]`
- `start_rotation`: 起始航向角（度，0=正北）
- `goals`: 目标位置列表（每个目标包含 `position`）
- `reference_path`: 参考路径（用于计算 SPL 等指标）
- `instruction`: 导航指令（包含 `instruction_text`）

### 4.3 场景文件

**场景文件要求**：
- 格式：GeoTIFF（.tif）
- 坐标系：建议使用 EPSG:3857，其他坐标系会自动重投影
- 位置：`{SCENES_DIR}/{scene_id}.tif`

**示例目录结构**：
```
data/
├── datasets/
│   └── satnav/
│       ├── train/
│       │   └── train.json.gz
│       └── val_seen/
│           └── val_seen.json.gz
└── scene_datasets/
    ├── map.tif
    └── scene_001.tif
```

---

## 5. 测试命令行

### 5.1 运行所有测试

```bash
# 确保在 conda 环境中
conda activate satnav

# 运行所有测试
pytest

# 运行所有测试（详细输出）
pytest -v

# 运行所有测试（显示覆盖率）
pytest --cov=satnav --cov-report=term-missing

# 运行所有测试（生成 HTML 覆盖率报告）
pytest --cov=satnav --cov-report=html
```

### 5.2 运行特定测试文件

```bash
# 配置加载测试
pytest tests/test_config.py -v

# 数据集加载测试
pytest tests/test_dataset.py -v

# 环境测试（包含集成测试）
pytest tests/test_env.py -v

# VLN 任务测试
pytest tests/test_vln_task.py -v

# SatSim 核心引擎测试
pytest tests/test_satsim.py -v

# 相机渲染测试
pytest tests/test_camera.py -v

# 坐标转换工具测试
pytest tests/test_geoutils.py -v

# 工具函数测试（测地距离）
pytest tests/test_utils.py -v
```

---

## 6. 快速开始示例

SatNav 提供了两个完整的示例脚本，演示如何使用不同的路径跟随策略进行导航。

### 6.1 SatNavPathFollower 示例

`satnav_path_follower_example.py` 演示如何使用 `SatNavPathFollower` 运行数据集中的所有 episode。

**特点**：
- 使用贪心策略直接导航到目标位置
- 支持运行数据集中的所有 episode
- 为每个 episode 生成视频（可选）
- 生成包含所有 episode 评估指标的 JSON 文件

**使用方法**：
```bash
# 运行所有 episode 并生成视频（默认）
python examples/satnav_path_follower_example.py

# 运行所有 episode 但不生成视频
python examples/satnav_path_follower_example.py --no-video
```

**输出**：
- 每个 episode 的视频：`output/episode_{episode_id}_video.mp4`
- 评估结果 JSON：`output/{dataset_name}_{timestamp}.json`
- 最终可视化图像：`output/topdown_map_satnav_follower.png`

**JSON 结果文件格式**：
```json
{
  "dataset_name": "satnav_dataset_complex",
  "timestamp": "20251202_151823",
  "total_episodes": 2,
  "summary": {
    "successful_episodes": 2,
    "success_rate": 1.0,
    "average_success": 1.0,
    "average_spl": 0.95,
    "average_steps": 45.5
  },
  "episodes": [
    {
      "episode_id": "complex_path_001",
      "scene_id": "map",
      "instruction": "Go forward, then turn left...",
      "success": 1.0,
      "spl": 0.95,
      "distance_to_goal": 0.5,
      "path_length": 120.5,
      "reference_path_length": 127.0,
      "path_efficiency": 0.95,
      "num_steps": 42,
      "action_distribution": {"MOVE_FORWARD": 30, "TURN_LEFT": 8, "TURN_RIGHT": 4}
    }
  ]
}
```

### 6.2 ReferencePathFollower 示例

`reference_follower_example.py` 演示如何使用 `ReferencePathFollower` 运行单个 episode。

**特点**：
- 沿着参考路径的 waypoint 顺序导航
- 严格遵循数据集中的参考路径
- 适用于生成 teacher forcing 数据和评估路径跟随精度
- 简洁的实现，专注于核心导航逻辑

**使用方法**：
```bash
python examples/reference_follower_example.py
```

**输出**：
- 最终可视化图像：`output/topdown_map_example.png`
- 控制台输出评估指标

---

## 7. Training and Evaluation（训练和评估）

### 7.1 Quick Start

SatNav 使用统一的入口点 `run.py` 进行训练和评估：

```bash
# Single-GPU training
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type train

# Multi-GPU training (4 cards on one node)
torchrun --nproc_per_node=4 run.py \
    --exp-config configs/baselines/seq2seq_offline.yaml \
    --run-type train \
    DISTRIBUTED.enabled true

# Evaluation
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type eval
```

### 7.2 Configuration System

采用 VLN-CE 风格的统一配置系统：
- **单一配置文件**：包含训练（IL）、评估（EVAL）、推理（INFERENCE）配置
- **灵活覆盖**：支持命令行参数覆盖任何配置项
- **自动同步**：评估时自动将 `EVAL.SPLIT` 同步到 `DATASET.SPLIT`
- **可选多卡训练**：默认单卡；设置 `DISTRIBUTED.enabled=true` 并用 `torchrun` 启动时启用 DDP 多卡训练

配置文件结构：
- `configs/default.yaml` - 默认配置模板（参考文档）
- `configs/baselines/seq2seq_offline.yaml` - Seq2Seq 离线训练配置
- `configs/debug_vln_task.yaml` - 调试任务配置

详细文档: [配置系统文档](doc/CONFIG_SYSTEM.md)

### 7.3 Evaluation Features

完整的评估功能包括：
- ✅ 环境中运行完整的 episode rollouts
- ✅ 计算所有 metrics（SPL, Success, DistanceToGoal, PathLength）
- ✅ 进度条显示和实时指标更新
- ✅ 保存 aggregated metrics 到 JSON
- ✅ 可选的视频生成（需启用 TOP_DOWN_MAP）

---

## 8. Baseline Models（基线模型）

SatNav 提供了 VLN 基线模型实现，用于训练和评估导航智能体。

### 7.1 Seq2Seq Model

**架构**：Sequence-to-Sequence baseline model
- **Instruction Encoder**: LSTM with GloVe 50d embeddings (128d hidden)
- **Visual Encoder**: ResNet-50 pretrained on ImageNet (256d output)
- **State Encoder**: GRU recurrent network (512d hidden)
- **Action Space**: 4 discrete actions (STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT)

**Quick Start**:
```bash
# Install dependencies
pip install torch torchvision

# Download GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove/

# Build vocabulary and embeddings
python -m satnav.utils.build_vocab \
    --dataset tests/test_data/satnav_dataset_complex.json \
    --output data/vocab/vocab.json

python -m satnav.utils.build_glove_embeddings \
    --vocab data/vocab/vocab.json \
    --glove data/glove/glove.6B.50d.txt \
    --output data/embeddings/glove_embeddings.json.gz
```

**Training**:
```bash
# Train with default config (single GPU)
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type train

# Train on GPU 3 only
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type train \
    TORCH_GPU_ID 3

# Train with 4 GPUs via DDP (recommended: use the launch script)
bash scripts/seq2seq/train_offline_ddp.sh

# Or launch manually with torchrun
torchrun --nproc_per_node=4 run.py \
    --exp-config configs/baselines/seq2seq_offline.yaml \
    --run-type train \
    DISTRIBUTED.enabled true

# Train with custom parameters
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type train \
    IL.epochs 20 IL.batch_size 8 IL.lr 1e-4

# Continue from checkpoint
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type train \
    IL.load_from_ckpt true IL.ckpt_to_load output/seq2seq_offline/checkpoints/latest/best.pth
```

多卡训练说明：
- 使用 `torchrun --nproc_per_node=<GPU数>` 启动，不再手动设置每个 rank 的 `TORCH_GPU_ID`
- `RecollectionDataset` 会按 rank 自动切分 episode，避免多卡重复训练同一批轨迹
- checkpoint 与单卡格式保持兼容，单卡和多卡可以相互续训

**Evaluation**:
```bash
# Evaluate on val_seen (uses checkpoints/latest symlink)
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type eval

# Evaluate on val_unseen
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type eval \
    EVAL.SPLIT val_unseen

# Evaluate a specific checkpoint
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type eval \
    EVAL.CKPT_PATH output/seq2seq_offline/checkpoints/latest/best.pth

# Quick evaluation (first 5 episodes)
python run.py --exp-config configs/baselines/seq2seq_offline.yaml --run-type eval \
    EVAL.EPISODE_COUNT 5
```

**Evaluation Output**:
- Metrics JSON: `data/results/seq2seq/eval_ckpt_0_{split}.json`
- Videos (if enabled): `data/videos/seq2seq/episode_{id}_ckpt_0.mp4`
- Console: Progress bar, aggregated metrics, timing info

**Programmatic Usage**:
```python
from satnav.models import ModelRegistry
from omegaconf import OmegaConf

# Load model
config = OmegaConf.load("configs/baselines/seq2seq_offline.yaml")
model_class = ModelRegistry.get_model("seq2seq")
model = model_class.from_config(config, obs_space, act_space)

# Forward pass
action, rnn_states = model.act(observations, rnn_states, prev_actions, masks)
```

**Documentation**:
- 📖 **Architecture Design**: `doc/models/SEQ2SEQ_IMPLEMENTATION.md`
- 🚀 **Quick Start Guide**: `doc/models/QUICKSTART.md`
- 🎓 **Training Guide**: `doc/training/TRAINING_GUIDE.md`
- ⚙️ **Config System**: `doc/CONFIG_SYSTEM.md`
- 📝 **Embedding Guide**: `doc/EMBEDDING_GUIDE.md`
- ✅ **Implementation Summary**: `doc/models/IMPLEMENTATION_SUMMARY.md`

**Tests**:
```bash
pytest tests/test_seq2seq_model.py -v
```

**Key Features**:
- ✅ VLN-CE compatible architecture
- ✅ GloVe pretrained embeddings support
- ✅ Flexible configuration system
- ✅ Model registry for easy model management

**Differences from VLN-CE**:
- ❌ No depth encoder (SatNav uses satellite overhead imagery)
- ✅ Same instruction and RGB encoders
- ✅ Compatible configuration format

---

## 8. 应用工具

SatNav 提供了多个实用工具应用，位于 `applications/` 目录下：

### 8.1 SatSim Viewer（交互式查看器）

`applications/satsim_viewer/` 目录包含两个交互式查看器应用：

#### 8.1.1 Free Viewer（自由探索查看器）

用于自由探索卫星地图的工具，支持键盘控制导航。

**快速开始**：
```bash
# 使用 -m 模块方式（默认运行 free viewer）
python -m applications.satsim_viewer
python -m applications.satsim_viewer free

# 或直接运行
python -m applications.satsim_viewer.free_viewer
```

**控制方式**：
- `w`: 向前移动
- `s`: 向后移动
- `a`: 向左转
- `d`: 向右转
- `q`: 上升（增加高度）
- `e`: 下降（减少高度）
- `p`: 保存当前图像
- `ESC`: 退出

**配置**：编辑 `applications/satsim_viewer/config.yaml` 设置 TIF 文件路径、相机参数等。

#### 8.1.2 Task Viewer（任务查看器）

用于交互式浏览 VLN 任务的工具，支持键盘控制导航和任务评估。

**快速开始**：
```bash
# 使用 -m 模块方式
python -m applications.satsim_viewer task

# 或直接运行
python -m applications.satsim_viewer.task_viewer
```

**控制方式**：
- `w`: 向前移动
- `a`: 向左转
- `d`: 向右转
- `t`: 切换 topdown 视图
- `SPACE`: 停止并显示评估指标，然后加载下一个 episode
- `ESC`: 退出

**配置**：使用 VLN 任务配置文件（如 `configs/vln_task.yaml`）。

详细文档：`applications/satsim_viewer/README.md`

### 8.2 Map Downloader（地图下载器）

从 Google Maps Static API 下载卫星图像并生成 GeoTIFF 文件的工具。

**快速开始**：
```bash
# 编辑 config.yaml 设置区域和 API key
python -m applications.map_downloader
```

**主要功能**：
- 支持角点定义或中心点+尺寸两种区域定义方式
- 支持顺序和并行两种下载模式（并行模式 4-9 倍速度提升）
- 自动生成带地理参考的 GeoTIFF 文件（EPSG:3857）

**配置**：编辑 `applications/map_downloader/config.yaml` 设置：
- `REGION`: 区域定义（角点或中心点）
- `API.API_KEY`: Google Maps API key
- `DOWNLOAD.MODE`: 下载模式（`sequential` 或 `parallel`）
- `DOWNLOAD.ZOOM`: 缩放级别（15-20）

详细文档：`applications/map_downloader/README.md`

### 8.3 Aerial Viewer（3D 航拍查看器）

使用 AerialSim 核心渲染非交互式 3D 航拍视图的工具，与 SatSim 相机模型兼容。

**快速开始**：
```bash
python -m applications.aerial_viewer
```

**主要功能**：
- 渲染 3D 航拍视图（使用 Google 3D Tiles）
- 与 SatSim 相机参数兼容（垂直俯视、HFOV 匹配）
- 支持序列渲染（优化后约 2x 加速）
- 支持多种浏览器（Chrome、Edge、Firefox）
- 命令行接口和配置文件支持

**示例输出对比**：

| SatSim 卫星视图（2D 正交投影） | AerialSim 3D 视图（3D 透视投影） |
|:---:|:---:|
| ![SatSim](applications/aerial_viewer/images/sat_crop_view.png) | ![AerialSim](applications/aerial_viewer/images/aerial_view.png) |

**配置**：编辑 `applications/aerial_viewer/config.yaml` 设置：
- `API.API_KEY`: Google 3D Tiles API key
- `AGENT`: 智能体状态（经纬度、高度、旋转角度）
- `CAMERA`: 相机参数（HFOV、宽度、高度）

**与 SatSim 的区别**：
- **投影模型**：SatSim 使用正交投影（无透视变形），AerialSim 使用透视投影（3D 渲染）
- **地形**：SatSim 使用 2D 卫星图像（平面），AerialSim 使用 3D Tiles（包含建筑物高度和地形）
- **图像来源**：SatSim 使用高分辨率 GeoTIFF 文件，AerialSim 使用 Google 3D Tiles（流式传输）

**依赖要求**：
- `selenium` - WebDriver 控制
- `omegaconf` - 配置管理
- `webdriver-manager`（推荐）- 自动管理浏览器驱动
- Google 3D Tiles API key

详细文档：`applications/aerial_viewer/README.md`

### 8.4 Sat-Drone Pair Generation（配对数据生产）

用于从多个公开数据源构建 sat-drone image pair 数据集的应用，位于 `applications/sat_drone_pair_generation/`。

当前纳入的数据源：

- `denseuav`
- `gta_uav`
- `sues`
- `uavvisloc`

**设计约束**：
- 保留各数据源原有 build/export 脚本边界
- 不改原有生产逻辑
- 统一入口仅负责路由和仓内模块化运行

**快速开始**：
```bash
python -m applications.sat_drone_pair_generation denseuav build_pairs --help
python -m applications.sat_drone_pair_generation gta-uav build_pairs --help
python -m applications.sat_drone_pair_generation sues pipeline --help
python -m applications.sat_drone_pair_generation uavvisloc export_selected --help
```

**说明**：
- 推荐使用仓库现有 `satnav` conda 环境
- 当前主流程依赖可由 SatNav 现有环境直接满足
- `gta-uav` / `uav-visloc` 会自动映射到仓内模块名 `gta_uav` / `uavvisloc`
- 默认会读取 `applications/sat_drone_pair_generation/config.yaml`
- 可把常用数据路径写到配置中的 `input` / `output`，命令行显式传参会覆盖配置值

详细文档：`applications/sat_drone_pair_generation/README.md`

---

## 9. 项目结构

```
SatNav/
├── run.py                     # 统一训练/评估入口点 ⭐
├── satnav/                    # 主包
│   ├── core/                 # 核心组件
│   │   ├── env.py           # 环境类
│   │   ├── simulator.py     # 仿真器接口
│   │   ├── episode.py        # Episode数据定义
│   │   ├── config.py         # 配置系统
│   │   └── utils.py          # 工具函数（测地距离等）
│   ├── task/                 # VLN任务
│   │   ├── vln_task.py       # VLN任务类
│   │   ├── sensors.py        # 传感器
│   │   ├── actions.py        # 动作定义
│   │   └── measures.py       # 评价指标
│   ├── dataset/              # 数据集
│   │   ├── satnav_dataset.py     # SatNav数据集加载器
│   │   └── recollect_dataset.py  # 实时收集数据集
│   ├── models/               # 模型
│   │   ├── registry.py       # 模型注册表
│   │   ├── base.py           # 基类
│   │   ├── baselines/        # 基线模型
│   │   │   └── seq2seq_policy.py  # Seq2Seq 模型
│   │   └── encoders/         # 编码器
│   │       ├── instruction_encoder.py
│   │       ├── visual_encoder.py
│   │       └── rnn_state_encoder.py
│   ├── training/             # 训练模块 ⭐
│   │   ├── registry.py       # Trainer 注册表
│   │   ├── base_il_trainer.py  # 基础 IL trainer
│   │   ├── recollect_trainer.py  # 实时收集 trainer
│   │   └── utils.py          # 训练工具
│   ├── navigation/           # 导航策略
│   │   ├── path_follower.py  # 路径跟随器
│   │   └── discrete_planner.py  # 离散规划器
│   └── sims/                 # 仿真器
│       ├── __init__.py       # 仿真器工厂 (create_simulator)
│       ├── satsim_wrapper.py # SatSim 适配器
│       ├── aerialsim_wrapper.py # AerialSim 适配器 ⭐
│       ├── satsim/           # SatSim 核心模块 (2D)
│       │   ├── __init__.py
│       │   ├── satsim.py     # 核心引擎
│       │   ├── camera.py     # 相机渲染
│       │   └── geoutils.py   # 坐标工具
│       └── aerialsim/        # AerialSim 核心模块 (3D) ⭐
│           ├── __init__.py
│           ├── aerialsim.py  # 核心引擎
│           └── cesium_template.html  # CesiumJS 模板
├── configs/                   # 配置文件 ⭐
│   ├── default.yaml          # 默认配置模板
│   ├── debug_vln_task.yaml   # 调试任务配置
│   ├── vln_task.yaml         # 完整任务配置
│   └── baselines/            # 基线模型配置
│       ├── seq2seq_offline.yaml  # Seq2Seq 离线训练配置
│       └── cma.yaml              # CMA 模型配置
├── examples/                  # 示例代码
│   ├── satnav_path_follower_example.py  # SatNavPathFollower 示例（批量运行）
│   └── reference_follower_example.py    # ReferencePathFollower 示例（单 episode）
├── tests/                     # 测试文件
│   ├── test_data/            # 测试数据
│   │   ├── map.tif           # 测试用卫星地图
│   │   ├── satnav_dataset_example.json
│   │   └── satnav_config_example.yaml
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_env.py
│   ├── test_satsim.py
│   ├── test_camera.py
│   ├── test_seq2seq_model.py  # 模型测试
│   ├── test_training.py       # 训练测试
│   └── test_geoutils.py
├── applications/             # 应用工具
│   ├── trajectory_generation/       # 轨迹生成
│   ├── satsim_viewer/       # 交互式查看器
│   │   ├── free_viewer.py   # 自由探索查看器
│   │   └── task_viewer.py   # 任务查看器
│   ├── aerial_viewer/        # 3D 航拍查看器
│   ├── map_downloader/      # 地图下载器
│   └── sat_drone_pair_generation/  # sat-drone 配对数据生产
├── doc/                      # 文档目录 ⭐
│   ├── CONFIG_SYSTEM.md      # 配置系统文档
│   ├── EMBEDDING_GUIDE.md    # Embedding 指南
│   ├── models/               # 模型文档
│   │   ├── SEQ2SEQ_IMPLEMENTATION.md
│   │   ├── CMA_IMPLEMENTATION.md
│   │   ├── QUICKSTART.md
│   │   └── IMPLEMENTATION_SUMMARY.md
│   ├── training/             # 训练文档
│   │   ├── TRAINING_GUIDE.md
│   │   └── SEQ2SEQ_SHAPE.md
│   └── refactor/             # 设计文档
│       └── ARCHITECTURE.md
├── setup.py                  # 安装脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

**⭐ 标记的是新增或重要更新的部分**

---

## 10. 许可证

MIT License

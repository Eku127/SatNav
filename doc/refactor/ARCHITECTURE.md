# SatNav 架构设计文档

## 1. 项目概述

SatNav 是一个用于评测连续状态下视觉语言导航（VLN）的测试平台。该平台设计简洁，专注于最小功能实现，不依赖 Habitat 组件，使用自定义的 satsim 仿真器。

### 设计原则
- **最小化实现**：只实现核心功能，代码简洁
- **无 Registry**：简化设计，直接导入使用
- **单一任务**：专注于 VLN 任务
- **独立实现**：不依赖 Habitat 任何组件
- **连续空间导航**：使用地理坐标系统（经纬度+高度），支持大尺度室外导航

### 核心特点
- **位置表示**：使用地理坐标系统（WGS84），xyz分别表示经度、纬度、高度
- **旋转表示**：使用roll角度（航向角），相比quaternion更简洁，适合地面导航
- **图像生成**：从现有卫星地图crop图像，避免复杂的3D渲染
- **距离计算**：使用测地距离（geodesic distance），考虑地球曲率

## 2. 整体架构

```
satnav/
├── satnav/
│   ├── __init__.py
│   ├── core/                    # 核心组件
│   │   ├── __init__.py
│   │   ├── env.py              # 环境类（连接所有组件）
│   │   ├── simulator.py         # 仿真器接口（satsim包装）
│   │   ├── episode.py           # Episode数据定义
│   │   ├── config.py            # 配置系统
│   │   └── utils.py             # 工具函数（测地距离等）
│   ├── task/                    # VLN任务
│   │   ├── __init__.py
│   │   ├── vln_task.py         # VLN任务类
│   │   ├── sensors.py          # 传感器（Instruction, RGB）
│   │   ├── actions.py          # 动作定义
│   │   └── measures.py         # 评价指标（SPL, Success, DistanceToGoal, NDTW等）
│   ├── dataset/                 # 数据集
│   │   ├── __init__.py
│   │   └── satnav_dataset.py   # SatNav数据集加载器
│   ├── sims/                    # 仿真器包装
│   │   ├── __init__.py
│   │   └── satsim_wrapper.py   # satsim仿真器包装
│   ├── navigation/              # 导航算法（可选）
│   │   └── path_follower.py    # 路径跟随算法
│   └── utils/                   # 工具函数
│       ├── maps.py              # 地图相关工具
│       └── examples.py          # 示例工具
├── configs/                     # 配置文件
│   └── satnav_task.yaml           # VLN任务配置
├── examples/                    # 示例代码
│   └── satnav_path_follower_example.py
└── README.md                    # 项目说明
```

## 3. 核心组件设计

### 3.1 环境类 (core/env.py)

**职责**：连接仿真器、任务和数据集，提供统一的 reset() 和 step() 接口

**主要方法**：
- `__init__(config, dataset=None)`: 初始化环境
- `reset()`: 重置环境，返回初始观测
- `step(action)`: 执行动作，返回 (obs, reward, done, info)
- `get_metrics()`: 获取当前评价指标

**数据流**：
```
reset():
  dataset.episodes[i] → task.reset(episode) → sim.reset(scene) → obs

step(action):
  action → task.step(action) → sim.step(action) → obs → measures.update()
```

### 3.2 仿真器接口 (core/simulator.py)

**职责**：定义仿真器抽象接口，包装 satsim

**状态定义**：
```python
AgentState:
  position: np.ndarray[3]  # (longitude, latitude, altitude) 经度、纬度、高度
  rotation: float          # roll角度（航向角，0-360度，0表示正北）
```

**主要接口**：
- `reset(scene_id)`: 重置仿真器，加载场景
- `step(action)`: 执行动作
- `get_agent_state()`: 获取智能体状态（位置、旋转）
- `set_agent_state(position, rotation)`: 设置智能体状态
- `get_observations()`: 获取传感器观测（RGB）
- `geodesic_distance(pos_a, pos_b)`: 计算测地距离（使用Haversine/Vincenty公式）

**注意**：
- 位置使用地理坐标系统（WGS84），xyz分别表示经度、纬度、高度
- 旋转使用roll角度（航向角），相比quaternion更简洁，适合地面导航
- 图像通过从卫星地图crop生成，而不是3D渲染

### 3.3 Episode 数据 (core/episode.py)

**职责**：定义 VLN Episode 数据结构

**数据结构**：
```python
VLNEpisode:
  episode_id: str
  scene_id: str
  start_position: List[float]  # [longitude, latitude, altitude] 经度、纬度、高度
  start_rotation: float        # roll角度（航向角，0-360度，0表示正北）
  goals: List[NavigationGoal]  # 目标位置列表
  reference_path: List[List[float]]  # 参考路径 [[lon,lat,alt], ...]
  instruction: InstructionData
    - instruction_text: str
  trajectory_id: str
```

**关键特点**：
- 使用连续坐标路径，而非离散导航图节点
- 所有位置使用地理坐标（经纬度+高度）
- 支持多目标点导航

### 3.4 VLN 任务 (task/vln_task.py)

**职责**：定义 VLN 任务逻辑，管理传感器和评价指标

**主要组件**：

- **传感器 (Sensors)**:
  - `InstructionSensor`: 提供指令文本
  - `RGBSensor`: RGB图像（从卫星地图crop获取）

- **动作 (Actions)**:
  - `STOP`: 停止
  - `MOVE_FORWARD`: 前进（step_size米）
  - `TURN_LEFT`: 左转（turn_angle度）
  - `TURN_RIGHT`: 右转（turn_angle度）

- **评价指标 (Measures)**:
  - `Success`: 是否到达目标（距离 < success_distance）
  - `SPL`: Success weighted by Path Length
  - `DistanceToGoal`: 到目标的测地距离
  - `PathLength`: 已走路径长度（使用测地距离累加）
  - `NDTW`: Normalized Dynamic Time Warping（路径相似度）

**主要方法**：
- `reset(episode)`: 重置任务，设置目标
- `step(action)`: 执行动作，更新指标
- `get_observations()`: 获取所有传感器观测
- `get_metrics()`: 获取所有评价指标

### 3.5 数据集 (dataset/satnav_dataset.py)

**职责**：加载 SatNav 格式的 VLN 数据集

**数据格式**：
- YAML 或 JSON 文件
- 包含 episodes 列表
- 每个 episode 包含 scene_id, start_position, goals, instruction, reference_path 等

**主要方法**：
- `__init__(config)`: 加载数据集
- `get_episode(episode_id)`: 获取指定episode

### 3.6 配置系统 (core/config.py)

**职责**：加载和管理 YAML 配置文件

**配置格式**：
```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  FORWARD_STEP_SIZE: 10        # 前进步长（米）
  TURN_ANGLE: 15              # 转向角度（度）
  RGB_SENSOR:
    WIDTH: 640                 # 图像宽度（像素）
    HEIGHT: 480                # 图像高度（像素）
    HFOV: 90                   # 水平视场角（度）

TASK:
  TYPE: VLN
  SUCCESS_DISTANCE: 10.0       # 成功距离阈值（米）
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, SPL, PATH_LENGTH, NDTW]

DATASET:
  TYPE: SatNav
  SPLIT: train
  DATA_PATH: data/datasets/satnav/{split}.yaml
  SCENES_DIR: data/scene_datasets/
```

## 4. 相机模型

### 4.1 设计目标

SatNav 的相机模型旨在**模拟无人机在特定高度下的俯视相机拍摄**。系统根据智能体的位置（经纬度、高度）和相机参数（HFOV），从卫星地图数据中裁剪出对应区域的图像。

### 4.2 相机模型类型

SatNav 使用一个**简化的、垂直俯瞰的针孔相机模型**：
- **俯视角度**：相机（无人机）垂直向下拍摄
- **透视效果**：由于是俯视且高度相对固定，透视变形较小
- **参数化**：通过高度和视场角来控制视野范围

### 4.3 视野范围计算

**核心公式**：
```
Ground_Width = 2 × Altitude × tan(HFOV / 2)
GSD = Ground_Width / WIDTH
```

其中：
- `Ground_Width`: 在地面上覆盖的视野宽度（米）
- `Altitude`: 无人机高度（米）
- `HFOV`: 水平视场角（度）
- `GSD`: 地面采样距离（米/像素）

**图像生成流程**：
1. 获取智能体当前的高度(altitude)和相机的HFOV配置
2. 计算在当前高度下，相机在地面上覆盖的视野宽度（米）
3. 根据视野宽度和图像像素宽度，计算出地面采样距离(GSD, 米/像素)
4. 使用智能体当前的经纬度作为中心点，结合计算出的地面范围和roll角度（航向），从卫星地图数据源中裁剪出相应区域的图像

## 5. 数据流设计

### 5.1 初始化流程

```
1. 加载配置 (config.yaml)
   ↓
2. 创建数据集 (SatNavDataset)
   - 加载 YAML/JSON 文件
   - 解析 episodes
   ↓
3. 创建仿真器 (SatSimWrapper)
   - 初始化 satsim
   ↓
4. 创建任务 (VLNTask)
   - 初始化传感器
   - 初始化评价指标
   ↓
5. 创建环境 (Env)
   - 连接 dataset, simulator, task
```

### 5.2 Reset 流程

```
env.reset()
  ↓
1. 从 dataset 获取下一个 episode
  ↓
2. task.reset(episode)
   - 设置目标位置
   - 重置所有 measures
   ↓
3. sim.reset(episode.scene_id)
   - 加载卫星地图场景
   - 设置智能体初始位置（经纬度+高度）和roll角度
   ↓
4. 获取初始观测
   - sim.get_observations() → RGB（从卫星地图crop）
   - task.get_observations() → Instruction
   ↓
5. 返回组合观测
   obs = {
     "rgb": rgb_image,        # 从卫星地图crop的图像
     "instruction": {
       "text": instruction_text
     }
   }
```

### 5.3 Step 流程

```
env.step(action)
  ↓
1. task.step(action)
   - 验证动作有效性
   ↓
2. sim.step(action)
   - 执行动作
   - 更新智能体状态
   ↓
3. 获取新观测
   - sim.get_observations() → RGB
   ↓
4. 更新评价指标
   - measures.update()
   - DistanceToGoal.update()
   - Success.update()
   - SPL.update()
   - PathLength.update()
   - NDTW.update()
   ↓
5. 检查是否结束
   - done = (Success == True) or (steps >= max_steps)
   ↓
6. 返回 (obs, reward, done, info)
   info = {
     "metrics": task.get_metrics(),
     "episode_id": episode.episode_id
   }
```

## 6. 评价指标设计

### 6.1 Success
- **定义**：智能体是否到达目标
- **计算**：`distance_to_goal < success_distance`
- **更新时机**：每次 step 后

### 6.2 DistanceToGoal
- **定义**：当前智能体位置到目标的测地距离（地球表面的最短路径）
- **计算**：`sim.geodesic_distance(agent_pos, goal_pos)`（使用Haversine/Vincenty公式）
- **注意**：需要考虑地球曲率，不是简单的欧氏距离
- **更新时机**：每次 step 后

### 6.3 PathLength
- **定义**：智能体已走的路径长度
- **计算**：累加每步的测地距离（使用geodesic_distance，考虑地球曲率）
- **注意**：在卫星地图中应使用测地距离而非欧氏距离
- **更新时机**：每次 step 后

### 6.4 SPL (Success weighted by Path Length)
- **定义**：考虑成功率和路径效率的综合指标
- **公式**：`SPL = Success * (reference_path_length / max(reference_path_length, actual_path_length))`
- **更新时机**：episode 结束时

### 6.5 NDTW (Normalized Dynamic Time Warping)
- **定义**：归一化的动态时间规整距离，衡量路径相似度
- **公式**：`NDTW = exp(-dtw_distance / normalization)`
- **归一化**：对于大尺度室外场景，使用参考路径的实际长度（geodesic distance累加）作为归一化基准
- **特点**：值越大表示路径匹配度越高，范围 [0, 1]
- **更新时机**：每次 step 后

## 7. 距离计算

### 7.1 测地距离 (Geodesic Distance)

SatNav 使用**测地距离**（geodesic distance）而非欧氏距离，因为：
- 地球是球面，不是平面
- 需要考虑地球曲率
- 使用 Haversine 或 Vincenty 公式计算

**实现**：
```python
def geodesic_distance(pos_a, pos_b) -> float:
    """
    计算两点间的测地距离（地球表面的最短路径距离）
    
    Args:
        pos_a: [longitude, latitude, altitude] 起点位置
        pos_b: [longitude, latitude, altitude] 终点位置
    
    Returns:
        float: 测地距离（米），使用Haversine或Vincenty公式计算
    """
```

### 7.2 使用场景

测地距离用于：
- `DistanceToGoal`: 计算到目标的距离
- `PathLength`: 累加路径长度
- `SPL`: 计算参考路径长度和实际路径长度
- `NDTW`: DTW算法中的距离计算
- `MOVE_FORWARD`: 根据步长和方向计算新位置

## 8. 与室内导航的对比

| 特性 | 室内导航（Habitat） | 卫星地图导航（SatNav） |
|------|-------------------|---------------------|
| **场景** | 3D室内场景 | 卫星地图 |
| **位置** | x, y, z (米) | 经度, 纬度, 高度 |
| **旋转** | Quaternion (4D) | Roll角度 (1D) |
| **坐标系统** | 局部3D坐标 | 地理坐标（WGS84） |
| **距离** | 欧氏距离 | 测地距离（Haversine） |
| **图像** | 3D渲染 | 卫星地图裁剪 |
| **深度** | 有（3D场景） | 无 |
| **范围** | 单个建筑物 | 整个地球表面 |
| **步长** | 0.25米 | 几米到几十米 |

## 9. 设计决策说明

### 9.1 为什么不用 Registry？
- **简化设计**：减少抽象层，代码更直观
- **最小实现**：当前只需要 VLN 任务，不需要动态注册
- **易于理解**：直接导入使用，降低学习成本

### 9.2 为什么独立实现？
- **无依赖**：不依赖 Habitat，避免版本冲突
- **轻量级**：只实现必需功能，代码量小
- **可控性**：完全控制代码实现，便于定制

### 9.3 为什么使用地理坐标？
- **真实应用**：卫星地图导航是真实世界的应用场景
- **标准格式**：地理坐标系统（WGS84）是标准格式
- **大尺度支持**：支持大范围室外导航，不受局部坐标限制

### 9.4 为什么使用roll角度而非quaternion？
- **简洁性**：对于地面导航，只需要航向角（heading/yaw）
- **直观性**：角度值更直观，0度表示正北
- **效率**：计算效率更高，存储空间更小

## 10. 使用示例

```python
from satnav.core.env import Env
from satnav.core.config import load_config
from satnav.dataset.satnav_dataset import SatNavDataset

# 1. 加载配置
config = load_config("configs/satnav_task.yaml")

# 2. 创建数据集
dataset = SatNavDataset(config.DATASET)

# 3. 创建环境
env = Env(config, dataset)

# 4. 运行一个 episode
obs = env.reset()
done = False
while not done:
    # 选择动作（这里用随机动作示例）
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    # 打印观测
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"Instruction: {obs['instruction']['text']}")

# 5. 获取最终评价指标
metrics = env.get_metrics()
print(f"Success: {metrics['success']}")
print(f"SPL: {metrics['spl']}")
print(f"Distance to Goal: {metrics['distance_to_goal']}")
print(f"NDTW: {metrics['ndtw']}")
```

## 11. 总结

SatNav 是一个专注于 VLN 评测的轻量级平台，设计简洁，易于理解和扩展。核心特点：

1. ✅ **应用场景真实**：卫星地图导航是实际应用
2. ✅ **状态表示简洁**：经纬度+roll角度，符合地理信息系统标准
3. ✅ **实现方式高效**：从卫星地图crop，避免复杂渲染
4. ✅ **设计简洁**：相比quaternion，roll角度更简单
5. ✅ **大尺度支持**：支持大范围室外导航，使用测地距离

主要需要注意：
1. ⚠️ 坐标转换和距离计算的准确性（考虑地球曲率）
2. ⚠️ 图像裁剪参数的合理设置
3. ⚠️ NDTW归一化方式需要适应大尺度场景

该架构为最小功能实现，后续可根据需要逐步扩展。

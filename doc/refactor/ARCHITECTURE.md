# SatNav 架构设计文档

## 1. 项目概述

SatNav 是一个用于评测连续状态下视觉语言导航（VLN）的测试平台。该平台设计简洁，专注于最小功能实现，不依赖 Habitat 组件，使用自定义的 satsim 仿真器。

### 设计原则
- **最小化实现**：只实现核心功能，代码简洁
- **无 Registry**：简化设计，直接导入使用
- **单一任务**：专注于 VLN 任务
- **独立实现**：不依赖 Habitat 任何组件
- **接口预留**：satsim 仿真器接口完整但待实现

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
│   │   └── config.py            # 配置系统
│   ├── task/                    # VLN任务
│   │   ├── __init__.py
│   │   ├── vln_task.py         # VLN任务类
│   │   ├── sensors.py          # 传感器（Instruction, RGB）
│   │   ├── actions.py          # 动作定义
│   │   └── measures.py         # 评价指标（SPL, Success, DistanceToGoal等）
│   ├── dataset/                 # 数据集
│   │   ├── __init__.py
│   │   └── r2r_dataset.py      # R2R数据集加载器
│   └── sims/                    # 仿真器包装
│       ├── __init__.py
│       └── satsim_wrapper.py   # satsim仿真器包装（接口待实现）
├── configs/                     # 配置文件
│   └── vln_task.yaml           # VLN任务配置（类似vlnce格式）
├── examples/                    # 示例代码
│   └── example_vln.py          # 使用示例
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

**主要接口**（待 satsim 实现）：
- `reset(scene_id)`: 重置仿真器，加载场景
- `step(action)`: 执行动作
- `get_agent_state()`: 获取智能体状态（位置、旋转）
- `set_agent_state(position, rotation)`: 设置智能体状态
- `get_observations()`: 获取传感器观测（RGB）
- `geodesic_distance(pos_a, pos_b)`: 计算测地距离（用于SPL）
- `is_navigable(position)`: 检查位置是否可导航
- `sample_navigable_point()`: 采样可导航点

**状态定义**：
```python
AgentState:
  position: np.ndarray[3]  # (longitude, latitude, altitude) 经度、纬度、高度
  rotation: float          # roll角度（航向角，0-360度，0表示正北）
```

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
  - `DistanceToGoal`: 到目标的距离
  - `PathLength`: 已走路径长度
  - `NDTW`: Normalized Dynamic Time Warping（可选）

**主要方法**：
- `reset(episode)`: 重置任务，设置目标
- `step(action)`: 执行动作，更新指标
- `get_observations()`: 获取所有传感器观测
- `get_metrics()`: 获取所有评价指标

### 3.5 数据集 (dataset/r2r_dataset.py)

**职责**：加载 R2R 格式的 VLN 数据集

**数据格式**：
- JSON.gz 文件
- 包含 episodes 列表
- 每个 episode 包含 scene_id, start_position, goals, instruction 等

**主要方法**：
- `__init__(config)`: 加载数据集
- `from_json(json_str, scenes_dir)`: 从JSON解析episodes
- `get_episode(episode_id)`: 获取指定episode

### 3.6 配置系统 (core/config.py)

**职责**：加载和管理 YAML 配置文件

**配置格式**（类似 VLN-CE）：
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
  TYPE: SatNav  # Kept for compatibility, not used (no registry system)
  SPLIT: train
  DATA_PATH: data/datasets/R2R/{split}/{split}.json.gz
  SCENES_DIR: data/scene_datasets/
```

## 4. 数据流设计

### 4.1 初始化流程

```
1. 加载配置 (config.yaml)
   ↓
2. 创建数据集 (R2RDataset)
   - 加载 JSON.gz 文件
   - 解析 episodes
   ↓
3. 创建仿真器 (SatSimWrapper)
   - 初始化 satsim（待实现）
   ↓
4. 创建任务 (VLNTask)
   - 初始化传感器
   - 初始化评价指标
   ↓
5. 创建环境 (Env)
   - 连接 dataset, simulator, task
```

### 4.2 Reset 流程

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
   - 加载卫星地图场景（待实现）
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

### 4.3 Step 流程

```
env.step(action)
  ↓
1. task.step(action)
   - 验证动作有效性
   ↓
2. sim.step(action)
   - 执行动作（待实现）
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

## 5. 评价指标设计

### 5.1 Success
- **定义**：智能体是否到达目标
- **计算**：`distance_to_goal < success_distance`
- **更新时机**：每次 step 后

### 5.2 DistanceToGoal
- **定义**：当前智能体位置到目标的测地距离（地球表面的最短路径）
- **计算**：`sim.geodesic_distance(agent_pos, goal_pos)`（使用Haversine/Vincenty公式）
- **注意**：需要考虑地球曲率，不是简单的欧氏距离
- **更新时机**：每次 step 后

### 5.3 PathLength
- **定义**：智能体已走的路径长度
- **计算**：累加每步的测地距离（使用geodesic_distance，考虑地球曲率）
- **注意**：在卫星地图中应使用测地距离而非欧氏距离
- **更新时机**：每次 step 后

### 5.4 SPL (Success weighted by Path Length)
- **定义**：考虑成功率和路径效率的综合指标
- **公式**：`SPL = Success * (reference_path_length / max(reference_path_length, actual_path_length))`
- **更新时机**：episode 结束时

### 5.5 NDTW (可选)
- **定义**：归一化的动态时间规整距离
- **计算**：使用 DTW 算法比较预测路径和参考路径
- **更新时机**：episode 结束时

## 6. 仿真器接口设计（待实现）

### 6.1 SatSimWrapper 类

```python
class SatSimWrapper:
    """satsim 仿真器包装类"""
    
    def __init__(self, config):
        # TODO: 初始化 satsim 引擎
        pass
    
    def reset(self, scene_id: str):
        """重置仿真器，加载场景"""
        # TODO: 调用 satsim 加载场景
        # TODO: 设置智能体初始状态
        raise NotImplementedError("需要实现 satsim.reset()")
    
    def step(self, action: str):
        """执行动作"""
        # TODO: 调用 satsim 执行动作
        # TODO: 返回是否成功执行
        raise NotImplementedError("需要实现 satsim.step()")
    
    def get_agent_state(self) -> AgentState:
        """获取智能体状态
        
        Returns:
            AgentState: 包含位置（经度、纬度、高度）和roll角度（航向角）
        """
        # TODO: 从 satsim 获取位置（经纬度+高度）和roll角度
        raise NotImplementedError("需要实现 satsim.get_agent_state()")
    
    def set_agent_state(self, position, rotation):
        """设置智能体状态
        
        Args:
            position: [longitude, latitude, altitude] 经度、纬度、高度
            rotation: roll角度（航向角，0-360度，0表示正北）
        """
        # TODO: 调用 satsim 设置状态
        raise NotImplementedError("需要实现 satsim.set_agent_state()")
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取传感器观测
        
        此方法模拟无人机俯瞰拍摄。它会执行以下步骤：
        1. 获取智能体当前的高度(altitude)和相机的HFOV配置
        2. 计算在当前高度下，相机在地面上覆盖的视野宽度（米）
        3. 根据视野宽度和图像像素宽度，计算出地面采样距离(GSD, 米/像素)
        4. 使用智能体当前的经纬度作为中心点，结合计算出的地面范围和roll角度（航向），
           从卫星地图数据源中裁剪出相应区域的图像
        
        计算公式：
        - Ground_Width = 2 * Altitude * tan(HFOV / 2)
        - GSD = Ground_Width / WIDTH
        """
        # TODO: 从 satsim 获取 RGB（从卫星地图crop）
        return {
            "rgb": rgb_image      # (H, W, 3) uint8，从卫星地图crop的图像
        }
        raise NotImplementedError("需要实现 satsim.get_observations()")
    
    def geodesic_distance(self, pos_a, pos_b) -> float:
        """计算两点间的测地距离（地球表面的最短路径距离）
        
        Args:
            pos_a: [longitude, latitude, altitude] 起点位置
            pos_b: [longitude, latitude, altitude] 终点位置
        
        Returns:
            float: 测地距离（米），使用Haversine或Vincenty公式计算
        """
        # TODO: 调用 satsim 计算测地距离（考虑地球曲率）
        raise NotImplementedError("需要实现 satsim.geodesic_distance()")
    
    def is_navigable(self, position) -> bool:
        """检查位置是否可导航"""
        # TODO: 调用 satsim 检查可导航性
        raise NotImplementedError("需要实现 satsim.is_navigable()")
    
    def sample_navigable_point(self) -> np.ndarray:
        """采样一个可导航点"""
        # TODO: 调用 satsim 采样
        raise NotImplementedError("需要实现 satsim.sample_navigable_point()")
```

## 7. 使用示例

```python
from satnav.core import Env
from satnav.core.config import load_config
from satnav.dataset import R2RDataset

# 1. 加载配置
config = load_config("configs/vln_task.yaml")

# 2. 创建数据集
dataset = R2RDataset(config.DATASET)

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
```

## 8. 实现优先级

### Phase 1: 核心框架（当前阶段）
1. ✅ 架构设计
2. ⏳ Episode 数据结构
3. ⏳ 配置系统
4. ⏳ 数据集加载器
5. ⏳ 仿真器接口定义（待实现部分）

### Phase 2: 任务实现
1. ⏳ VLN 任务类
2. ⏳ 传感器实现
3. ⏳ 动作定义
4. ⏳ 评价指标实现

### Phase 3: 环境集成
1. ⏳ 环境类实现
2. ⏳ 数据流整合
3. ⏳ 测试和验证

### Phase 4: satsim 集成（后续）
1. ⏳ 实现 SatSimWrapper 的所有方法
2. ⏳ 与 satsim 引擎对接
3. ⏳ 完整功能测试

## 9. 设计决策说明

### 9.1 为什么不用 Registry？
- **简化设计**：减少抽象层，代码更直观
- **最小实现**：当前只需要 VLN 任务，不需要动态注册
- **易于理解**：直接导入使用，降低学习成本

### 9.2 为什么独立实现？
- **无依赖**：不依赖 Habitat，避免版本冲突
- **轻量级**：只实现必需功能，代码量小
- **可控性**：完全控制代码实现，便于定制

### 9.3 为什么保留完整接口？
- **前瞻性**：为后续 satsim 实现预留接口
- **测试友好**：可以先实现 mock 版本进行测试
- **文档清晰**：接口即文档，明确需要实现的功能

## 10. 与 VLN-CE 和 Habitat-Lab 的对比

| 特性 | Habitat-Lab | VLN-CE | SatNav |
|------|-------------|--------|--------|
| **定位** | 通用框架 | VLN专用实现 | VLN测试平台 |
| **复杂度** | 高 | 中 | 低 |
| **Registry** | ✅ | ✅ | ❌ |
| **依赖** | Habitat-Sim | Habitat-Sim | satsim（自定义） |
| **任务** | 多种 | VLN | VLN |
| **基线模型** | 可选 | ✅ | ❌ |
| **强化学习** | ✅ | ✅ | ❌ |
| **代码量** | 大 | 中 | 小 |

## 11. 总结

SatNav 是一个专注于 VLN 评测的轻量级平台，设计简洁，易于理解和扩展。核心组件清晰分离，数据流明确，为后续 satsim 集成预留了完整接口。该架构为最小功能实现，后续可根据需要逐步扩展。


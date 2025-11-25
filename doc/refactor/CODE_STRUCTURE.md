# SatNav 代码结构说明

## 目录结构

```
satnav/
├── satnav/                          # 主包
│   ├── __init__.py                  # 包初始化
│   │
│   ├── core/                        # 核心组件
│   │   ├── __init__.py
│   │   ├── env.py                  # 环境类（Env）
│   │   ├── simulator.py            # 仿真器抽象接口
│   │   ├── episode.py              # Episode数据定义
│   │   └── config.py               # 配置加载和管理
│   │
│   ├── task/                        # VLN任务模块
│   │   ├── __init__.py
│   │   ├── vln_task.py             # VLN任务主类
│   │   ├── sensors.py              # 传感器实现
│   │   ├── actions.py              # 动作定义和枚举
│   │   └── measures.py             # 评价指标实现
│   │
│   ├── dataset/                     # 数据集模块
│   │   ├── __init__.py
│   │   └── r2r_dataset.py          # R2R数据集加载器
│   │
│   └── sims/                        # 仿真器包装模块
│       ├── __init__.py
│       └── satsim_wrapper.py       # satsim仿真器包装（接口待实现）
│
├── configs/                          # 配置文件目录
│   └── vln_task.yaml               # VLN任务配置文件
│
├── examples/                         # 示例代码
│   └── example_vln.py              # 使用示例
│
├── tests/                           # 测试代码（可选）
│   └── test_env.py
│
├── ARCHITECTURE.md                   # 架构设计文档
├── ARCHITECTURE_DIAGRAM.md          # 架构图
├── CODE_STRUCTURE.md                # 本文件
└── README.md                        # 项目说明
```

## 文件详细说明

### 核心组件 (core/)

#### core/env.py
**职责**：环境类，连接所有组件，提供统一的接口

**主要类**：
```python
class Env:
    """SatNav环境类"""
    
    def __init__(self, config, dataset=None):
        """初始化环境
        Args:
            config: 配置对象
            dataset: 数据集对象（可选）
        """
    
    def reset(self):
        """重置环境，返回初始观测
        Returns:
            obs: 观测字典
        """
    
    def step(self, action):
        """执行动作
        Args:
            action: 动作字典 {"action": "MOVE_FORWARD"}
        Returns:
            obs: 新观测
            reward: 奖励（当前为0）
            done: 是否结束
            info: 信息字典（包含metrics）
        """
    
    def get_metrics(self):
        """获取当前评价指标
        Returns:
            metrics: 指标字典
        """
    
    @property
    def observation_space(self):
        """观测空间"""
    
    @property
    def action_space(self):
        """动作空间"""
```

#### core/simulator.py
**职责**：定义仿真器抽象接口

**主要类**：
```python
class AgentState:
    """智能体状态（卫星地图）"""
    position: np.ndarray  # [longitude, latitude, altitude] 经度、纬度、高度
    rotation: float      # roll角度（航向角，0-360度，0表示正北）

class Simulator(ABC):
    """仿真器抽象基类"""
    
    @abstractmethod
    def reset(self, scene_id: str):
        """重置仿真器"""
        pass
    
    @abstractmethod
    def step(self, action: str):
        """执行动作"""
        pass
    
    @abstractmethod
    def get_agent_state(self) -> AgentState:
        """获取智能体状态"""
        pass
    
    @abstractmethod
    def set_agent_state(self, position, rotation):
        """设置智能体状态"""
        pass
    
    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取传感器观测"""
        pass
    
    @abstractmethod
    def geodesic_distance(self, pos_a, pos_b) -> float:
        """计算测地距离"""
        pass
```

#### core/episode.py
**职责**：定义Episode数据结构

**主要类**：
```python
@dataclass
class InstructionData:
    """指令数据"""
    instruction_text: str

@dataclass
class NavigationGoal:
    """导航目标"""
    position: List[float]  # [x, y, z]

@dataclass
class VLNEpisode:
    """VLN Episode数据（卫星地图）"""
    episode_id: str
    scene_id: str
    start_position: List[float]  # [longitude, latitude, altitude] 经度、纬度、高度
    start_rotation: float       # roll角度（航向角，0-360度，0表示正北）
    goals: List[NavigationGoal]
    reference_path: List[List[float]]  # [[lon,lat,alt], ...]
    instruction: InstructionData
    trajectory_id: str
```

#### core/config.py
**职责**：配置加载和管理

**主要函数**：
```python
def load_config(config_path: str) -> Dict:
    """加载YAML配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        config: 配置字典
    """
```

### 任务模块 (task/)

#### task/vln_task.py
**职责**：VLN任务主类

**主要类**：
```python
class VLNTask:
    """VLN任务类"""
    
    def __init__(self, config, simulator):
        """初始化任务
        Args:
            config: 任务配置
            simulator: 仿真器对象
        """
    
    def reset(self, episode: VLNEpisode):
        """重置任务
        Args:
            episode: VLN episode
        """
    
    def step(self, action: str):
        """执行动作
        Args:
            action: 动作字符串
        Returns:
            observations: 观测字典
        """
    
    def get_observations(self) -> Dict:
        """获取所有传感器观测"""
    
    def get_metrics(self) -> Dict:
        """获取所有评价指标"""
```

#### task/sensors.py
**职责**：传感器实现

**主要类**：
```python
class Sensor(ABC):
    """传感器基类"""
    
    @abstractmethod
    def get_observation(self, **kwargs):
        """获取观测"""
        pass

class RGBSensor(Sensor):
    """RGB传感器（卫星地图）"""
    def get_observation(self, sim_obs):
        """从仿真器获取RGB图像（从卫星地图crop）"""
        return sim_obs["rgb"]

class InstructionSensor(Sensor):
    """指令传感器"""
    def get_observation(self, episode: VLNEpisode):
        """从episode获取指令"""
        return {
            "text": episode.instruction.instruction_text
        }
```

#### task/actions.py
**职责**：动作定义

**主要内容**：
```python
class Action:
    """动作枚举"""
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"

# 动作参数（当前为空，后续可扩展）
ACTION_ARGS = {
    Action.STOP: {},
    Action.MOVE_FORWARD: {},
    Action.TURN_LEFT: {},
    Action.TURN_RIGHT: {},
}
```

#### task/measures.py
**职责**：评价指标实现

**主要类**：
```python
class Measure(ABC):
    """评价指标基类"""
    
    @abstractmethod
    def reset(self, episode: VLNEpisode):
        """重置指标"""
        pass
    
    @abstractmethod
    def update(self, simulator, action):
        """更新指标"""
        pass
    
    @abstractmethod
    def get_metric(self):
        """获取指标值"""
        pass

class Success(Measure):
    """成功率指标"""
    def __init__(self, config):
        self.success_distance = config.SUCCESS_DISTANCE
    
    def update(self, simulator, action):
        distance = self._compute_distance_to_goal(simulator)
        self._metric = distance < self.success_distance
    
    def get_metric(self):
        return self._metric

class DistanceToGoal(Measure):
    """到目标距离（测地距离）"""
    def update(self, simulator, action):
        agent_pos = simulator.get_agent_state().position  # [lon, lat, alt]
        goal_pos = self._goal_position  # [lon, lat, alt]
        # 使用测地距离（考虑地球曲率），而非欧氏距离
        self._metric = simulator.geodesic_distance(agent_pos, goal_pos)

class PathLength(Measure):
    """路径长度（测地距离累加）"""
    def update(self, simulator, action):
        current_pos = simulator.get_agent_state().position  # [lon, lat, alt]
        # 使用测地距离（考虑地球曲率），而非欧氏距离
        self._metric += simulator.geodesic_distance(self._prev_pos, current_pos)
        self._prev_pos = current_pos

class SPL(Measure):
    """Success weighted by Path Length"""
    def get_metric(self):
        if not self._success:
            return 0.0
        ref_len = self._reference_path_length
        actual_len = self._actual_path_length
        return ref_len / max(ref_len, actual_len)
```

### 数据集模块 (dataset/)

#### dataset/r2r_dataset.py
**职责**：R2R数据集加载

**主要类**：
```python
class R2RDataset:
    """R2R数据集加载器"""
    
    def __init__(self, config):
        """初始化数据集
        Args:
            config: 数据集配置
        """
        self.episodes = []
        self._load_dataset(config)
    
    def _load_dataset(self, config):
        """加载数据集
        Args:
            config: 数据集配置
        """
        # 1. 读取JSON.gz文件
        # 2. 解析episodes
        # 3. 转换为VLNEpisode对象
    
    def get_episode(self, episode_id: str) -> VLNEpisode:
        """获取指定episode"""
        pass
    
    def __iter__(self):
        """迭代器"""
        return iter(self.episodes)
    
    def __len__(self):
        """数据集大小"""
        return len(self.episodes)
```

### 仿真器模块 (sims/)

#### sims/satsim_wrapper.py
**职责**：satsim仿真器包装（接口待实现）

**主要类**：
```python
class SatSimWrapper(Simulator):
    """satsim仿真器包装类"""
    
    def __init__(self, config):
        """初始化仿真器
        Args:
            config: 仿真器配置
        """
        self.config = config
        # TODO: 初始化satsim引擎
        # self._satsim = SatSim(config)
    
    def reset(self, scene_id: str):
        """重置仿真器，加载场景
        Args:
            scene_id: 场景ID
        """
        # TODO: 实现satsim.reset(scene_id)
        raise NotImplementedError("需要实现satsim.reset()")
    
    def step(self, action: str):
        """执行动作
        Args:
            action: 动作字符串
        """
        # TODO: 实现satsim.step(action)
        raise NotImplementedError("需要实现satsim.step()")
    
    def get_agent_state(self) -> AgentState:
        """获取智能体状态
        
        Returns:
            AgentState: 包含位置（经度、纬度、高度）和roll角度（航向角）
        """
        # TODO: 实现satsim.get_agent_state()
        raise NotImplementedError("需要实现satsim.get_agent_state()")
    
    def set_agent_state(self, position, rotation):
        """设置智能体状态
        
        Args:
            position: [longitude, latitude, altitude] 经度、纬度、高度
            rotation: roll角度（航向角，0-360度，0表示正北）
        """
        # TODO: 实现satsim.set_agent_state()
        raise NotImplementedError("需要实现satsim.set_agent_state()")
    
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
        # TODO: 实现satsim.get_observations()
        return {
            "rgb": rgb_image      # (H, W, 3) uint8，从卫星地图crop的图像
        }
        raise NotImplementedError("需要实现satsim.get_observations()")
    
    def geodesic_distance(self, pos_a, pos_b) -> float:
        """计算测地距离（地球表面的最短路径距离）
        
        Args:
            pos_a: [longitude, latitude, altitude] 起点位置
            pos_b: [longitude, latitude, altitude] 终点位置
        
        Returns:
            float: 测地距离（米），使用Haversine或Vincenty公式计算
        """
        # TODO: 实现satsim.geodesic_distance()（考虑地球曲率）
        raise NotImplementedError("需要实现satsim.geodesic_distance()")
    
    def is_navigable(self, position) -> bool:
        """检查位置是否可导航"""
        # TODO: 实现satsim.is_navigable()
        raise NotImplementedError("需要实现satsim.is_navigable()")
    
    def sample_navigable_point(self) -> np.ndarray:
        """采样可导航点"""
        # TODO: 实现satsim.sample_navigable_point()
        raise NotImplementedError("需要实现satsim.sample_navigable_point()")
```

## 配置文件格式

### configs/vln_task.yaml
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

## 使用示例

### examples/example_vln.py
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

# 4. 运行一个episode
obs = env.reset()
done = False
step_count = 0

while not done and step_count < config.ENVIRONMENT.MAX_EPISODE_STEPS:
    # 选择动作（这里用随机动作示例）
    action = {"action": "MOVE_FORWARD"}  # 或使用env.action_space.sample()
    
    # 执行动作
    obs, reward, done, info = env.step(action)
    
    # 打印信息
    print(f"Step {step_count}:")
    print(f"  RGB shape: {obs['rgb'].shape}")
    print(f"  Instruction: {obs['instruction']['text'][:50]}...")
    print(f"  Distance to goal: {info['metrics']['distance_to_goal']:.2f}m")
    
    step_count += 1

# 5. 获取最终评价指标
metrics = env.get_metrics()
print("\n=== Episode Results ===")
print(f"Success: {metrics['success']}")
print(f"SPL: {metrics['spl']:.4f}")
print(f"Distance to Goal: {metrics['distance_to_goal']:.2f}m")
print(f"Path Length: {metrics['path_length']:.2f}m")
```

## 实现顺序建议

### Phase 1: 基础数据结构
1. `core/episode.py` - Episode数据定义
2. `core/config.py` - 配置加载
3. `dataset/r2r_dataset.py` - 数据集加载器

### Phase 2: 仿真器接口
1. `core/simulator.py` - 仿真器抽象接口
2. `sims/satsim_wrapper.py` - satsim包装（接口定义）

### Phase 3: 任务组件
1. `task/actions.py` - 动作定义
2. `task/sensors.py` - 传感器实现
3. `task/measures.py` - 评价指标实现
4. `task/vln_task.py` - VLN任务类

### Phase 4: 环境集成
1. `core/env.py` - 环境类实现
2. `examples/example_vln.py` - 使用示例

### Phase 5: satsim集成（后续）
1. 实现 `SatSimWrapper` 的所有方法
2. 与satsim引擎对接
3. 完整功能测试

## 注意事项

1. **无Registry设计**：所有组件直接导入使用，不使用注册机制
2. **最小实现**：只实现核心功能，代码简洁
3. **接口预留**：satsim相关方法都抛出`NotImplementedError`，等待后续实现
4. **配置驱动**：所有参数通过YAML配置文件管理
5. **独立实现**：不依赖Habitat任何组件

## 扩展点

后续可以根据需要扩展：
- 添加更多评价指标（NDTW、SDTW等）
- 支持更多数据集格式
- 添加可视化功能
- 支持多环境并行
- 添加日志和调试工具


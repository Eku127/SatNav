# SatNav 项目实现计划

## 项目目标

实现一个最小功能的VLN评测平台，支持连续空间下的卫星地图导航任务，不依赖Habitat组件，使用自定义satsim仿真器。

## 实现阶段

### Phase 1: 项目基础结构

**目标**: 创建项目目录结构和基础文件

1. **创建目录结构**

- `satnav/` 主包目录
- `satnav/core/` 核心组件目录
- `satnav/task/` 任务目录
- `satnav/dataset/` 数据集目录
- `satnav/sims/` 仿真器目录
- `configs/` 配置文件目录
- `examples/` 示例代码目录

2. **创建基础文件**

- 所有 `__init__.py` 文件
- `setup.py` 安装脚本
- `requirements.txt` 依赖列表

### Phase 2: 核心数据结构

**目标**: 实现Episode、Config等基础数据结构（参考VLN-CE的连续空间设计）

1. **core/episode.py** - Episode数据定义

- `InstructionData` 类（仅instruction_text字段）
- `NavigationGoal` 类（position字段，用于连续空间的目标点）
- `VLNEpisode` 类（参考VLN-CE的VLNExtendedEpisode结构）
- `episode_id`: str
- `scene_id`: str
- `start_position`: List[float] - [longitude, latitude, altitude]
- `start_rotation`: float - roll角度（0-360度）
- `goals`: List[NavigationGoal] - **连续空间的关键**：目标位置列表（非导航图节点）
- `reference_path`: List[List[float]] - **连续空间的关键**：参考路径（连续坐标点列表，非节点序列）
- `instruction`: InstructionData
- `trajectory_id`: str
- 使用dataclass装饰器
- **关键差异**：连续空间使用连续坐标路径，而非离散导航图节点

2. **core/config.py** - 配置加载系统

- `load_config(config_path)` 函数
- 使用yaml和omegaconf加载配置
- 返回配置对象

3. **core/simulator.py** - 仿真器抽象接口

- `AgentState` dataclass（position: np.ndarray[3], rotation: float）
- `Simulator` 抽象基类（ABC）
- 定义所有抽象方法：reset, step, get_agent_state, set_agent_state, get_observations, geodesic_distance, is_navigable, sample_navigable_point

### Phase 3: 数据集模块

**目标**: 实现R2R数据集加载器（参考VLN-CE的VLNCEDatasetV1）

1. **dataset/r2r_dataset.py** - R2R数据集加载器

- `R2RDataset` 类
- `__init__(config)` 方法：加载数据集
- `from_json(json_str, scenes_dir=None)` 方法：解析JSON字符串
- **参考VLN-CE的VLNCEDatasetV1实现**：
- 支持`goals`字段（连续空间导航必需）
- 支持`reference_path`字段（连续路径，非导航图节点）
- 处理场景路径调整（类似VLN-CE的scene_id处理）
- 支持CONTENT_SCENES和EPISODES_ALLOWED过滤（可选）
- 将JSON数据转换为VLNEpisode对象
- 支持gzip压缩的JSON文件
- 正确处理连续坐标路径的解析

### Phase 4: 仿真器包装

**目标**: 实现satsim包装器接口（方法抛出NotImplementedError）

1. **sims/satsim_wrapper.py** - satsim仿真器包装

- `SatSimWrapper` 类继承自Simulator
- `__init__(config)` 方法：保存配置
- 所有方法抛出NotImplementedError并添加TODO注释
- 方法包括：reset, step, get_agent_state, set_agent_state, get_observations, geodesic_distance, is_navigable, sample_navigable_point

### Phase 5: 任务组件 - 动作和传感器

**目标**: 实现动作定义和传感器（参考VLN-CE的连续空间传感器）

1. **task/actions.py** - 动作定义

- `Action` 类：定义动作常量（STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT）
- 动作枚举或常量定义

2. **task/sensors.py** - 传感器实现

- `Sensor` 抽象基类
- `RGBSensor` 类：从sim_obs获取RGB图像（从卫星地图crop）
- `InstructionSensor` 类：从episode获取instruction_text
- **可选传感器**（参考VLN-CE，用于连续空间导航）：
- `GlobalGPSSensor`：提供当前GPS位置（经纬度），用于连续空间定位
- `VLNOracleProgressSensor`：提供到目标的相对进度（使用geodesic_distance计算）
- 实现get_observation方法
- 注意：连续空间导航中GPS和Progress传感器很有用，但当前最小实现可以只包含RGB和Instruction

### Phase 6: 任务组件 - 评价指标

**目标**: 实现VLN评价指标（使用测地距离）

1. **task/measures.py** - 评价指标实现

- `Measure` 抽象基类（reset, update, get_metric方法）
- `DistanceToGoal` 类：计算到目标的测地距离（使用simulator.geodesic_distance）
- `Success` 类：判断是否成功（distance < success_distance）
- `PathLength` 类：累加路径长度（使用测地距离）
- `SPL` 类：Success weighted by Path Length（episode结束时计算）
- **关键**：所有距离计算使用测地距离（Haversine公式），而非欧氏距离（连续空间的关键）

### Phase 7: VLN任务类

**目标**: 整合传感器和评价指标，实现VLN任务

1. **task/vln_task.py** - VLN任务类

- `VLNTask` 类
- `__init__(config, simulator)` 方法：初始化传感器和评价指标
- `reset(episode)` 方法：重置任务，设置目标，重置所有measures
- `step(action)` 方法：执行动作，更新measures，返回观测
- `get_observations()` 方法：获取所有传感器观测
- `get_metrics()` 方法：获取所有评价指标

### Phase 8: 环境类

**目标**: 实现环境类，连接所有组件

1. **core/env.py** - 环境类

- `Env` 类
- `__init__(config, dataset=None)` 方法：初始化simulator, task, dataset
- `reset()` 方法：获取episode，重置task和simulator，返回初始观测
- `step(action)` 方法：执行动作，更新指标，检查done，返回(obs, reward, done, info)
- `get_metrics()` 方法：获取当前评价指标
- `observation_space` 属性：可选，仅用于文档说明（不依赖gym，简化实现）
- `action_space` 属性：可选，仅用于文档说明（不依赖gym，简化实现）

### Phase 9: 配置文件和示例

**目标**: 创建配置文件和示例代码

1. **configs/vln_task.yaml** - VLN任务配置文件

- ENVIRONMENT配置（MAX_EPISODE_STEPS）
- SIMULATOR配置（FORWARD_STEP_SIZE, TURN_ANGLE, RGB_SENSOR）
- TASK配置（TYPE, SUCCESS_DISTANCE, POSSIBLE_ACTIONS, MEASUREMENTS）
- DATASET配置（TYPE, SPLIT, DATA_PATH, SCENES_DIR）
- 参考VLN-CE的vlnce_task.yaml格式

2. **examples/example_vln.py** - 使用示例

- 加载配置
- 创建数据集
- 创建环境
- 运行一个episode示例
- 打印评价指标

### Phase 10: 工具函数

**目标**: 实现必要的工具函数

1. **core/utils.py** - 工具函数

- 测地距离计算函数（Haversine公式）- **连续空间的关键**
- 其他辅助函数

## 关键实现细节

### 连续空间导航的特殊性（参考VLN-CE）

1. **路径表示**：

- 传统VLN：导航图节点序列
- 连续空间VLN：连续坐标点列表（reference_path）
- 实现时需要支持连续路径的解析和处理

2. **目标表示**：

- 传统VLN：导航图节点
- 连续空间VLN：连续坐标点（goals列表）
- 数据集需要正确解析goals字段

3. **传感器**：

- GPS传感器：连续空间中提供精确位置（可选，但有用）
- Progress传感器：使用geodesic_distance计算进度（连续空间的关键）

4. **距离计算**：

- 必须使用测地距离（Haversine公式），考虑地球曲率
- 不能使用欧氏距离（不适合地理坐标）

### 坐标系统

- 位置：[longitude, latitude, altitude]（地理坐标）
- 旋转：roll角度（0-360度，0表示正北）
- 距离：使用测地距离（Haversine公式），考虑地球曲率

### 无Registry设计

- 所有组件直接导入使用
- 不使用装饰器注册机制
- 简化代码结构

### 仿真器接口

- 所有satsim相关方法抛出NotImplementedError
- 添加详细的TODO注释说明需要实现的功能
- 为后续satsim集成预留完整接口

### 参考代码

- **Dataset**: VLN-CE的`VLNCEDatasetV1`和`VLNExtendedEpisode`（连续空间路径处理）
- **Sensors**: VLN-CE的`GlobalGPSSensor`和`VLNOracleProgressSensor`（连续空间定位）
- **Task**: habitat-lab的`VLNTask`和`NavigationTask`
- **Measures**: habitat-lab的nav.py中的评价指标实现（注意使用测地距离）
- **Env**: habitat-lab的`Env`类实现

## 文件清单

### 核心文件（必须实现）

- `satnav/__init__.py`
- `satnav/core/__init__.py`
- `satnav/core/episode.py`
- `satnav/core/config.py`
- `satnav/core/simulator.py`
- `satnav/core/env.py`
- `satnav/core/utils.py`（测地距离计算）
- `satnav/task/__init__.py`
- `satnav/task/actions.py`
- `satnav/task/sensors.py`
- `satnav/task/measures.py`
- `satnav/task/vln_task.py`
- `satnav/dataset/__init__.py`
- `satnav/dataset/r2r_dataset.py`
- `satnav/sims/__init__.py`
- `satnav/sims/satsim_wrapper.py`
- `configs/vln_task.yaml`
- `examples/example_vln.py`
- `setup.py`
- `requirements.txt`

## 依赖项

- numpy
- omegaconf
- pyyaml
- attrs（或使用dataclass）

**注意**：不依赖gym，简化实现

## 测试策略

- 每个阶段完成后进行基本测试
- 使用Mock仿真器测试框架逻辑
- 验证数据流和接口正确性
- 验证连续路径解析和测地距离计算
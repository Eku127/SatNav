# SatNav Core API

本文从使用者视角介绍 SatNav 的核心运行接口。重点不是逐个罗列源码，而是解释一个 Episode 如何被加载、渲染、执行和评测，以及应用或模型应该在哪一层与 SatNav 交互。

如果尚未准备环境和数据，请先阅读[环境安装](../getting-started/INSTALLATION.md)、[数据格式](../dataset/DATASET_FORMAT.md)和[示例程序](../getting-started/EXAMPLES.md)。需要接入新的导航模型时，参阅[模型接入](../development/MODEL_INTEGRATION.md)。

## 1. 核心对象

SatNav 的运行栈由 Dataset、Environment、Task 和 Simulator 四部分组成：

```text
SatNavDataset
    └── VLNEpisode
           │
           v
          Env
           │
           └── VLNTask
                  ├── Sensors
                  ├── Measures
                  └── Simulator
                         └── SatSimWrapper -> SatSim -> GeoTIFF
```

| 对象 | 作用 |
| --- | --- |
| `SatNavDataset` | 从 Episode JSON 创建 `VLNEpisode` 对象，并解析本地场景路径 |
| `VLNEpisode` | 保存指令、起点、目标、reference path 和场景逻辑标识 |
| `Env` | 管理 Episode 顺序、step 计数、终止状态和公共交互接口 |
| `VLNTask` | 组织 sensor、action 和 measure，并连接 `Env` 与 simulator |
| `Simulator` | 定义场景加载、状态设置、动作执行和 observation 获取接口 |
| `SatSimWrapper` | 将内置 SatSim 适配到公共 `Simulator` 接口 |

大多数用户只需要直接使用 `Env`。应用和 baseline 不应访问 `env._dataset`、`env._task` 或 `env._sim` 等私有成员。

## 2. 最小示例

仓库内置了两个合成 Episode 和一张 CC0 GeoTIFF，可以在没有下载 SatNav-v0.1 的情况下测试 Core API：

```python
from applications.resources import load_example_task_config
from satnav.core.env import Env

config = load_example_task_config()
env = Env(config)

try:
    observation = env.reset()

    print(env.current_episode.episode_key)
    print(observation["rgb"].shape)
    print(observation["instruction"]["text"])
    print(observation["agent_pose"])

    observation, done, info = env.step("STOP")
    print(done, info["termination_reason"])
    print(env.get_metrics())
finally:
    env.close()
```

这段代码完成了一次完整的环境生命周期：创建环境、加载 Episode、获取初始 observation、执行动作、读取指标并释放资源。

> `Env` 当前没有 context manager 接口，建议始终使用 `try/finally` 调用 `close()`。`close()` 可以重复调用。

## 3. 配置

`Env` 接收完整的 `DictConfig` 或 Python `dict`。推荐通过 `load_config()` 加载 YAML：

```python
from satnav.core.config import load_config

config = load_config("configs/satnav_eval_task.yaml")
```

配置由四个核心 section 组成：

```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  TYPE: satsim
  FORWARD_STEP_SIZE: 10
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 448
    HEIGHT: 448
    HFOV: 90

TASK:
  TYPE: VLN
  SUCCESS_DISTANCE:
    DEFAULT: 10.0
    Boundary: 10.0
    LandmarkSet: 30.0
    Road: 10.0
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, ORACLE_SUCCESS, SPL, PATH_LENGTH]

DATASET:
  TYPE: SatNav
  SPLIT: val_seen
  DATA_PATH: path/to/all_episodes.json
  SCENES_DIR: path/to/scenes
```

| 配置 | 说明 |
| --- | --- |
| `ENVIRONMENT.MAX_EPISODE_STEPS` | 一个 Episode 允许执行的最大动作数 |
| `SIMULATOR.TYPE` | 内置 simulator 使用 `satsim` |
| `SIMULATOR.FORWARD_STEP_SIZE` | `MOVE_FORWARD` 的移动距离，单位为米 |
| `SIMULATOR.TURN_ANGLE` | 左右转的角度，单位为度 |
| `SIMULATOR.RGB_SENSOR` | RGB observation 的尺寸和水平视场角 |
| `TASK.SUCCESS_DISTANCE` | 默认或按 trajectory type 设置的成功半径 |
| `TASK.POSSIBLE_ACTIONS` | 任务允许的离散动作 |
| `TASK.MEASUREMENTS` | 启用的评测指标 |
| `DATASET.SPLIT` | 当前数据划分，同时参与构造稳定 Episode key |
| `DATASET.DATA_PATH` | Episode JSON 或 JSON.GZ 路径，支持 `{split}` 占位符 |
| `DATASET.SCENES_DIR` | `<scene_id>.tif` 所在目录 |

在线评测应使用 `configs/satnav_eval_task.yaml`。该配置中 Boundary、LandmarkSet 和 Road 的成功半径分别为 10 米、30 米和 10 米。轨迹生成配置中的 LandmarkSet 3 米是 expert waypoint 到达半径，不是在线评测阈值。

真实数据路径不应写入公共配置。建议复制配置到 ignored 的 `configs/local_*.yaml`，或通过本地环境变量注入。

## 4. 创建 Env

构造函数签名为：

```python
Env(config, dataset=None, cycle=False)
```

| 参数 | 说明 |
| --- | --- |
| `config` | 包含 `ENVIRONMENT`、`SIMULATOR`、`TASK` 和 `DATASET` 的完整配置 |
| `dataset` | 可选的 `SatNavDataset`；未提供时由 `config.DATASET` 自动创建 |
| `cycle` | Episode 耗尽后是否从头开始，默认 `False` |

评测时应保留 `cycle=False`，避免 Episode 被重复执行。训练或持续交互场景可以使用 `cycle=True`：

```python
env = Env(config, cycle=True)
```

也可以先显式创建 Dataset，再传入 Env：

```python
from satnav.dataset import SatNavDataset
from satnav.core.env import Env

dataset = SatNavDataset(config.DATASET)
env = Env(config, dataset=dataset)
```

当显式传入 Dataset 时，`Env` 仍会从完整配置中读取 simulator 和 task 参数，并优先使用 `DATASET.SCENES_DIR` 解析场景。

## 5. Episode 生命周期

### 5.1 `reset()`

```python
observation = env.reset()
```

`reset()` 从 Dataset iterator 获取下一个 Episode，然后依次完成：

1. 清空 step 数、终止状态和上一次 `info`；
2. 将 logical `scene_id` 解析到本地 GeoTIFF；
3. 加载或复用场景；
4. 将 agent 设置到 `start_position` 和 `start_rotation`；
5. 重置 sensors 和 measures；
6. 返回初始 observation。

当 `cycle=False` 且全部 Episode 已耗尽时，再次调用 `reset()` 会抛出 `RuntimeError`。当 Dataset 为空时，Env 会在构造阶段拒绝启动。

### 5.2 `reset_to_episode()`

```python
episode = env.episodes[0]
observation = env.reset_to_episode(episode)
```

`reset_to_episode()` 绕过 Dataset iterator，直接加载指定的 `VLNEpisode`。它适合调试、固定 Episode 可视化和确定性数据收集。

调用后：

- `env.current_episode` 指向传入的 Episode；
- `env.episode_over` 变为 `False`；
- `env.last_step_info` 变为 `None`；
- elapsed step 重新从 0 开始。

### 5.3 `step()`

```python
observation, done, info = env.step("MOVE_FORWARD")
```

每次 `step()` 会执行动作、更新 measure、生成下一帧 observation，并将 elapsed step 加一。Episode 只会因为以下原因结束：

- agent 执行 `STOP`；
- 达到 `MAX_EPISODE_STEPS`。

进入成功半径本身不会结束 Episode。agent 必须在合适的位置主动执行 `STOP`，`SUCCESS` 才可能为 1。

在 `reset()` 之前调用 `step()`，或在 `done=True` 后继续调用 `step()`，都会抛出 `RuntimeError`。

### 5.4 `close()`

```python
env.close()
```

`close()` 释放 task 和 simulator 持有的资源。内置 SatSim 会关闭缓存的 Rasterio scene，并清空场景缓存。重复调用是安全的。

## 6. Observation

默认 `VLNTask` 在 `reset()` 和每次 `step()` 后返回三个 observation：

| Key | 类型 | 形状/结构 | 说明 |
| --- | --- | --- | --- |
| `rgb` | `numpy.ndarray` | `(H, W, 3)`, `uint8` | 根据 agent pose 从 GeoTIFF 渲染的 RGB 图像 |
| `instruction` | `dict` | `{"text": str}` | 当前 Episode 的自然语言指令 |
| `agent_pose` | `numpy.ndarray` | `(4,)`, `float32` | 相对 Episode 起点的 ego-frame pose |

RGB 尺寸由 `SIMULATOR.RGB_SENSOR.WIDTH` 和 `HEIGHT` 决定，不应在应用或模型中硬编码为 224 或 448。

### 6.1 Agent pose

`agent_pose` 的定义为：

```text
[delta_forward_m, delta_right_m, sin(delta_heading), cos(delta_heading)]
```

- `delta_forward_m`：沿初始朝向的位移，单位为米；
- `delta_right_m`：初始朝向右侧的位移，单位为米；
- `delta_heading`：当前 heading 相对初始 heading 的变化。

Episode 刚 reset 时，其值为：

```text
[0.0, 0.0, 0.0, 1.0]
```

使用正弦和余弦表示 heading，可以避免 `-180°` 与 `180°` 处的不连续。这个 pose 是相对坐标，不依赖场景的绝对南北方向。

### 6.2 `observation_space`

```python
print(env.observation_space)
```

`observation_space` 是便于应用读取的轻量描述字典，不是 `gym.Space`。当前描述重点提供 RGB shape 和 instruction 结构；实际 task observation 还包含 `agent_pose`，因此编写通用 adapter 时应以 `reset()` 或 `step()` 返回的 key 为准。

## 7. Action

SatNav 使用四个离散动作：

| ID | Action | 效果 |
| ---: | --- | --- |
| 0 | `STOP` | 不移动，并结束当前 Episode |
| 1 | `MOVE_FORWARD` | 沿当前 heading 前进 `FORWARD_STEP_SIZE` 米 |
| 2 | `TURN_LEFT` | heading 减少 `TURN_ANGLE` 度 |
| 3 | `TURN_RIGHT` | heading 增加 `TURN_ANGLE` 度 |

`step()` 接受三种等价输入：

```python
env.step("MOVE_FORWARD")
env.step(1)
env.step({"action": "MOVE_FORWARD"})
```

动作字符串区分大小写。无效字符串、越界整数、缺少 `action` 的字典或其他类型会触发异常。

```python
print(env.action_space["actions"])
# ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
```

SatSim 会保证完整相机视野位于场景范围内。如果一次前进会让相机 footprint 越出 GeoTIFF，agent 会停留在原位；该动作仍会消耗一个 step。转向不会改变位置。

> 离线 trajectory 中的 `-1` 是初始 observation 的对齐标记，不是可以传给 `Env.step()` 的动作。

## 8. `step()` 返回值

```python
observation, done, info = env.step(action)
```

| 返回值 | 说明 |
| --- | --- |
| `observation` | 执行动作后的 sensor observation |
| `done` | 是否因 `STOP` 或最大步数而结束 |
| `info` | 当前指标、Episode identity 和终止状态 |

`info` 的稳定字段为：

| 字段 | 说明 |
| --- | --- |
| `metrics` | 当前已启用 measure 的结果 |
| `episode_id` | 当前 Episode 的原始 ID |
| `episode_key` | `<split>::<scene_id>::<episode_id>` 稳定标识 |
| `scene_id` | 场景逻辑名称，不包含本机路径 |
| `elapsed_steps` | 当前已执行的动作数 |
| `episode_over` | 与返回值 `done` 一致 |
| `stop_called` | 本 step 是否已经触发 STOP 终止 |
| `max_steps_reached` | 是否达到最大 step 数 |
| `termination_reason` | `"stop"`、`"max_steps"` 或 `None` |

`env.last_step_info` 指向最近一次 `step()` 返回的 `info`；在 reset 后为 `None`。

## 9. 完整交互循环

一个典型的 policy rollout 如下：

```python
from applications.resources import load_example_task_config
from satnav.core.env import Env


def choose_action(observation):
    # 在这里调用自己的 policy。
    return "STOP"


env = Env(load_example_task_config())
try:
    observation = env.reset()
    done = False

    while not done:
        action = choose_action(observation)
        observation, done, info = env.step(action)

    metrics = env.get_metrics()
    print(env.current_episode.episode_key)
    print(info["termination_reason"])
    print(metrics)
finally:
    env.close()
```

真实评测不应依赖 Episode 在 JSON 中的原始顺序，也不应只用 `episode_id` 关联结果。请使用 `episode_key` 作为跨场景、跨 split 的稳定 identity。

## 10. Metrics

`TASK.MEASUREMENTS` 决定 `get_metrics()` 和 `info["metrics"]` 中出现哪些结果：

| 配置名 | 返回 key | 说明 |
| --- | --- | --- |
| `DISTANCE_TO_GOAL` | `distance_to_goal` | 当前状态到第一个 goal 的距离，单位为米 |
| `SUCCESS` | `success` | 在成功半径内执行 STOP 时为 1，否则为 0 |
| `ORACLE_SUCCESS` | `oracle_success` | Episode 过程中曾进入成功半径则为 1，不要求 STOP |
| `PATH_LENGTH` | `path_length` | 相邻 agent position 的累计距离，单位为米 |
| `SPL` | `spl` | 成功率与路径效率的组合指标 |
| `TOP_DOWN_MAP` | `top_down_map` | 用于调试和视频的可视化数据 |

### 10.1 Success

普通 Episode 的 Success 定义为：

```text
STOP 已执行，并且 distance_to_goal < success_distance
```

对于起点与目标重合的 Boundary Episode，SatNav 使用 leave-and-return 逻辑：agent 必须先离开起点超过 `2 × success_distance`，再返回成功半径并执行 STOP。这样可以避免 agent 原地 STOP 获得成功。

### 10.2 Oracle Success

Oracle Success 衡量 agent 是否曾经到达目标区域，不要求最终在该处 STOP。一旦变为 1，在当前 Episode 剩余步骤中保持为 1。Boundary Episode 同样必须先离开起点区域再返回。

### 10.3 SPL

SatNav 使用 reference path 的地理距离计算 SPL：

```text
SPL = Success × reference_path_length
                / max(reference_path_length, actual_path_length)
```

当 Episode 没有有效 reference path 时，系统尝试使用起点到目标的直线距离；如果仍不存在有效路径长度，SPL 为 0。

### 10.4 Top-down map

启用 `TOP_DOWN_MAP` 后，返回值为字典：

| 字段 | 说明 |
| --- | --- |
| `map` | `(H, W, 3)` 的 top-down map 图像 |
| `agent_map_coord` | agent 在 map 中的 `(row, col)` |
| `agent_angle` | 当前 heading |
| `bounds` | 可视化区域的地理边界 |
| `step_count` | 当前可视化记录的 step 数 |

Top-down map 会增加渲染和内存开销。训练或大规模评测不需要视频时，可以从 `TASK.MEASUREMENTS` 中移除它。

## 11. Env 公共属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `episodes` | `list[VLNEpisode]` | 当前 Dataset 中的 Episode；无 Dataset 时为空列表 |
| `current_episode` | `VLNEpisode | None` | 当前活动 Episode |
| `simulator` | `Simulator` | 当前 simulator 公共接口 |
| `agent_state` | `AgentState` | 当前 WGS84 position 和 heading |
| `last_step_info` | `dict | None` | 最近一次 step 的 info |
| `episode_over` | `bool` | 当前 Episode 是否结束 |
| `observation_space` | `dict` | observation 的轻量描述 |
| `action_space` | `dict` | 支持动作的轻量描述 |
| `max_episode_steps` | `int` | 当前 Episode 最大 step 数 |

读取 agent 状态：

```python
state = env.agent_state
print(state.position)  # [longitude, latitude, altitude]
print(state.rotation)  # 0° = North, clockwise
```

`AgentState.position` 是 list-like 对象，并支持 `.tolist()`。

## 12. Episode API

`env.current_episode` 返回 `VLNEpisode`。常用字段的完整定义参阅[数据格式](../dataset/DATASET_FORMAT.md)。Core API 中最重要的是 identity 与序列化边界：

```python
episode = env.current_episode

print(episode.episode_id)
print(episode.scene_id)
print(episode.episode_key)

public_data = episode.to_dict()
debug_data = episode.to_dict(include_runtime=True)
```

`episode.to_dict()` 默认只返回可公开序列化的 benchmark 字段，不包含本机路径。只有显式设置 `include_runtime=True` 时，才会加入：

- `split`；
- `scene_path`；
- `episode_key`。

`scene_id` 始终应保持为稳定的 logical ID。不要把本机 GeoTIFF 绝对路径写入 benchmark 结果。

## 13. Simulator API

大多数代码应通过 Env 与 simulator 交互。需要 navigation helper 或调试底层状态时，可以使用 `env.simulator`：

| 方法/属性 | 说明 |
| --- | --- |
| `reset(scene_id)` | 加载场景并重置 simulator scene 状态 |
| `set_agent_state(position, rotation)` | 设置 WGS84 position 和 heading |
| `get_agent_state()` | 返回 `AgentState` |
| `step(action)` | 直接执行 simulator action |
| `get_observations()` | 获取 simulator 原始 observation |
| `geodesic_distance(a, b)` | 计算两个地理位置之间的距离 |
| `is_navigable(position)` | 检查完整相机视野是否位于场景安全范围内 |
| `sensor_suite` | simulator sensor 描述 |
| `action_space` | simulator 支持的动作 |
| `close()` | 释放 simulator 资源 |

直接调用 `simulator.step()` 不会更新 `Env` 的 elapsed step、termination state 或 task measures，因此正常 rollout 必须使用 `env.step()`。

### 13.1 坐标与渲染

公共 API 使用 WGS84：

```text
[longitude, latitude, altitude]
```

heading 以正北为 `0°`，顺时针增加。SatSim 内部将 WGS84 转为 EPSG:3857 Web Mercator，在平面米制坐标中完成移动和相机范围计算，然后从 GeoTIFF 裁剪、旋转并缩放得到 RGB observation。

相机地面覆盖范围由 altitude、HFOV 和图像宽高共同决定。altitude 越高，单帧覆盖的地面区域越大；如果场景尺寸不足以容纳完整相机 footprint，初始状态会被拒绝或前进动作会停留在原位。

### 13.2 场景解析与缓存

Episode 中只保存 logical `scene_id`。`SatNavDataset` 根据 `SCENES_DIR` 生成 runtime-only `scene_path`，SatSim 在加载时自动补充 `.tif` 后缀。

同一个 scene 连续用于多个 Episode 时，SatSim 会复用已打开的 Rasterio scene，减少重复加载。调用 `close()` 后缓存会被关闭并清空。

## 14. 外部 Simulator

除内置 `satsim` 外，SatNav 支持通过完整类路径加载外部 backend：

```yaml
SIMULATOR:
  TYPE: custom
  CLASS: my_package.my_simulator.MySimulator
```

外部类必须：

1. 继承 `satnav.core.simulator.Simulator`；
2. 构造函数接受 `(config, scenes_dir)`；
3. 实现 `reset`、`step`、agent state、observation、distance 和 navigability 等抽象接口；
4. 返回与 Task sensor 兼容的 `rgb` observation。

最小结构如下：

```python
from satnav.core.simulator import AgentState, Simulator


class MySimulator(Simulator):
    def __init__(self, config, scenes_dir):
        ...

    def reset(self, scene_id):
        ...

    def step(self, action):
        ...

    def get_agent_state(self) -> AgentState:
        ...

    def set_agent_state(self, position, rotation):
        ...

    def get_observations(self):
        ...

    def geodesic_distance(self, position_a, position_b):
        ...

    def is_navigable(self, position):
        ...

    @property
    def sensor_suite(self):
        ...

    @property
    def action_space(self):
        ...
```

未知 `TYPE` 不会自动回退到 SatSim。未配置 `SIMULATOR.CLASS`、类无法导入或没有继承 `Simulator` 时，factory 会直接报错。

## 15. 常见问题

### 到达目标后为什么 `done` 仍然是 `False`？

到达成功半径不会自动终止 Episode。policy 需要执行 `STOP`；否则 Episode 会继续，直到达到最大 step 数。

### 已经执行 `STOP`，为什么 `success` 仍然为 0？

检查 STOP 时的 `distance_to_goal` 是否小于当前 trajectory type 的 `SUCCESS_DISTANCE`。对于 Boundary Episode，还必须先离开起点区域再返回。

### 为什么执行 `MOVE_FORWARD` 后位置没有变化？

目标位置可能使旋转后的完整相机视野越出 GeoTIFF 安全范围。此时 SatSim 保持原位，但该动作仍计入 elapsed step。

### 为什么 `reset()` 报告所有 Episode 已耗尽？

默认 `cycle=False`。评测完成后应结束运行；需要循环数据时，在构造 Env 时设置 `cycle=True`。

### 为什么 RGB shape 不是 224 × 224？

RGB shape 由当前 simulator 和配置决定。读取 `observation["rgb"].shape` 或 `env.observation_space["rgb"]["shape"]`，不要硬编码分辨率。

### 为什么找不到场景？

确认 `DATASET.SCENES_DIR` 下存在与 logical `scene_id` 同名的 GeoTIFF，例如：

```text
<SCENES_DIR>/Amsterdam-1.tif
```

数据和场景配置可以通过以下命令检查：

```bash
bash scripts/validation/data_validation.sh
```

### 应该使用 Env 还是直接使用 SatSim？

模型 rollout、应用和评测应使用 Env，因为它会同步更新 observation、metrics、终止状态和 Episode identity。只有实现 simulator backend、底层渲染调试或 navigation helper 时，才需要直接访问 `env.simulator`。

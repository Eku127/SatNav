# SatNav 数据格式

本文介绍 SatNav-v0.1 的 Episode、场景和离线 trajectory 数据格式。Episode 数据只包含导航元数据；卫星场景和离线 trajectory 需要用户另行准备。

## 1. 数据组成

| 数据 | 主要文件 | 说明 |
| --- | --- | --- |
| Episode | `episodes/**/all_episodes.json` | 指令、起点、目标、waypoint 和 reference path |
| 场景列表 | `scenes_list.yaml` | 59 个场景的逻辑名称和经纬度范围，不包含影像 |
| GeoTIFF 场景 | `<scene_id>.tif` | 用户自行下载，用于 SatSim 渲染 observation |
| 离线 trajectory | `annotations.json` 和 `images/` | 根据 train Episode 和 GeoTIFF 生成，用于模型训练 |

Episode 下载参阅[Episode 数据下载](DATA_DOWNLOAD.md)，场景和 trajectory 的生成分别参阅[卫星场景下载](APPLICATION_MAP_DOWNLOAD.md)和[轨迹数据生成](APPLICATION_TRAJ_GENERATION.md)。

## 2. 数据划分

| Split | Episodes | Scenes | 说明 |
| --- | ---: | ---: | --- |
| train | 105,164 | 56 | 模型训练 |
| val_seen | 4,574 | 56 | 使用 train 中出现过的场景 |
| val_unseen | 8,756 | 3 | 使用 train 中未出现的场景 |
| **总计** | **118,494** | **59** |  |

三个 split 的核心文件均采用相同结构：

```text
episodes/
├── train/all_episodes.json
└── eval/
    ├── val_seen/all_episodes.json
    └── val_unseen/all_episodes.json
```

每个 JSON 文件的顶层是一个 `episodes` 列表：

```json
{
  "episodes": [
    {
      "episode_id": 0,
      "trajectory_id": 0,
      "trajectory_type": "Road",
      "trajectory_subtype": "road",
      "scene_id": "Amsterdam-1",
      "start_position": [4.8784032, 52.3762329, 50],
      "start_rotation": 90.0,
      "goals": [{"position": [4.8810, 52.3770, 50]}],
      "instruction": {
        "instruction_text": "Continue along the road and stop at the junction.",
        "instruction_type": "natural"
      },
      "waypoints": [
        [4.8784032, 52.3762329, 50],
        [4.8810, 52.3770, 50]
      ],
      "reference_path": [
        [4.8784032, 52.3762329, 50],
        [4.8797, 52.3766, 50],
        [4.8810, 52.3770, 50]
      ],
      "aux_info": {}
    }
  ]
}
```

> 上述内容是用于说明字段结构的简化示例，不对应完整的发布 Episode。

## 3. Episode 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `episode_id` | integer/string | Episode 在当前 split 和 scene 中的标识 |
| `trajectory_id` | integer/string | 导航路线标识；同一路线可以对应多条不同风格的指令 |
| `trajectory_type` | string | `Boundary`、`LandmarkSet` 或 `Road` |
| `trajectory_subtype` | string | 更细粒度的任务类型 |
| `scene_id` | string | 稳定的场景逻辑名称，例如 `Amsterdam-1` |
| `start_position` | list | 起点 `[longitude, latitude, altitude]` |
| `start_rotation` | number | 初始朝向，单位为度 |
| `goals` | list | 目标列表，通常包含一个 `position` |
| `instruction` | object | 导航指令及指令类型 |
| `waypoints` | list | 原始稀疏导航点 |
| `reference_path` | list | 从起点到目标的稠密参考路径 |
| `aux_info` | object | 任务相关的附加元数据 |

`episode_id` 和 `trajectory_id` 都不应被当作全局唯一标识。跨文件保存或关联结果时，应使用：

```text
<split>::<scene_id>::<episode_id>
```

SatNav loader 会将 ID 转为字符串，并保留 Episode、instruction 和 goal 中未识别的扩展字段。

## 4. 坐标与场景

位置统一使用 WGS84 坐标：

```text
[longitude, latitude, altitude]
```

- `longitude`、`latitude`：十进制度；
- `altitude`：米，用于控制 SatSim observation 的覆盖范围；
- `start_rotation`：以正北为 `0°`，顺时针增加，`90°` 表示正东。

`scene_id` 是逻辑名称，不是本地文件路径。配置 `DATASET.SCENES_DIR` 后，SatNav 会在运行时将其解析到对应场景：

```text
<SCENES_DIR>/Amsterdam-1.tif
```

本机路径只保存在 runtime-only 的 `scene_path` 中，默认不会写入 Episode 或评测结果。

`scenes_list.yaml` 中的 `lat1`、`lon1`、`lat2` 和 `lon2` 定义场景下载范围，但该文件本身不包含地图瓦片或卫星影像。

## 5. 任务类型

| Task | `trajectory_type` | 常见 `trajectory_subtype` |
| --- | --- | --- |
| Boundary | `Boundary` | `loop`、`arc`、`extended` |
| Landmark | `LandmarkSet` | `one_turn`、`two_turn` |
| Road | `Road` | `road`、`waterway`、`hybrid` |

`waypoints` 表示任务生成时的高层路径节点，`reference_path` 是用于导航和评测的稠密路径。生成 expert trajectory 时应使用 `reference_path`，不要根据指令文本重新构造路径。

## 6. 离线 trajectory

离线 trajectory 不包含在 SatNav-Episodes-v0.1 下载包中。使用仓库生成器后，默认结构为：

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <scene_id>_satnav_<episode-index>/
        ├── .done
        ├── .annotation.json
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

`annotations.json` 是用于训练的 JSON 列表。单条 annotation 的结构如下：

```json
{
  "id": 0,
  "trajectory_id": "0",
  "steps": 3,
  "video": "images/Amsterdam-1_satnav_000000",
  "instructions": ["Continue along the road and stop at the junction."],
  "actions": [-1, 1, 1, 0]
}
```

| 字段 | 说明 |
| --- | --- |
| `id` | Episode 在源 JSON `episodes` 列表中的索引，不等同于 `episode_id` |
| `trajectory_id` | 从源 Episode 复制的路线标识 |
| `steps` | 可执行动作数量，等于 `len(actions) - 1` |
| `video` | RGB 帧目录的相对路径；该字段不是 MP4 文件 |
| `instructions` | 与该 trajectory 对应的指令列表 |
| `actions` | 与 observation 对齐的离散动作序列 |

动作编码为：

| ID | Action |
| ---: | --- |
| -1 | `INIT`，表示执行第一个动作前的 observation |
| 0 | `STOP` |
| 1 | `MOVE_FORWARD` |
| 2 | `TURN_LEFT` |
| 3 | `TURN_RIGHT` |

生产配置生成 448 × 448 RGB JPEG。每条 trajectory 满足：

```text
JPEG 数量 = len(actions) = steps + 1
```

`summary.json`、`.done` 和 `.annotation.json` 用于完整性检查与断点续跑，不建议手动修改。模型训练应读取公开的 `annotations.json` 和对应的 `images/`。

## 7. 使用 SatNavDataset 加载

```python
from omegaconf import OmegaConf
from satnav.dataset import SatNavDataset

config = OmegaConf.create({
    "DATA_PATH": "data/satnav_datasets/SatNav-v0.1/episodes/train/all_episodes.json",
    "SPLIT": "train",
    "SCENES_DIR": "data/satnav_datasets/scenes",
})

dataset = SatNavDataset(config)
episode = dataset.episodes[0]

print(len(dataset.episodes))
print(episode.episode_key)
print(episode.scene_id, episode.scene_path)
```

`SatNavDataset` 同时支持 `.json`、`.json.gz` 和带 `{split}` 占位符的路径。加载前可以运行：

```bash
bash scripts/validation/data_validation.sh
```

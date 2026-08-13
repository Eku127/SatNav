# SatNav 示例

SatNav 提供两个可直接运行的导航示例，用于演示环境创建、Episode 执行、指标计算和结果可视化。

| 示例 | 作用 | 默认输出 |
| --- | --- | --- |
| `reference_follower_example.py` | 使用 `ReferencePathFollower` 运行一个 Episode，依次跟随数据中的 reference path | 最终 observation、top-down map 和终端指标 |
| `satnav_path_follower_example.py` | 使用 `SatNavPathFollower` 运行全部 Episode，逐个 waypoint 进行贪心导航 | Episode 指标 JSON 和可选 MP4 视频 |

> 两个示例默认使用仓库内置的合成 Episode 和 CC0 场景，不需要提前下载 SatNav-v0.1 数据或卫星场景。

## 1. 准备环境

先完成[环境安装](INSTALLATION.md)，然后在 SatNav 仓库根目录执行：

```bash
python -m pip install -e .
```

如果需要生成 MP4 视频，请安装 applications 依赖：

```bash
python -m pip install -e '.[applications]'
```

默认示例配置和数据位于：

```text
applications/resources/
├── map.tif
├── satnav_example_episodes.json
└── satnav_example_task.yaml
```

## 2. Reference Path Follower

该示例读取一个 Episode，并沿 Episode 中记录的 reference path 依次导航。它适合快速了解 SatNav 的环境交互过程，以及检查 reference path 是否能够被正确执行。

运行示例：

```bash
python examples/reference_follower_example.py
```

运行完成后，终端会输出 Episode 信息、动作统计和 `Success`、`SPL`、`Distance to Goal`、`Path Length` 等指标，并在以下目录保存图像：

```text
output/examples/reference_follower/
├── final_observation_example.png
└── topdown_map_example.png
```

![Reference Path Follower top-down map](assets/examples/topdown_map_example.png)

*Reference Path Follower 的 top-down map 输出。*

指定输出目录：

```bash
python examples/reference_follower_example.py \
  --output-dir output/my_reference_example
```

## 3. SatNav Path Follower

该示例遍历配置中的全部 Episode。`SatNavPathFollower` 会依次以 reference path 中的 waypoint 为目标，根据当前位置和方向贪心选择前进或转向动作。

首次运行时建议关闭视频生成，以便快速检查环境和数据配置：

```bash
python examples/satnav_path_follower_example.py --no-video
```

运行完成后，结果会写入：

```text
output/examples/satnav_path_follower/
└── satnav_example_episodes_<timestamp>.json
```

JSON 文件包含总体成功率、平均 SPL、平均步数，以及每个 Episode 的路径长度、终点距离和动作分布。

生成导航视频：

```bash
python examples/satnav_path_follower_example.py
```

每个 Episode 会生成一个包含 RGB observation 和 top-down map 的视频：

```text
output/examples/satnav_path_follower/
├── episode_<episode_id>_video.mp4
└── satnav_example_episodes_<timestamp>.json
```

> 视频生成需要 `imageio-ffmpeg`，并要求任务配置启用 `TOP_DOWN_MAP` measurement。

<video controls width="100%">
  <source src="assets/examples/episode_1602_video.mp4" type="video/mp4">
</video>

*SatNavPathFollower 视频示例（NewYork-3，Episode 1602）：左侧为 RGB observation，右侧为 top-down map，底部为导航指令。*

如果当前文档页面不支持内嵌播放，可以[下载或观看 MP4 视频](assets/examples/episode_1602_video.mp4)。

## 4. 使用自定义配置

两个示例均支持通过 `--config` 使用其他任务配置：

```bash
python examples/reference_follower_example.py \
  --config path/to/task.yaml \
  --output-dir output/reference_follower

python examples/satnav_path_follower_example.py \
  --config path/to/task.yaml \
  --output-dir output/satnav_path_follower \
  --no-video
```

自定义配置至少需要正确设置：

- `DATASET.DATA_PATH`：Episode JSON 路径；
- `DATASET.SCENES_DIR`：GeoTIFF 场景目录；
- `TASK.POSSIBLE_ACTIONS`：任务支持的动作；
- `TASK.SUCCESS_DISTANCE`：不同 trajectory type 的成功距离；
- `SIMULATOR.FORWARD_STEP_SIZE` 和 `SIMULATOR.TURN_ANGLE`：移动与转向步长。

建议先使用内置资源确认示例能够正常运行，再替换为自己的 Episode 和场景配置。

## 5. 两个示例的区别

`ReferencePathFollower` 直接接收完整 reference path，并在单个 Episode 中跟随路径；`SatNavPathFollower` 接收当前 waypoint，根据模拟器状态实时计算下一步动作，并支持批量 Episode、结构化指标和视频输出。

这两个 follower 都是确定性的导航辅助工具，不是需要训练或加载 checkpoint 的学习模型。

# SatSim Viewer 使用指南

本文介绍如何使用 `applications/satsim_viewer` 交互式检查 GeoTIFF 场景和 SatNav VLN Episode。

| 模式 | 用途 |
| --- | --- |
| `free` | 在单个 GeoTIFF 场景中自由移动，查看坐标、朝向和高度 |
| `task` | 按 Episode 查看 RGB observation、指令、waypoint、top-down map 和评测指标 |

相机高度、视场角、朝向和画面之间的关系见[SatSim 观测原理](../concepts/SATSIM.md)，其中包含真实卫星场景的参数对比图。

## 1. 准备环境

先完成[环境安装](../getting-started/INSTALLATION.md)。如需查看真实场景或 Episode，还需完成 [Episode 数据下载](../dataset/DATA_DOWNLOAD.md)和[卫星场景下载](MAP_DOWNLOAD.md)。

在 SatNav 仓库根目录执行：

```bash
python -m pip install -e '.[applications]'
```

> Viewer 使用 OpenCV 打开交互式窗口，需要桌面显示环境。通过 SSH 使用时，请配置 X11 forwarding，并确认 `DISPLAY` 环境变量可用。

## 2. 运行内置示例

仓库内置了合成 GeoTIFF 和两个示例 Episode，用于测试。

自由查看场景：

```bash
python -m applications.satsim_viewer free
```

![Free viewer：显示当前 observation、坐标、高度和朝向](../../assets/viewer/free_viewer.png)

<p class="figure-caption" align="center"><em>Free viewer 使用合成 GeoTIFF 渲染 observation，并在画面中显示 WGS84、Web Mercator、相机高度和朝向。</em></p>

查看 VLN Episode：

```bash
python -m applications.satsim_viewer task
```

![Task viewer：同时显示 RGB observation、top-down map 和任务信息](../../assets/viewer/task_viewer.png)

<p class="figure-caption" align="center"><em>Task viewer 左侧显示 RGB observation，右侧显示 agent、waypoint 和 reference path，底部显示指令及距离。</em></p>

不指定模式时默认进入 `free`：

```bash
python -m applications.satsim_viewer
```

## 3. 查看自定义 GeoTIFF

复制 free viewer 配置到 Git 忽略的本地目录：

```bash
mkdir -p .local
cp applications/satsim_viewer/config.yaml .local/satsim_viewer_free.yaml
```

修改 `TIF_PATH`，并按需调整移动、相机和初始高度参数：

```yaml
TIF_PATH: /absolute/path/to/Geneva-1.tif

SIMULATOR:
  FORWARD_STEP_SIZE: 30
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 512
    HEIGHT: 512
    HFOV: 90

AGENT:
  ALTITUDE: 100.0
  ROTATION: 0.0
  ALTITUDE_STEP_SIZE: 10.0
```

启动 viewer：

```bash
python -m applications.satsim_viewer free \
  --config .local/satsim_viewer_free.yaml
```

free viewer 从场景中心启动，同时显示 WGS84 和 Web Mercator 坐标。

## 4. 查看真实 Episode

设置数据路径：

```bash
export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
export SATNAV_EPISODES_PATH="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
```

复制示例 task 配置：

```bash
mkdir -p .local
cp applications/resources/satnav_example_task.yaml \
  .local/satsim_viewer_task.yaml
```

将本地配置中的 `DATASET` 修改为：

```yaml
DATASET:
  TYPE: SatNav
  SPLIT: train
  DATA_PATH: ${oc.env:SATNAV_EPISODES_PATH}
  SCENES_DIR: ${oc.env:SATNAV_SCENES_DIR}
```

然后运行：

```bash
python -m applications.satsim_viewer task \
  --config .local/satsim_viewer_task.yaml
```

![真实 Episode 示例：Amsterdam-1 场景中的 RGB observation、reference path 和 waypoint](../../assets/viewer/real_episode_viewer.png)

<p class="figure-caption" align="center"><em>真实 Episode 示例（Amsterdam-1，Episode 2144）：左侧为当前 RGB observation，右侧为完整 reference path 和 waypoint，底部为导航指令及距离。</em></p>

场景文件需使用 `<scene_id>.tif` 命名。task viewer 会显示当前指令、距离、RGB observation 和 top-down map；执行 `STOP` 后显示 Success、SPL、Distance to Goal 和 Path Length，并加载下一个 Episode。

## 5. 快捷键

### Free viewer

| 按键 | 操作 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左转 / 右转 |
| `Q` / `E` | 升高 / 降低相机高度 |
| `P` | 保存当前画面 |
| `Esc` | 退出 |

### Task viewer

| 按键 | 操作 |
| --- | --- |
| `W` | 前进 |
| `A` / `D` | 左转 / 右转 |
| `T` | 显示或隐藏 top-down map |
| `Space` | 执行 `STOP` 并显示当前 Episode 指标 |
| `Esc` | 退出 |

> 键盘操作作用于 OpenCV 窗口；如果按键无响应，请先点击窗口使其获得焦点。

## 6. 截图输出

free viewer 中按 `P` 会将带有坐标和状态信息的 PNG 保存到仓库的 `output/` 目录，文件名包含时间、经纬度、高度和朝向。`output/` 已被 Git 忽略。

## 7. 常见问题

### 为什么 OpenCV 无法打开窗口？

确认当前机器具有图形桌面。通过 SSH 使用 Viewer 时，需要正确配置 X11 forwarding 和
`DISPLAY`；无图形界面的计算节点不能直接显示 OpenCV 窗口。

### 为什么出现 `TIF file not found`？

检查 Viewer 配置中的 GeoTIFF 路径是否存在。本机路径应写入 local config 或通过 CLI
提供，不要写入公共配置。

### 为什么 agent 无法继续移动？

当前相机视野可能已经接近场景边界。改变前进方向或降低相机高度，确保旋转后的完整视野
仍位于 GeoTIFF 覆盖范围内。

### 为什么初始画面超出场景边界？

减小 `AGENT.ALTITUDE`，或使用覆盖范围更大的 GeoTIFF。高度越高，相机需要的地面覆盖
范围越大。

### 为什么 Task viewer 找不到 Episode 对应的场景？

确认 `SCENES_DIR` 指向 GeoTIFF 目录，并且场景文件名与 Episode 的 logical `scene_id`
一致，例如 `Amsterdam-1.tif` 对应 `scene_id: Amsterdam-1`。

> SatSim 使用 WGS84 经纬度描述 agent state，并从 EPSG:3857 GeoTIFF 渲染 observation。

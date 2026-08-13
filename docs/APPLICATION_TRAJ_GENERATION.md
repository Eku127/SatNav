# SatNav 轨迹数据生成

本文介绍如何使用 `applications/trajectory_generation` 生成 VLM-based VLN 方法所需的离线训练数据。生成器按照每个 Episode 的 `reference_path` 驱动 SatSim，并保存 RGB 帧、动作序列和指令标注。

## 1. 准备环境

开始前请完成[环境安装](INSTALLATION.md)、[Episode 数据下载](DATA_DOWNLOAD.md)和[卫星场景下载](APPLICATION_MAP_DOWNLOAD.md)，然后在 SatNav 仓库根目录执行：

```bash
python -m pip install -e '.[applications]'

export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_TRAIN_EPISODES_PATH="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
export SATNAV_TRAJECTORY_DIR="$SATNAV_DATA_ROOT/trajectory_data"
```

> 完整 train split 当前生成的 trajectory 数据约占 233 GB，开始生成前建议至少预留 250 GB 可用磁盘空间。

先确认 Episode 和 59 个 GeoTIFF 场景均已就绪：

```bash
bash scripts/validation/data_validation.sh
```

## 2. 快速测试

仓库提供了两个示例 Episode 和一张合成 GeoTIFF，可在不下载真实数据的情况下测试生成流程：

```bash
python -m applications.trajectory_generation.generate \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/trajectory_generation_test
```

成功后终端会显示 `Trajectory generation completed successfully`，并在输出目录生成 `annotations.json`、`summary.json` 和 RGB 图像。

## 3. 生产配置

SatNav-v0.1 使用以下配置生成轨迹数据：

```text
applications/episode_processing/configs/trajectory_generation.yaml
```

主要参数包括 10 米前进步长、15° 转向角、448 × 448 RGB、90° HFOV 和最多 500 个 step。不同轨迹类型的 waypoint 到达半径为：

| Trajectory type | 到达半径 |
| --- | ---: |
| Boundary | 10 m |
| LandmarkSet | 3 m |
| Road | 10 m |

> 这里的 3 米是生成 expert trajectory 时的 waypoint 到达半径，用于更精确地判断 LandmarkSet waypoint 是否到达，从而生成更准确的训练数据。

> 为保证数据可复现，建议不要修改生产配置。

## 4. 生成完整训练数据

完整 train split 使用并行入口：

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir "$SATNAV_TRAJECTORY_DIR"
```

生成器默认根据场景数量和 CPU 核数选择 worker，并按场景分组以降低 GeoTIFF 内存占用。也可以通过 `--num_workers N` 手动设置；内存不足时应减小该值。

> 推荐使用并行入口生成完整数据，其效率明显高于串行入口。

## 5. 断点续跑

生成过程中断后，直接重新执行同一条并行命令即可续跑，无需添加额外参数。

> 断点续跑必须使用与首次生成相同的输出目录。

## 6. 输出结构

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <scene>_satnav_<episode-index>/
        ├── .done
        ├── .annotation.json
        └── rgb/
            ├── 001.jpg
            └── ...
```

- `annotations.json`：用于模型训练的最终标注。
- `summary.json`：JSONL 格式的逐 Episode 摘要。
- `images/`：与动作序列逐帧对齐的 RGB 图像及续跑缓存。

> `.done` 和 `.annotation.json` 用于断点续跑，不建议手动修改。

## 7. 确认生成完成

并行命令结束时会输出处理统计。完整的 SatNav-v0.1 train split 应显示：

```text
Success (incl. cached): 105164
Discarded (max steps): 0
Failed: 0
Generated annotations: 105164 / 105164 episodes
```

如果数量不足，可以先重跑同一命令以重试未完成 Episode；若问题持续存在，请根据终端中的 `failure reason` 检查场景文件和生产配置。

## 8. 常见问题

### 为什么运行生成命令时出现 `ModuleNotFoundError`？

在 SatNav 仓库根目录为当前环境安装 applications 依赖：

```bash
python -m pip install -e '.[applications]'
```

### 为什么并行生成占用过多内存？

使用 `--num_workers N` 减少并行 worker 数量。每个 worker 都会持有 simulator、场景缓存和
图像编码资源，因此 worker 数不应超过机器内存和本地磁盘吞吐能够支持的范围。

### 为什么 trajectory 生成速度较慢？

确认 Episode、GeoTIFF 和输出目录位于本地高速磁盘，并保留默认的 scene affinity，使同一
worker 尽量连续处理相同场景，减少 GeoTIFF 切换和缓存重建。

### 为什么找不到 Episode 或场景？

重新设置 `SATNAV_TRAIN_EPISODES_PATH` 和 `SATNAV_SCENES_DIR`，然后运行：

```bash
bash scripts/validation/data_validation.sh
```

确认每个 Episode 的 logical `scene_id` 都能在场景目录中解析为对应 GeoTIFF。

### 为什么重跑后部分 Episode 被重新生成？

对应 Episode、生成配置或 GeoTIFF 内容已经变化，因此已有缓存不能继续复用。检查本地路径
是否指向同一批输入；如果输入确实发生变化，应保留本次重新生成的结果。

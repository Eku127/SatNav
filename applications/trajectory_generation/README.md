# SatNav Trajectory Generation

从 SatNav VLN episodes 生成 StreamVLN 训练数据。

## 功能说明

该工具遍历 SatNav 数据集中的所有 episodes，使用路径跟随器沿 `reference_path` 导航，并保存：

- RGB 图像序列（JPG 格式）
- 动作序列标注（StreamVLN 格式）

生成的数据可直接用于训练 StreamVLN 模型。

## 使用方法

### 串行版本

```bash
python -m applications.trajectory_generation.generate \
    --config configs/satnav_task.yaml \
    --output_dir /path/to/output
```

### AerialSim episode 级先筛后产

AerialSim 当前保留为可选 3D 渲染和数据生产能力，不是 SatNav/SwiftVLN 训练评测主链的默认路径。当你希望快速筛选 3D 资产更丰富、渲染更稳定的代表 episodes 时，可以先做 episode 级预筛选：

```bash
python -m applications.trajectory_generation.generate \
    --config configs/satnav_task.yaml \
    --output_dir output/aerialsim_prod \
    --preselect_aerial \
    --preselect_candidates 300 \
    --preselect_samples 4 \
    --preselect_topk 80 \
    --preselect_min_avg_std 35 \
    --preselect_max_black_frac 0.01 \
    --preselect_min_height_range 3.0
```

只做预筛选（不生成轨迹）：

```bash
python -m applications.trajectory_generation.generate \
    --config configs/satnav_task.yaml \
    --output_dir output/aerialsim_prod \
    --preselect_aerial \
    --preselect_only
```

### AerialSim trajectory group 级预筛选

当你已有 trajectory group 候选列表，并希望按代表 episode 的 AerialSim 质量推荐整组 trajectory 时，使用 group 级预筛入口：

```bash
python -m applications.trajectory_generation.preselect_aerial_groups \
    --config configs/satnav_task.yaml \
    --preselect_json output/preselect_0404_train_trajectory/preselect_trajectory_list.json \
    --output_dir output/aerialsim_train_traj_quality_eval_full \
    --samples_per_episode 2
```

该入口支持断点续跑，并输出：

- `recommended_top.json`
- `evaluated_results.json`
- `failed.json`
- `summary.json`

### AerialSim 推荐轨迹生产 Pipeline

当你已经有一份经过 AerialSim 筛选的 trajectory group 列表，希望把整条 trajectory 对应的全部 episode 稳定产出时，不建议再用一个超长进程直接全量生成。仓库现在提供专门的生产编排脚本：

```bash
python -m applications.trajectory_generation.produce_aerialsim_recommended \
    --config configs/satnav_task.yaml \
    --recommended_json output/aerialsim_train_traj_quality_eval_full/recommended_top.json \
    --trajectory_groups_json output/preselect_0404_train_trajectory/trajectory_groups_full.json \
    --output_dir output/aerialsim_recommended_prod \
    --runtime_root /mnt/data3/.../aerialsim_recommended_runtime \
    --episodes_per_batch 1 \
    --trajectories_per_batch 3 \
    --max_rounds 4 \
    --max_episode_attempts 4
```

这条 pipeline 会做以下事情：

- 将 `recommended` 里的 trajectory group 展开为完整 episode 集合（通常每条 trajectory 对应 3 个 episode）
- 每次只生成一个小批次，批次之间强制 fresh process / fresh browser，避免长生命周期 Chrome tab crash 污染整轮生产
- 每个 episode 默认独立进程生成；脚本会把 AerialSim HTML runtime、Chrome user-data、disk-cache、crash-dumps 和 batch 级 `TMPDIR` 全部迁到 `data3` 上的短路径（默认 `/mnt/data3/jiangjiajun/tmp/sa/<hash>/{r,t}`），并在每个 batch 结束后自动清理，避免把系统根分区 `/` 挤满或因路径过长导致 Chrome 启动失败
- 每个批次结束后，根据 `summary.json` 和磁盘上的 JPG 序列双重校验 episode 是否真正可用，同时自动清理该批次 runtime 临时目录
- 对仍不完整的 episode 做定向清理后再重试，而不是盲目全量重跑
- 最终输出 `production_manifest.json`，按 trajectory 和 episode 两个粒度给出 `completed / pending / failed` 状态

如果你只想用串行生成入口跑一个明确的 episode 子集，也可以直接传：

```bash
python -m applications.trajectory_generation.generate \
    --config configs/satnav_task.yaml \
    --output_dir output/aerialsim_recommended_prod \
    --episode_indices_file output/aerialsim_train_traj_quality_eval_full/recommended_top.json
```

### 并行版本（推荐）

使用多进程并行生成，速度显著提升：

```bash
python -m applications.trajectory_generation.generate_parallel \
    --config configs/satnav_task.yaml \
    --output_dir /path/to/output \
    --num_workers 64
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--config` | SatNav 任务配置文件路径（YAML 格式） |
| `--output_dir` | 输出目录，用于保存生成的数据 |
| `--num_workers` | （仅并行版本）工作进程数，默认为 min(CPU核心数, 64) |
| `--preselect_aerial` | （仅串行版本）启用 AerialSim 预筛选后再生产 |
| `--preselect_candidates` | 预筛选候选 episode 数量 |
| `--preselect_samples` | 每个候选 episode 采样帧数 |
| `--preselect_topk` | 阈值过滤后最多保留多少个 episode |
| `--preselect_min_avg_std` | 平均纹理方差下限（越高越倾向 3D 细节丰富） |
| `--preselect_max_black_frac` | 平均黑像素比例上限（过滤空白/异常渲染） |
| `--preselect_min_height_range` | 局部高度起伏下限（越高越倾向高楼场景） |
| `--preselect_only` | 只输出预筛选结果，不执行轨迹生成 |
| `--episode_indices_file` | 指定只生成哪些 episode（JSON/JSONL/逗号分隔文本） |

**性能对比：**

| 版本 | 3808 episodes 耗时 | 速度 |
|------|-------------------|------|
| 串行 | ~4 小时 | ~0.3 episode/s |
| 并行 (64 workers) | **~2 分钟** | ~28 episode/s |

## 输出格式

```
output_dir/
├── images/
│   ├── MN2_satnav_000000/
│   │   └── rgb/
│   │       ├── 001.jpg
│   │       ├── 002.jpg
│   │       └── ...
│   └── MN2_satnav_000001/
│       └── rgb/
│           └── ...
├── annotations.json    # 完整 JSON 格式
└── summary.json        # JSONL 格式（逐行追加）
```

### annotations.json

完整的 JSON 数组，包含所有 episodes 的标注：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | Episode 索引（从 0 开始） |
| trajectory_id | string | 轨迹 ID |
| steps | int | 实际导航步数（不含初始占位符） |
| video | string | 图像目录相对路径 |
| instructions | array | 导航指令（数组格式） |
| actions | array | 动作序列（StreamVLN 编码） |

### summary.json

JSONL 格式（每行一个 JSON），在生成过程中逐行追加，包含额外的元数据（trajectory_id, scene_id, episode_id）。

## 动作编码

| 编码 | 动作 | 说明 |
|------|------|------|
| -1 | 初始状态 | 占位符，表示第一张图是初始观察 |
| 0 | STOP | 停止导航 |
| 1 | MOVE_FORWARD | 向前移动 |
| 2 | TURN_LEFT | 向左转 |
| 3 | TURN_RIGHT | 向右转 |

## 图像与动作的对应关系

**核心规则**：看到第 N 张图像，应该执行 `actions[N]` 这个动作。

| 图像 | 看到后应执行 | 说明 |
|------|-------------|------|
| 001.jpg | actions[1] | 初始观察，actions[0]=-1 是占位符 |
| 002.jpg | actions[2] | 执行 actions[1] 后的观察 |
| ... | ... | ... |
| 069.jpg | actions[69]=STOP | 执行 actions[68] 后的观察 |
| 070.jpg | - | STOP 后的最终画面，不参与预测 |

**示例**：70 张图像对应 70 个动作
- actions[0] = -1（占位符，无实际意义）
- actions[1] ~ actions[68]：实际导航动作
- actions[69] = 0（STOP，结束导航）

## 特性

- **断点续传**：自动跳过已生成的 episodes，支持中断后继续生成
- **进度显示**：使用 tqdm 显示生成进度
- **路径跟随**：使用 `SatNavPathFollower` 沿 `reference_path` 导航
- **Aerial 预筛选**：（串行版本）支持按多帧质量 + 局部高度起伏评分，优先生产 3D 资产更丰富的 episode
- **多进程并行**：（并行版本）支持多核 CPU 并行处理，大幅提升生成速度

## 配置要求

配置文件需包含以下内容：

- `DATASET.DATA_PATH`：数据集 JSON 文件路径
- `DATASET.SCENES_DIR`：场景目录路径
- `TASK.SUCCESS_DISTANCE`：到达 waypoint 的判定半径
- `SIMULATOR.TURN_ANGLE`：转向角度

## 注意事项

1. 确保 conda 环境为 `satnav`
2. 确保配置文件中的数据路径正确
3. 输出目录会自动创建
4. 并行版本建议 `--num_workers` 设置为 CPU 核心数的 50%-75%
5. 每个 worker 进程会加载独立的环境实例，注意内存使用

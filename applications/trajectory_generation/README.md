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

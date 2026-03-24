# Codex Repo Context (SatNav)

本文件是 SatNav 仓库内的默认工作上下文。
在本仓库执行任务时，优先以本文件为准。

## Repository Scope

- Repo: `SatNav`
- Root: `/mnt/data1/home/jiangjiajun/workspace/SatNav`
- 主要代码域：`satnav/*`
- 定位：连续状态 VLN 评测平台，使用卫星/航拍地图作为场景

## Current Architecture Snapshot

SatNav 是一个独立的 Python 包，核心结构如下：

- 包根：`satnav/`
- 仿真器：`satnav/sims/`（`SatSim` 2D 卫星图、`AerialSim` 3D Google Tiles）
- 核心逻辑：`satnav/core/`
- 数据集接口：`satnav/dataset/`
- 导航逻辑：`satnav/navigation/`
- 任务定义：`satnav/task/`
- 训练接口：`satnav/training/`
- 模型：`satnav/models/`
- 工具函数：`satnav/utils/`
- 数据构建：`satnav/data_construction/`

## Key Directories & Entry Scripts

- 配置目录：`configs/`（`default.yaml`、`satnav_task.yaml`、`vln_task.yaml`、`configs/baselines/`）
- 脚本目录：`scripts/`
- 应用工具：`applications/`
  - 轨迹生成：`applications/trajectory_generation/generate.py`（串行）、`generate_parallel.py`（并行）
  - 航拍查看器：`applications/aerial_viewer/`
- 实验分析：`experiments/`
- 使用示例：`examples/`
- 测试：`tests/`（pytest，配置见 `pytest.ini`）
- 主入口：`run.py`

## Current SatNav Dataset Defaults

- Dataset root: `/mnt/data3/jiangjiajun/dataset/satnav_datasets`
- 当前常用版本：`ver_260317`
- Eval episodes (val_seen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/eval/val_seen/all_episodes.json`
- Eval episodes (val_unseen):
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/eval/val_unseen/all_episodes.json`（当前无城市，目录暂不存在，后续引入新城市后生成）
- **注意**：`episodes/eval/` 下只有 `val_seen/` 和 `val_unseen/` 子目录，不再有顶层扁平文件
- QA JSONL:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/data/qa_swift.jsonl`
- Trajectory data:
  `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data`
- Scene maps:
  - active: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`
  - backup(old): `/mnt/data3/jiangjiajun/dataset/satnav_datasets/old_scenes`

### SatNav Data Processing Convention (Updated: 2026-03-14)

- 数据处理默认会先执行 trajectory type 标准化：
  - 删除不完整城市目录 `Venezia`
  - 将 `highway / multiway / multway / waterway` 统一映射为 `trajectory_type = Road`
  - 同时保留细分类到顶层字段 `trajectory_subtype`，规范值为 `Highway / Multiway / Waterway`
- 当前默认城市划分（0316 起）：
  - eval: `Amsterdam-1`, `Rome-1`, `NewYork-1`
  - train: 其余全部城市
- Eval 城市按 seen/unseen 自动分类（0319 起）：
  - **val_seen**：eval 城市的基础名（如 `Amsterdam`）在 train 中有任意 TIF → 当前全部 3 个 eval 城市均为 val_seen
  - **val_unseen**：eval 城市的基础名完全不出现于 train → 当前无，后续引入新城市时自动归入

## Runtime/Infra Conventions

### Server Settings

三台服务器共享同一挂载工作区，所有文件操作（脚本、配置、输出）在任意服务器上本地可见。

| Server | Host | Access | GPUs | Notes |
|---|---|---|---|---|
| **98** | localhost | 直接执行 | 8× | 本机，无需 SSH |
| **73** | `10.246.152.73` | `ssh 10.246.152.73` | 8× | 远程 SSH |
| **17** | `10.246.132.17` | `ssh 10.246.132.17` 后进入 Docker 容器 | 8× | 远程 SSH + Docker |

- 共享工作区挂载点：`/mnt/data1/home/jiangjiajun/workspace/SatNav`
- **SSH 仅用于**：远端 GPU/进程状态检查、远程 tmux 启动/停止。
- **文件操作**（脚本、配置写入、输出读写）：始终是本地操作，无需 SSH。
- server 17 Docker 容器名查询：`ssh 10.246.132.17 "docker ps"`

### Conda Environments

| Env | Purpose |
|---|---|
| `satnav` | SatNav 平台开发与运行（仿真器、数据处理、评测） |

Conda 初始化命令（所有服务器统一）：
```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
```

## Execution Rules

- 优先使用 `configs/` 下现有配置文件，通过 `run.py` 启动任务，避免硬编码参数。
- 修改仿真器或任务参数时，优先编辑 `configs/satnav_task.yaml` 或 `configs/vln_task.yaml`。
- 新增 baseline 配置请放入 `configs/baselines/` 目录。
- 测试用例放入 `tests/`，并通过 `pytest` 运行。

## Seq2Seq Eval Infrastructure (Updated: 2026-03-24)

- 评测入口脚本：`scripts/seq2seq/eval.sh`
- 评测专用配置：`configs/baselines/seq2seq_eval.yaml`
- 调用方式：
  ```bash
  # 按实验名评测（推荐）
  bash scripts/seq2seq/eval.sh <exp_name> [split] [max_episodes]
  # 例：
  bash scripts/seq2seq/eval.sh seq2seq-offline-ddp-g8-bs64-lr3e-4-20260320-164653 val_seen
  bash scripts/seq2seq/eval.sh seq2seq-offline-ddp-g8-bs64-lr3e-4-20260320-164653 val_seen 20

  # 直接指定 checkpoint 路径
  bash scripts/seq2seq/eval.sh /path/to/best.pth val_seen
  ```
- 默认 split：`val_seen`（`val_unseen` 当前暂无 episodes）
- 输出约定：`output/seq2seq_offline/results/<exp_name>/<split>/eval_ckpt_0_<split>.json`
- 依赖 `satnav` conda 环境（脚本内自动 activate）
- 使用 `offline_trainer.eval()` 入口：
  1. `OfflineTrajectoryDataset` 提供 obs/action space 供 policy 初始化
  2. `SatNavDataset` + `Env`（仿真器 online rollout）运行完整 episode
  3. `Evaluator` 记录 spl / success / distance_to_goal / path_length
- `DATA_PATH` 使用 `{split}` 占位符，自动展开 eval split 路径
- 评测需要 SCENES_DIR（卫星 TIF 图）；不支持无仿真器的离线评测

## Offline Training (Updated: 2026-03-24)

- 已新增离线训练入口：`TRAINER_NAME=offline_trainer`
- baseline 脚本已按模型拆分：
  - `scripts/seq2seq/*`
  - `scripts/cma/*`
  - 根目录旧脚本保留为兼容 wrapper，不再作为首选入口
- 核心文件：
  - `satnav/dataset/offline_trajectory_dataset.py`
  - `satnav/training/offline_trainer.py`
  - `configs/baselines/seq2seq_offline.yaml`
  - `configs/baselines/seq2seq_offline_smoke.yaml`
  - `scripts/seq2seq/make_offline_smoke_subset.py`
  - `scripts/seq2seq/train_offline.sh`
  - `scripts/seq2seq/train_offline_ddp.sh`
  - `scripts/seq2seq/organize_output.sh`
  - `scripts/cma/train.sh`
  - `scripts/cma/train_ddp.sh`
- 离线训练默认数据源：
  - annotations: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json`
  - images: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/images`
  - vocab: `/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/artifacts/vocab_260317/train_vocab.json`
  - embeddings: `/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/artifacts/embeddings_260317/embeddings_glove50d.json.gz`
- 关键语义约定：
  - 训练对齐 `RecollectTrainer`，**不训练最终 STOP**
  - 轨迹末尾 post-STOP 帧与前一帧重复，离线训练时丢弃
  - 离线图像默认 resize 到 `224x224`
  - `instruction` 在离线 dataset 中按时间维重复，保持 `collate_fn` 兼容
- 冒烟测试默认输出：
  - subset: `output/seq2seq_offline/artifacts/smoke_260317/offline_annotations_128.json`
  - checkpoint: `output/seq2seq_offline/checkpoints/seq2seq_offline_smoke_260317/best.pth`
- 8xH100 当前离线 seq2seq 推荐起点：
  - `IL.batch_size=8`（per GPU，global batch 64）
  - `IL.lr=3e-4`
  - `IL.OFFLINE.num_workers=8`
- 训练监控：
  - 默认推荐脚本：`scripts/seq2seq/train_offline_ddp.sh`
  - 默认输出根目录：`output/seq2seq_offline`
  - 运行产物统一写入：
    - logs: `output/seq2seq_offline/logs`
    - checkpoints: `output/seq2seq_offline/checkpoints/<EXP_NAME>`
    - results: `output/seq2seq_offline/results/<EXP_NAME>`
    - videos: `output/seq2seq_offline/videos/<EXP_NAME>`
    - swanlab: `output/seq2seq_offline/swanlab`
    - legacy: `output/seq2seq_offline/legacy`
    - artifacts: `output/seq2seq_offline/artifacts/{vocab_260317, embeddings_260317, smoke_260317}`
  - 历史顶层 `output/checkpoints`、`output/results`、`output/videos`、`output/logs`、`output/swanlab*` 如需收敛到新结构，使用 `scripts/seq2seq/organize_output.sh`
## CMA Baseline (Updated: 2026-03-24)

- 默认配置：`configs/baselines/cma.yaml`
- 默认脚本：
  - `scripts/cma/train.sh`
  - `scripts/cma/train_ddp.sh`
- 默认输出根目录：`output/cma`
- 运行产物默认写入：
  - checkpoints: `output/cma/checkpoints/<EXP_NAME>`（脚本模式）
  - results: `output/cma/results/<EXP_NAME>`（脚本模式）
  - videos: `output/cma/videos/<EXP_NAME>`（脚本模式）
  - logs: `output/cma/logs/<EXP_NAME>.log`
  - swanlab: `output/cma/swanlab`
- 直接用 `run.py` 时，`configs/baselines/cma.yaml` 的默认路径为：
  - checkpoint: `output/cma/checkpoints/default`
  - results: `output/cma/results/default`
  - videos: `output/cma/videos/default`
  - 已支持 `SWANLAB.*` 与 `OUTPUT_ROOT` 配置；脚本默认通过 SwanLab `cloud` 模式写入上述目录树
  - Python 3.8 + `swanlab==0.7.13` 下，`WxWebhookCallback` 在当前环境有兼容性问题，基础监控可用，企业微信通知暂不默认开启

## Webhook

- **Codex Webhook（企业微信机器人）**：
  `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=48e434da-fb2d-453c-a180-c4041b4c7f1e`
- 凡需要发送 Codex webhook 通知时，使用上述地址（POST JSON，`msgtype: text`）。

## Commit Style

- 使用 conventional commit：`feat/fix/refactor/docs/test/perf/chore`
- commit message 优先简洁中文
- 保持原子提交，避免混入无关改动

## Operating Rules for Codex

- 优先使用现有脚本，不拼装临时一次性流程。
- 高风险步骤（远程/资源密集）需先说明预期和阻塞点。
- 除非用户明确要求，不创建 commit。
- 不回滚用户已有的无关改动。
- 默认失败策略：单项失败可继续后续队列项，并记录失败原因。

# SatNav Repo Refactor & Slimming Audit

日期：2026-05-01  
范围：`/mnt/data1/home/jiangjiajun/workspace/SatNav`

本文是后续 refactor 与仓库瘦身的初始盘点，不是删除执行清单。当前 worktree 已有大量未提交/未跟踪变更，本次只做结构观察与候选项标注，真正移动或删除文件前需要逐项确认。后续瘦身过程中已经处理的候选项可能不再存在，应以最新 git 状态为准。

## 1. 当前结论

SatNav 的核心 Python 包并不大，主要复杂度来自三类内容混在同一仓库根目录：

1. 核心平台代码：`satnav/`、`run.py`、`configs/`、`scripts/{seq2seq,cma}`、`tests/`。
2. 应用和数据生产工具：`applications/trajectory_generation`、`applications/aerial_viewer`、`applications/satsim_viewer`、`applications/map_downloader`、`applications/sat_drone_pair_generation`。
3. 本地运行产物与实验沉积：`output/`、`__pycache__/`、`.pytest_cache/`、`satnav.egg-info/`、临时备份文件。

磁盘占用的主要来源不是核心代码，而是本地 `output/`：

| 路径 | 当前大小 | 说明 |
|---|---:|---|
| `.` | 782M | 仓库目录总大小，不含外部数据集挂载 |
| `output/` | 756M | 本地实验输出、checkpoint、日志、评测诊断；已被 `.gitignore` 忽略 |
| `output/cma` | 274M | CMA checkpoint / result / swanlab |
| `output/seq2seq_offline` | 228M | Seq2Seq checkpoint / result / swanlab |
| `output/preselect_0404_train_trajectory` | 86M | AerialSim 预筛选/轨迹生产中间结果 |
| `tests/` | 8.3M | 主要由 `tests/test_data/map.tif` 约 8.1M 贡献 |
| `satnav/` | 2.2M | 核心包源码 |
| `applications/` | 2.5M | 应用与数据生产工具源码，不含外部数据 |

`output/` 当前包含两个大 checkpoint：

| 文件 | 大小 |
|---|---:|
| `output/cma/checkpoints/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab/best.pth` | 153M |
| `output/seq2seq_offline/checkpoints/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab/best.pth` | 115M |

## 2. 仓库入口与核心链路

### 2.1 统一入口

`run.py` 是训练、评测、推理的统一入口：

```text
python run.py --exp-config <config.yaml> --run-type {train|eval|inference} [opts...]
```

启动过程：

1. 读取实验配置。
2. 合并 `_base_` / `BASE_CONFIG_PATH`。
3. 合并 `BASE_TASK_CONFIG_PATH` 指向的任务配置。
4. 应用 CLI dotlist 覆盖。
5. 初始化分布式运行时。
6. 从 `satnav.training.registry` 获取 trainer。
7. 调用 `trainer.train()` / `trainer.eval()` / `trainer.inference()`。

当前显式注册的 trainer：

| 名称 | 文件 | 用途 |
|---|---|---|
| `offline_trainer` | `satnav/training/offline_trainer.py` | Seq2Seq / CMA 离线训练与在线评测 |
| `recollect_trainer` | `satnav/training/recollect_trainer.py` | 早期 recollection / teacher forcing 训练链路 |
| `dagger_trainer` | `satnav/training/dagger_trainer.py` | 在线 rollout + expert label + dataset aggregation |

### 2.2 Env / Task / Simulator 链路

主链路如下：

```text
run.py
  -> Trainer
    -> Env
      -> SatNavDataset
      -> VLNTask
        -> Simulator interface
          -> SatSimWrapper -> SatSim
          -> AerialSimWrapper -> AerialSim
```

关键文件：

| 层级 | 文件 |
|---|---|
| 环境 | `satnav/core/env.py` |
| Episode dataclass | `satnav/core/episode.py` |
| Simulator 抽象接口 | `satnav/core/simulator.py` |
| Simulator factory | `satnav/sims/__init__.py` |
| 2D SatSim | `satnav/sims/satsim/*`、`satnav/sims/satsim_wrapper.py` |
| 3D AerialSim | `satnav/sims/aerialsim/*`、`satnav/sims/aerialsim_wrapper.py` |
| VLN task | `satnav/task/vln_task.py` |
| Sensors / metrics | `satnav/task/sensors.py`、`satnav/task/measures.py` |

### 2.3 Model 链路

`satnav/models/__init__.py` 负责导入并注册 baseline 模型：

| 注册名 | 文件 |
|---|---|
| `seq2seq` | `satnav/models/baselines/seq2seq_policy.py` |
| `cma` | `satnav/models/baselines/cma_policy.py` |
| `random` | `satnav/models/baselines/random_agent.py` |
| `greedy` | `satnav/models/baselines/greedy_agent.py` |

共享组件：

| 模块 | 文件 |
|---|---|
| policy base | `satnav/models/base.py` |
| registry | `satnav/models/registry.py` |
| instruction encoder | `satnav/models/encoders/instruction_encoder.py` |
| visual encoder | `satnav/models/encoders/visual_encoder.py` |
| recurrent state encoder | `satnav/models/encoders/rnn_state_encoder.py` |

## 3. 目录盘点

### 3.1 核心代码

| 目录 | 角色 | 备注 |
|---|---|---|
| `satnav/core/` | Env、配置、episode、simulator 抽象 | 平台核心 |
| `satnav/dataset/` | 在线 episode dataset、离线轨迹 dataset、DAgger/Recollect dataset | 训练和评测共同依赖 |
| `satnav/navigation/` | reference path follower、离散 planner | expert / oracle 相关 |
| `satnav/sims/` | SatSim / AerialSim 适配和实现 | 运行时依赖差异大，是后续拆 optional deps 的重点 |
| `satnav/task/` | action、sensor、measure、VLN task | `measures.py` 约 984 行，是拆分候选 |
| `satnav/training/` | trainer、evaluator、distributed、collate 工具 | `base_il_trainer.py`、`evaluator.py` 较重 |
| `satnav/models/` | baseline 模型和 encoder | 结构基本清晰 |
| `satnav/utils/` | vocab、embedding、map/visualization 等工具 | `maps.py` 约 1195 行，是拆分候选 |

### 3.2 应用层

| 目录 | 角色 | 当前观察 |
|---|---|---|
| `applications/trajectory_generation/` | 基于 SatNav 环境生成轨迹和图像 | 与核心评测/训练关系密切 |
| `applications/aerial_viewer/` | AerialSim 预览/截图工具 | Selenium / 浏览器依赖重 |
| `applications/satsim_viewer/` | SatSim 交互查看器 | 开发调试工具 |
| `applications/map_downloader/` | 当前 map downloader，新结构含 `google_downloader/` 与 `mapbox_downloader/` | worktree 中处于重构中：旧文件被删除，新子包未跟踪 |
| `applications/map_downloader_old/` | 旧版 map downloader | 未跟踪，明显是瘦身/归档候选 |
| `applications/sat_drone_pair_generation/` | sat-drone pair 数据生产工具 | 与 SatNav 主评测平台边界较弱，但当前有测试覆盖 |

### 3.3 配置

已跟踪配置主要是稳定基线：

```text
configs/default.yaml
configs/debug_vln_task.yaml
configs/satnav_task.yaml
configs/vln_task.yaml
configs/baselines/{seq2seq_offline,seq2seq_eval,seq2seq_dagger,cma,cma_eval,...}.yaml
```

当前还存在一批未跟踪日期配置：

```text
configs/satnav_task_aerialsim_0404*.yaml
configs/satnav_task_satsim_0404*.yaml
configs/baselines/*_260404*.yaml
```

这些配置有保留价值，但不适合作为长期散落在顶层的无限增长模式。建议后续改成：

```text
configs/
  tasks/
    satsim.yaml
    aerialsim.yaml
    aerialsim_production.yaml
  baselines/
    seq2seq/
    cma/
  experiments/
    2026-04-04/
```

或者至少把日期实验配置统一放入 `configs/experiments/YYYY-MM-DD/`，避免污染主配置列表。

### 3.4 脚本

已跟踪脚本：

```text
scripts/seq2seq/{eval.sh,make_offline_smoke_subset.py,organize_output.sh,train_dagger.sh,train_offline_ddp.sh}
scripts/cma/{eval.sh,eval_parallel.sh,train.sh,train_ddp.sh}
```

当前未跟踪脚本：

```text
scripts/seq2seq/eval_parallel.sh
scripts/cma/select_best_by_rollout.sh
scripts/trajectory_generation/*
```

观察：

- `scripts/seq2seq` 与 `scripts/cma` 结构已经比较清楚。
- `scripts/trajectory_generation` 和 `applications/trajectory_generation` 之间职责有交叉：前者偏生产编排，后者偏库/CLI。
- 多个 shell 脚本硬编码 conda init：`/mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh`。适合后续抽成环境变量或公共 shell helper。

### 3.5 测试

当前 `tests/` 约 8.3M，测试数量较多，覆盖了 dataset、env、simulator、task、navigation、models、training、sat-drone pair generation 等。

重点观察：

- `tests/test_data/map.tif` 约 8.1M，是测试目录体积主因。
- `pytest.ini` 已配置 `testpaths = tests` 和 `slow` / `integration` marker。
- 当前没有看到把 AerialSim 浏览器依赖与普通单元测试强制隔离的系统性 marker 约束，后续拆 optional deps 时需要补充。

## 4. 当前瘦身候选

### 4.1 可直接清理的本地产物

这些内容已被 `.gitignore` 忽略，不应进入版本库。真正清理前只需要确认是否还要保留本地结果。

| 候选 | 原因 |
|---|---|
| `output/` | 756M，本地实验产物；应迁移到外部 artifact/result 目录或按需保留 |
| `__pycache__/`、`*/__pycache__/` | Python 缓存 |
| `.pytest_cache/` | pytest 缓存 |
| `satnav.egg-info/` | editable install 产物 |
| `.tmp/` | 临时目录 |
| `satnav/task/measures.py.backup` | 备份文件，已被 `.gitignore` 忽略 |
| `applications/map_downloader.zip` | 未跟踪压缩包 |

### 4.2 需要确认后归档/迁出的内容

这些不是简单垃圾，可能仍有业务价值，需要先确定 owner 和保留策略。

| 候选 | 风险/建议 |
|---|---|
| `applications/map_downloader_old/` | 未跟踪旧版 downloader。若新版 `applications/map_downloader/{google_downloader,mapbox_downloader}` 可替代，应迁出或删除 |
| `applications/sat_drone_pair_generation/` | 与 SatNav 主平台边界较弱。建议确认是否保留在本仓库，或拆成独立工具包/数据生产 repo |
| `experiments/` | 偏分析脚本，有硬编码 checkpoint/output 路径。建议归档到 `research/experiments` 或迁出 |
| `doc/` 中历史训练/对比文档 | 有参考价值，但需要分层：用户文档、开发文档、历史分析 |
| 日期配置 `*_260404.yaml`、`*_0404*.yaml` | 有复现实验价值，但应统一归到实验配置目录 |

### 4.3 值得拆分的代码文件

| 文件 | 当前规模 | 建议 |
|---|---:|---|
| `satnav/utils/maps.py` | 约 1195 行 | 拆成 map rendering、topdown map、video/visualization 工具 |
| `satnav/sims/aerialsim/aerialsim.py` | 约 1083 行 | 拆浏览器生命周期、quality check、Cesium runtime、capture retry |
| `satnav/task/measures.py` | 约 984 行 | 拆成 distance/success/spl/path/topdown_map 等 measure |
| `satnav/training/evaluator.py` | 约 763 行 | 拆 episode rollout、metrics aggregation、video output、result writing |
| `satnav/training/base_il_trainer.py` | 约 610 行 | 拆 checkpoint、loss、swanlab、policy init |
| `applications/trajectory_generation/runner.py` | 约 654 行 | 拆 runner core、image writer、annotation writer、quality/preselect hooks |

## 5. 架构风险与重构机会

### 5.1 核心包和应用工具边界不够硬

`applications/trajectory_generation` 依赖核心 SatNav 是合理的；但 `applications/sat_drone_pair_generation` 更像独立数据生产工具，目前与 SatNav 主 runtime 耦合弱。后续可以考虑：

- 保留：把它标记为 `applications/` 下的可选数据工具，并单独维护依赖和测试 marker。
- 迁出：移动到独立 repo 或 `tools/sat_drone_pair_generation`，SatNav 只保留产物格式说明。

### 5.2 AerialSim 依赖应变成 optional

`requirements.txt` 当前把 Selenium、webdriver-manager、swanlab、imageio 等全部作为默认依赖安装。对只跑 SatSim/离线训练的环境来说偏重。

建议后续迁移到 `pyproject.toml` extras：

```text
satnav                 # core + satsim + training minimal
satnav[aerial]          # selenium / browser deps
satnav[training]        # torch / torchvision / swanlab
satnav[dev]             # pytest / pytest-cov
satnav[apps]            # downloader/viewer/data production extras
```

### 5.3 配置继承方式已有基础，但目录组织需要收敛

`run.py` 已支持 `BASE_CONFIG_PATH` 与 `BASE_TASK_CONFIG_PATH`。下一步重点不是新配置机制，而是整理命名与目录：

- `configs/tasks/`：任务/仿真器配置。
- `configs/baselines/{seq2seq,cma}/`：模型稳定配置。
- `configs/experiments/YYYY-MM-DD/`：一次性实验和日期版本。
- `configs/local_*.yaml`：本机私有覆盖，继续忽略。

### 5.4 硬编码本机路径较多

当前配置和脚本里有多处 `/mnt/data1/...`、`/mnt/data3/...`。这符合当前集群现实，但会增加迁移和开源成本。

建议保留 `.codex/CODEX_CONTEXT.md` 中的本机事实来源，同时在可执行脚本里支持环境变量：

```text
SATNAV_DATA_ROOT=/mnt/data3/jiangjiajun/dataset/satnav_datasets
SATNAV_OUTPUT_ROOT=output
SATNAV_CONDA_INIT=/mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
SATNAV_CONDA_ENV=satnav
```

### 5.5 测试数据需要体积策略

`tests/test_data/map.tif` 是合理的集成测试 fixture，但占 8M。后续可选：

- 保留当前 TIF，换取真实 rasterio/pyproj 测试稳定性。
- 生成更小的 synthetic GeoTIFF fixture。
- 把慢/重测试标成 `integration`，普通 CI 默认只跑轻量集。

## 6. 建议的重构顺序

### Phase 0: 建立安全基线

目标：不改变行为，先让后续删除有边界。

1. 确认当前 dirty worktree 哪些属于用户正在做的改动。
2. 跑一遍轻量测试基线：`pytest -m "not integration and not slow"`。
3. 把 `output/`、cache、egg-info 等本地产物清理策略写成文档或脚本，但先不自动删除。
4. 为 AerialSim/browser 测试补 marker，避免普通测试被重依赖拖住。

### Phase 1: 仓库瘦身

目标：先减少本地噪音和目录混杂。

1. 清理 ignored cache：`__pycache__`、`.pytest_cache`、`satnav.egg-info`、`.tmp`。
2. 将 `output/` 中需要保留的结果迁到外部 artifact 目录，只保留 README/manifest。
3. 处理 `applications/map_downloader_old/` 和 `applications/map_downloader.zip`。
4. 归档日期实验配置到 `configs/experiments/`。

### Phase 2: 配置和脚本收敛

目标：减少重复配置与硬编码。

1. 拆 `configs/tasks/` 和 `configs/baselines/{seq2seq,cma}/`。
2. 抽公共 shell helper：conda activate、repo root、GPU list、output root。
3. 统一 seq2seq/cma eval_parallel 参数和输出约定。
4. 把 `/mnt/data*` 路径降级为默认值，可由环境变量覆盖。

### Phase 3: 模块拆分

目标：降低核心文件复杂度，不改变公开接口。

1. 拆 `satnav/task/measures.py`。
2. 拆 `satnav/utils/maps.py`。
3. 拆 `satnav/sims/aerialsim/aerialsim.py`。
4. 拆 `satnav/training/evaluator.py`。
5. 拆 `applications/trajectory_generation/runner.py`。

### Phase 4: Optional deps 与 packaging

目标：让 SatSim/离线训练/AerialSim/应用工具可以按需安装。

1. 引入 `pyproject.toml`。
2. 保留 `setup.py` 兼容或迁移掉。
3. 将 `requirements.txt` 改为开发锁定/参考文件。
4. 拆 extras：`aerial`、`training`、`dev`、`apps`。

## 7. 推荐优先级

| 优先级 | 项目 | 理由 |
|---|---|---|
| P0 | 清理/迁移 `output/` | 最大体积来源，且已 ignored，对核心代码无行为影响 |
| P0 | 固定测试基线和 marker | 防止后续 refactor 没有回归保护 |
| P1 | 处理 `map_downloader_old` / zip / 新旧 downloader 共存 | 当前 worktree 明显处于迁移中，容易误用 |
| P1 | 收敛日期配置目录 | 配置数量已经开始膨胀 |
| P1 | 抽 shell 公共逻辑和环境变量 | 训练/评测脚本重复且硬编码路径多 |
| P2 | 拆 `measures.py`、`maps.py`、`aerialsim.py` | 复杂度高，但需要测试保护后再动 |
| P2 | optional dependencies | 改 packaging 影响安装路径，适合在代码边界清晰后做 |

## 8. 建议保留的核心边界

后续瘦身时建议先把仓库拆成以下概念层，而不是按文件大小盲删：

```text
satnav/                  # 平台核心包，必须保持可安装、可测试
configs/                 # 稳定配置 + 实验配置归档
scripts/                 # 训练/评测/生产编排入口
applications/            # 可选应用，需标注 owner 和依赖
tests/                   # 行为保护
doc/                     # 当前约定、设计、历史分析
output/                  # 本地 artifact，不属于 repo 源码资产
```

最先应该明确的是 `applications/` 的 owner 边界：

- `trajectory_generation`：SatNav 数据生产主链路，建议保留。
- `aerial_viewer` / `satsim_viewer`：调试工具，建议保留但 optional deps。
- `map_downloader`：如果继续服务 SatSim scene 获取，保留；旧版迁出。
- `sat_drone_pair_generation`：建议单独决策，可能更适合独立工具包。

## 9. 本次盘点用到的观察命令

主要命令：

```bash
git status --short
find . -maxdepth 2 -type d | sort
du -h --max-depth=2 . | sort -h | tail -80
rg --files
find . -type f -size +5M -printf '%s %p\n' | sort -nr | head
rg -n "TRAINER_NAME|BASE_TASK_CONFIG_PATH|SIMULATOR:|DATASET:" configs -g '*.yaml'
rg -n "TODO|FIXME|deprecated|legacy|backup|HACK|兼容|旧|临时" satnav -g '*.py'
git status --ignored --short
```

## 10. 下一步建议

建议下一步先做一个只涉及 ignored/local artifact 的瘦身 PR 或本地清理清单，不动核心代码：

1. 生成 `output/` 保留清单：哪些 checkpoint/result 需要迁走，哪些可删。
2. 清理 cache 和 egg-info。
3. 决策 `map_downloader_old`、`map_downloader.zip`、新 downloader 子包的最终形态。
4. 补一份 `doc/refactor/SLIMMING_PLAN.md` 或把本文转成 checklist。

这样可以先把噪音和体积降下来，再进入配置和模块级 refactor。

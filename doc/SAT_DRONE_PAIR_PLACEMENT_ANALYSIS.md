# Sat-Drone Pair 功能放置分析

## 背景

当前有一组用于从多个公开数据源构建 `sat-drone pair` 数据集的脚本，散落在
`/mnt/data1/home/jiangjiajun/workspace/SatDronePair` 目录下的不同 repo 中。

这些脚本虽然服务于不同数据源，但在职责上做的是同一类事情：

- 从原始数据集读取卫星图 / 无人机图 / 姿态或几何信息
- 通过各自既定逻辑构建对齐的 sat-drone image pair
- 输出统一风格的数据集产物，例如：
  - `dataset_info.json`
  - `pairs.csv`
  - `satellite/`
  - `drone/`

本次分析的约束是：

- 需要把这类功能整合起来，不再分散在不同 repo
- 现有 `build pair` 逻辑不改
- 需要判断这类功能应当放在 `SwiftVLN` 还是 `SatNav`

## 已定位的 pair 构建功能

### 1. DenseUAV

路径：

- `SatDronePair/DenseUAV/scripts/denseuav_pair_export/build_pairs.py`
- `SatDronePair/DenseUAV/scripts/denseuav_pair_export/sample_preview.py`

职责：

- 从 DenseUAV 原始数据中提取三个高度档的无人机-卫星图像对
- 执行 GPS 解析、朝向估算、NCC 校验、过滤、导出

文档说明：

- `SatDronePair/DenseUAV/scripts/denseuav_pair_export/README.md`

### 2. GTA-UAV

路径：

- `SatDronePair/GTA-UAV/scripts/gta_pair_export/build_pairs.py`
- `SatDronePair/GTA-UAV/scripts/gta_pair_export/sample_preview.py`

相关旧/辅助路径：

- `SatDronePair/GTA-UAV/scripts/prepare_dataset/extract_nadir_pairs.py`

职责：

- 从 GTA-UAV metadata 中筛选 near-nadir 样本
- 选择最佳正样本卫星瓦片
- 通过 heading post-check 做旋转确认
- 导出清洗后的 pair 数据集

文档说明：

- `SatDronePair/GTA-UAV/scripts/gta_pair_export/README.md`

### 4. SUES

路径：

- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/build_pairs.py`
- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/pipeline.py`
- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/center_recrop_pairs.py`
- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/sample_preview.py`
- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/merge_variants_dense_style.py`

职责：

- 对 SUES 数据执行候选遍历、旋转/尺度匹配、nadir 过滤
- 导出 pair，并可额外生成 recrop 变体和 preview

文档说明：

- `SatDronePair/SUES-200-Benchmark/scripts/sues_pair_export/README.md`

### 5. UAV-VisLoc

路径：

- `SatDronePair/UAV-VisLoc/scripts/uavvisloc_pair_export/build_pairs.py`
- `SatDronePair/UAV-VisLoc/scripts/uavvisloc_pair_export/export_selected.py`
- `SatDronePair/UAV-VisLoc/scripts/uavvisloc_pair_export/center_recrop_pairs.py`
- `SatDronePair/UAV-VisLoc/scripts/uavvisloc_pair_export/sample_preview.py`

职责：

- 从 UAV-VisLoc 原始数据导出 sat-drone pair
- 其中当前推荐主入口不是通用 `build_pairs.py`，而是 `export_selected.py`
- `export_selected.py` 固定使用当前确认过的高质量序列参数

文档说明：

- `SatDronePair/UAV-VisLoc/scripts/uavvisloc_pair_export/README.md`

## 这些功能的共同抽象

虽然各数据源内部算法差异很大，但抽象层面高度一致：

1. 输入是某种原始 sat/drone 数据源
2. 中间执行几何、姿态、尺度、筛选或后验校验
3. 输出统一风格的 pair 数据集
4. 通常还配套：
   - preview
   - recrop / variant
   - pipeline wrapper

因此，这更像是一个“多数据源 pair 数据产品构建应用族”，而不是模型训练逻辑。

## SatNav 和 SwiftVLN 的职责边界

### SatNav 的职责

从仓库定位和现有目录看，`SatNav` 的职责是：

- 连续状态 VLN 评测平台
- 管理卫星图和航拍图这两类观察域
- 提供仿真器、数据集接口、导航逻辑、任务定义
- 承载独立运行的数据和工具应用

当前已有明确的 `applications/` 目录，包括：

- `applications/trajectory_generation/`
- `applications/map_downloader/`
- `applications/satsim_viewer/`
- `applications/aerial_viewer/`

其中 `applications/trajectory_generation/` 的角色非常关键：

- 它就是一个独立的数据生成应用
- 它从 SatNav episode 生成下游训练所需数据
- 这说明 `SatNav/applications` 已经是“数据生产型应用”的既有落点

### SwiftVLN 的职责

`SwiftVLN` 的定位更明确，是训练/评测仓：

- 模型实现位于 `src/swiftvln/models/`
- 通用训练与评测组件位于 `src/swiftvln/common/`
- `src/swiftvln/scripts/` 主要承载：
  - train queue
  - eval queue
  - SatNav 数据版本处理
  - 训练相关分析或辅助脚本

目前 `SwiftVLN` 里的 `data_process` 主要处理的是：

- episode 生成
- QA 转换
- trajectory type 规范化

也就是说，`SwiftVLN` 中的数据处理脚本仍然紧贴其训练主流程，并没有承担一个“多外部数据源 pair 数据产品工厂”的职责。

## pair 功能与业务目标的关系

当前 `SwiftVLN` 训练出来的端到端模型主要消费卫星图。

而这组 pair 构建功能的目标是：

- 产出 sat-drone 图像对
- 为后续 domain adaptation / domain bridging 提供数据基础
- 帮助模型从纯 sat domain 向真实无人机俯视图 domain 过渡

因此，这组功能虽然最终服务于 `SwiftVLN` 模型能力提升，但它本身更靠上游：

- 它不是训练器的一部分
- 它不是模型定义的一部分
- 它是一个独立的数据资产构建应用

这类职责与 `SatNav` 更匹配，而不是 `SwiftVLN`

## 结论

推荐把这套功能统一整合到：

- `SatNav/applications/`

更具体地，建议新建：

- `SatNav/applications/sat_drone_pair/`

而不建议放到：

- `SwiftVLN/src/swiftvln/...`
- `SwiftVLN/src/swiftvln/scripts/data_process/...`

## 结论理由

### 1. 这是数据应用，不是模型主干

pair exporter 的本质是数据集生产应用。

`SwiftVLN/src` 是训练/评测产品代码主干，把多个外部数据源的 pair 构建器放进去，会把“数据生产应用”和“模型系统”混在一起。

### 2. SatNav 已经有 applications 作为同类落点

`SatNav/applications/trajectory_generation/` 已经说明：

- SatNav 接受“独立运行的数据生产应用”
- application 目录不是 demo-only，而是实际生产工具落点

pair builder 放在这里是顺着现有架构扩展，不是硬塞进去。

### 3. pair 的核心语义是观察域构建

pair 数据做的是：

- 卫星图
- 无人机俯视图
- 两种观察域之间的桥接

这与 `SatNav` 的双域语义更一致，因为 `SatNav` 本来就在处理卫星视角与航拍/真实视角相关问题。

### 4. SwiftVLN 应该消费 pair，而不是拥有 pair builder

从职责分层看，理想关系是：

- `SatNav` 负责生产数据资产
- `SwiftVLN` 负责消费这些资产做训练、适配、评测

这样边界更清晰，也更方便未来支持更多 pair 数据源。

## 推荐目录方案

建议目录：

```text
SatNav/
└── applications/
    └── sat_drone_pair/
        ├── README.md
        ├── __init__.py
        ├── registry.py
        ├── main.py
        ├── denseuav/
        │   ├── build_pairs.py
        │   └── sample_preview.py
        ├── gta_uav/
        │   ├── build_pairs.py
        │   └── sample_preview.py
        ├── ortholoc/
        │   ├── build_pairs.py
        │   ├── pipeline.py
        │   └── sample_preview.py
        ├── sues/
        │   ├── build_pairs.py
        │   ├── pipeline.py
        │   ├── center_recrop_pairs.py
        │   ├── sample_preview.py
        │   └── merge_variants_dense_style.py
        └── uavvisloc/
            ├── build_pairs.py
            ├── export_selected.py
            ├── center_recrop_pairs.py
            └── sample_preview.py
```

说明：

- 每个数据源保留原有脚本边界
- 不改 `build pair` 内部逻辑
- 只做目录收敛、统一入口和统一文档

## 不建议的放置方式

### 不建议放到 `SwiftVLN/src/swiftvln`

原因：

- 这会把独立数据构建应用并入训练包主干
- 会让 `swiftvln` 包职责膨胀
- 后续维护时，pair 构建与模型训练发布节奏会互相耦合

### 不建议只放到 `SwiftVLN/src/swiftvln/scripts`

原因：

- 虽然形式上是脚本，但这不是“训练辅助脚本”
- 它是一个长期存在、可扩展、多数据源的数据产品构建应用
- 放在 `scripts` 下会弱化其产品边界

## 大致迁移计划

### Phase 1. 梳理与冻结

- 列出所有当前主入口脚本
- 确认每个数据源的推荐运行命令
- 确认输入目录结构、输出结构、依赖环境
- 明确哪些脚本属于主流程，哪些属于 dev/调参辅助

### Phase 2. 在 SatNav 建立统一壳层

- 新建 `applications/sat_drone_pair/`
- 编写总 README，说明支持的数据源和统一调用方式
- 增加一个简单 registry 或 launcher

### Phase 3. 平移各数据源脚本

- 按数据源把现有 pair exporter 原样迁入
- 尽量保持文件名、CLI 参数、调用方式不变
- 不修改内部 build 逻辑

### Phase 4. 建立统一入口

例如提供类似：

```bash
python -m applications.sat_drone_pair.main denseuav ...
python -m applications.sat_drone_pair.main gta_uav ...
python -m applications.sat_drone_pair.main uavvisloc ...
```

这个入口只负责分发，不负责改写各数据源逻辑。

### Phase 5. 兼容旧路径

- 旧 repo 中保留短期 wrapper 或 README 跳转
- 保证已有命令在过渡期仍可定位到新入口

### Phase 6. 在 SwiftVLN 接消费链路

迁移完成后，`SwiftVLN` 只做下游接入，例如：

- dataset loader
- 训练配置
- domain adaptation 实验入口
- 结果分析

而不承担 pair 数据构建本身。

## 最终建议

最终建议可以概括为一句话：

> 把 `sat-drone pair` 功能视为“多数据源数据资产构建应用”，统一收敛到 `SatNav/applications/`，由 `SwiftVLN` 作为下游训练消费者接入，而不是把 pair builder 并入 `SwiftVLN/src`。


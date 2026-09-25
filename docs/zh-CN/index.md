# SatNav 文档

**[SatNav 论文](https://openreview.net/forum?id=hOEniyN6hl)已被 NeurIPS 2026 Evaluations & Datasets Track 录取，展示形式为 Poster。**

**基于卫星地图的连续状态视觉语言导航**

SatNav 将场景准备、轨迹生成、模型训练与在线评测整合到同一导航平台。核心环境 SatSim 根据智能体在连续地理空间中的位置与朝向，渲染对应的 RGB 观测。

![SatNav 导航任务与 Episode 示例](../assets/readme/overview-final.png)

## 从这里开始

| 你的目标 | 使用指南 |
| --- | --- |
| 理解系统原理 | [系统全景与运行闭环](concepts/OVERVIEW.md) · [SatSim 观测原理](concepts/SATSIM.md) · [任务与评测原理](concepts/TASKS_AND_METRICS.md) · [专家轨迹原理](concepts/EXPERT_TRAJECTORIES.md) |
| 运行第一个导航 Episode | [安装](getting-started/INSTALLATION.md) · [运行示例](getting-started/EXAMPLES.md) |
| 准备场景与 Episodes | [数据集](dataset/DATA_DOWNLOAD.md) · [卫星地图](applications/MAP_DOWNLOAD.md) |
| 训练导航策略 | [训练指南](training/README.md) |
| 评测模型 | [在线评测](evaluation/README.md) |

## 导航流程

![SatNav 训练与评测流程](../assets/readme/workflow.svg)

先生成用于训练的专家轨迹，再通过统一的 SatSim 观测与动作接口评测导航策略。

## 项目资源

- [SatNav Episodes](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1)：覆盖 59 个场景的 118,494 条 Episodes，包含边界、地标与路线导航任务。
- [Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)：已发布的 SatNav 基线模型权重。
- [SwiftVLN](https://github.com/Eku127/SwiftVLN)：SwiftVLN 在 SatNav 上的配置说明。

```{toctree}
:hidden:
:maxdepth: 2
:caption: 快速开始

安装 <getting-started/INSTALLATION>
运行示例 <getting-started/EXAMPLES>
使用流程 <README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 系统原理

系统全景与运行闭环 <concepts/OVERVIEW>
SatSim 观测原理 <concepts/SATSIM>
任务与评测原理 <concepts/TASKS_AND_METRICS>
专家轨迹原理 <concepts/EXPERT_TRAJECTORIES>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 数据集

下载 Episodes <dataset/DATA_DOWNLOAD>
数据格式 <dataset/DATASET_FORMAT>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: SatSim

准备卫星地图 <applications/MAP_DOWNLOAD>
可视化工具 <applications/VIEWER>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 训练

训练概览 <training/README>
专家轨迹生成 <applications/TRAJECTORY_GENERATION>
经典基线 <training/CLASSIC>
StreamVLN <training/vlm/STREAMVLN>
NaVILA <training/vlm/NAVILA>
Uni-NaVid <training/vlm/UNINAVID>
OpenFly <training/vlm/OPENFLY>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 评测

在线评测 <evaluation/README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 接口与开发

核心 API <core/CORE_API>
接入新模型 <development/MODEL_INTEGRATION>
```

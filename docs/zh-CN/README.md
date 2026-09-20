# SatNav 文档导航

本页是 SatNav 文档的统一入口。根据当前目标选择一条阅读路线即可，不需要按文件名顺序阅读
全部文档。

如果是第一次使用 SatNav，建议从[环境安装](getting-started/INSTALLATION.md)开始，并先使用仓库示例完成一次
环境交互。训练与正式评测需要额外准备 SatNav-v0.1 Episode、GeoTIFF 场景和模型资源。

## 系统原理

- [系统全景与运行闭环](concepts/OVERVIEW.md)
- [SatSim 观测原理](concepts/SATSIM.md)
- [任务与评测原理](concepts/TASKS_AND_METRICS.md)
- [专家轨迹原理](concepts/EXPERT_TRAJECTORIES.md)

## 1. 第一次运行 SatNav

按照以下顺序完成安装并运行最小示例：

1. [环境安装](getting-started/INSTALLATION.md)：安装 SatNav Core、可选应用或 classic baseline 依赖；
2. [SatNav 示例](getting-started/EXAMPLES.md)：使用仓库自带的合成场景和两个 Episode 完成第一次 rollout；
3. [Core API](core/CORE_API.md)：了解 `Env`、observation、action、Episode 和 metric 接口；
4. [数据格式](dataset/DATASET_FORMAT.md)：了解真实 Episode、GeoTIFF 和 trajectory 的组织方式。

仓库示例不需要下载 SatNav-v0.1、卫星影像或模型 checkpoint，适合检查安装和公开 API。

## 2. 准备 SatNav-v0.1 数据

训练模型或运行正式评测时，按照以下顺序准备数据：

1. [Episode 数据下载](dataset/DATA_DOWNLOAD.md)：下载 train、`val_seen`、`val_unseen` 和场景列表；
2. [卫星场景下载](applications/MAP_DOWNLOAD.md)：根据场景列表生成 SatSim 使用的 GeoTIFF；
3. [SatSim Viewer](applications/VIEWER.md)：检查场景、Episode 起点和 reference path；
4. [轨迹数据生成](applications/TRAJECTORY_GENERATION.md)：为离线训练生成 RGB frame 与 expert action；
5. [数据格式](dataset/DATASET_FORMAT.md)：检查 Episode 与 trajectory 字段是否符合公开格式。

只运行在线评测时需要 Episode、GeoTIFF 和模型 checkpoint，不需要提前生成离线 trajectory。
训练 Seq2Seq、CMA 或 VLM baseline 时需要 trajectory。

## 3. 训练模型

先阅读[模型训练](training/README.md)，了解通用训练流程、训练输入以及 Classic 与 VLM 两条路线。

### Classic baselines

[Classic Baselines](training/CLASSIC.md)介绍 Seq2Seq 和 CMA 的数据准备、vocabulary、训练、
checkpoint 检查以及单卡或多卡评测。第一次检查完整训练链路时，可以先运行 Training 文档中的
tiny example quickstart。

Random 和 ReferenceFollower 没有可学习参数，不需要训练。

### VLM baselines

每套 VLM 必须使用自己的 Python 环境和模型资源：

| Baseline | 训练与评测文档 |
| --- | --- |
| StreamVLN | [StreamVLN Baseline](training/vlm/STREAMVLN.md) |
| NaVILA | [NaVILA Baseline](training/vlm/NAVILA.md) |
| Uni-NaVid | [Uni-NaVid Baseline](training/vlm/UNINAVID.md) |
| OpenFly | [OpenFly Baseline](training/vlm/OPENFLY.md) |

SatNav 已训练 checkpoint 发布在
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)。

不要在不同 VLM 之间共用 PyTorch、Transformers 或 FlashAttention 环境。模型、数据、外部源码
和输出路径应写入对应 baseline 的 Git ignored `.local/env.sh`。

## 4. 评测模型

[统一评测](evaluation/README.md)介绍所有 Classic 与 VLM baseline 共用的在线 rollout、Episode
选择、多 rank 分片、输出格式、聚合、错误处理和 resume 行为。

推荐的评测顺序为：

1. 使用少量 Episode 和 5 步上限运行单 GPU smoke；
2. 使用相同 Episode 运行多 GPU smoke，确认分片与聚合正常；
3. 使用全部 Episode 和 500 步上限运行 `val_seen`；
4. 使用相同设置运行 `val_unseen`。

模型环境、checkpoint 和 launcher 参数仍以对应 baseline 文档为准。

## 5. 接入新模型

如果需要将仓库之外的模型接入 SatNav，建议依次阅读：

1. [Core API](core/CORE_API.md)：确认环境交互、observation 和 action contract；
2. [统一评测](evaluation/README.md)：了解 evaluator 负责的功能与输出格式；
3. [模型接入](development/MODEL_INTEGRATION.md)：实现 `PolicyAdapter`、独立环境、launcher 和本地配置；
4. 选择一个结构相近的 [VLM baseline](#vlm-baselines) 作为完整参考。

新模型应通过公开 `Env` 和 `PolicyAdapter` 接口运行，不应访问 SatNav 的私有对象。模型专用
状态、历史帧、tokenizer、processor 和 action queue 均由 adapter 管理。

## 6. 按目标查找文档

| 目标 | 文档 |
| --- | --- |
| 安装 SatNav | [环境安装](getting-started/INSTALLATION.md) |
| 运行仓库示例 | [SatNav 示例](getting-started/EXAMPLES.md) |
| 使用 Python API | [Core API](core/CORE_API.md) |
| 理解数据字段 | [数据格式](dataset/DATASET_FORMAT.md) |
| 下载 Episode | [Episode 数据下载](dataset/DATA_DOWNLOAD.md) |
| 准备 GeoTIFF | [卫星场景下载](applications/MAP_DOWNLOAD.md) |
| 检查场景和 Episode | [SatSim Viewer](applications/VIEWER.md) |
| 生成离线 trajectory | [轨迹数据生成](applications/TRAJECTORY_GENERATION.md) |
| 选择训练路线 | [模型训练](training/README.md) |
| 训练 Seq2Seq 或 CMA | [Classic Baselines](training/CLASSIC.md) |
| 训练或评测已有 VLM | [VLM baselines](#vlm-baselines) |
| 运行统一在线评测 | [统一评测](evaluation/README.md) |
| 接入新的模型 | [模型接入](development/MODEL_INTEGRATION.md) |

## 7. 文档约定

- 文档中的命令默认从 SatNav 仓库根目录运行；
- `/path/to/...` 表示需要替换的本机路径，不应原样执行或提交；
- 数据、模型、外部源码和输出路径保存在 `.local/env.sh` 或其他 Git ignored 配置中；
- Smoke 命令用于检查链路，不用于报告模型性能；
- 正式 benchmark 结果使用完整 split、500 步上限和对应 baseline 的完整 checkpoint；
- 每篇操作文档末尾的常见问题用于排查该流程特有的错误。

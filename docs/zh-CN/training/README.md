# SatNav 模型训练

本文介绍 SatNav 的通用训练流程，并使用仓库内置的 tiny example 完成第一次端到端训练。
Quickstart 会生成离线轨迹、训练 Seq2Seq 和 CMA、保存 checkpoint，并在 SatSim 中运行评测。

开始前建议先阅读[环境安装](../getting-started/INSTALLATION.md)和[数据格式](../dataset/DATASET_FORMAT.md)。如果只需要将
已有模型接入在线评测，请直接阅读[模型接入](../development/MODEL_INTEGRATION.md)。

## 1. 训练流程

SatNav 将训练和在线评测分为两个阶段：

```text
Episode + GeoTIFF
        │
        ▼
Trajectory Generation
        │
        ├── annotations.json
        └── images/<episode>/rgb/*.jpg
                    │
                    ▼
             Offline Training
                    │
                    ▼
                Checkpoint
                    │
                    ▼
            Online SatSim Evaluation
```

训练数据由 Episode、导航指令、expert action 和每一步 RGB observation 组成。模型训练时直接
读取已经生成的 JPEG 与 action，不需要在每个 batch 中运行 SatSim。训练完成后，checkpoint
通过在线环境逐步预测 action，并由 SatNav 计算 Success、SPL 和 Distance to Goal 等指标。

当前仓库维护两类可训练 baseline：

| 类型 | 模型 | 训练环境 | 训练入口 |
| --- | --- | --- | --- |
| Classic | Seq2Seq、CMA | SatNav classic 环境 | `run.py`、`scripts/seq2seq/`、`scripts/cma/` |
| VLM | StreamVLN、NaVILA、Uni-NaVid、OpenFly | 每个模型的独立环境 | `baselines/vlm/<name>/scripts/train.sh` |

Random 和 ReferenceFollower 不包含可学习参数，不需要训练。Classic 与 VLM 可以使用同一份
SatNav trajectory export，但各模型负责自己的采样、processor、优化器和 checkpoint 格式。
四个 VLM 的依赖互不兼容，选择模型后应进入对应的独立环境。

## 2. 选择训练路线

### 2.1 Classic baselines

Classic 适合第一次验证 SatNav 的数据、训练和在线评测链路，也可以作为新方法的轻量对照。
Seq2Seq 和 CMA 共用 SatNav classic 环境、trajectory 格式与统一评测接口：

| 模型 | 主要结构 | 训练起点 |
| --- | --- | --- |
| Seq2Seq | Instruction encoder + RGB encoder + recurrent policy | 随机初始化模型与本地 vocabulary/embedding |
| CMA | Cross-modal attention + recurrent policy | 随机初始化模型与本地 vocabulary/embedding |

第一次运行时，建议先完成本文第 4 至 6 节的 tiny example。它会生成示例 trajectory，并实际
训练和评测两个模型。准备完整 SatNav-v0.1 后，按照
[Classic Baselines](CLASSIC.md)完成数据校验、完整训练、checkpoint 检查以及单卡或
多卡评测。

Random 和 ReferenceFollower 只用于在线评测，不属于训练路线。

### 2.2 VLM baselines

VLM baseline 适合复现已有视觉语言模型，或在相同 SatNav trajectory 上进行大模型微调。每个
模型都有独立的 Python 环境、上游代码或运行时实现、模型资源与 launcher：

| Baseline | 模型代码 | 训练起点 | 完整流程 |
| --- | --- | --- | --- |
| StreamVLN | 固定版本的外部 StreamVLN checkout | 官方 StreamVLN checkpoint 或 LLaVA-Video base model | [StreamVLN Baseline](vlm/STREAMVLN.md) |
| NaVILA | 固定版本的外部 NaVILA checkout | NaVILA SFT 或 pretrain checkpoint | [NaVILA Baseline](vlm/NAVILA.md) |
| Uni-NaVid | 固定版本的外部 Uni-NaVid checkout | Uni-NaVid checkpoint 与 EVA 权重 | [Uni-NaVid Baseline](vlm/UNINAVID.md) |
| OpenFly | SatNav 中随附的固定 OpenFly runtime | 完整 HF checkpoint，或原生 `.pt` 与 processor | [OpenFly Baseline](vlm/OPENFLY.md) |

选择 VLM 后，直接从表格中的对应文档开始，不需要运行 Classic tiny example。各文档依次说明
环境安装、模型准备、本地路径、数据校验、Smoke 训练、完整训练和在线评测。

四个 VLM baseline 的 SatNav scratch 与 continue checkpoint 均已发布到
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)。

同一台机器可以准备多套 VLM 环境，但不要在环境之间复用 PyTorch、Transformers 或
FlashAttention。模型、数据集、上游 checkout 和输出路径应保存在对应 baseline 的 Git
ignored `.local/env.sh` 中。

## 3. 训练输入

一次完整训练通常需要以下输入：

| 输入 | 作用 | 准备方式 |
| --- | --- | --- |
| Episode JSON | 提供指令、起点、目标和 reference path | [Episode 数据下载](../dataset/DATA_DOWNLOAD.md) |
| GeoTIFF scenes | 渲染 expert trajectory 的 RGB observation | [卫星场景下载](../applications/MAP_DOWNLOAD.md) |
| Offline trajectory | 提供 `annotations.json`、RGB frame 和 action | [轨迹数据生成](../applications/TRAJECTORY_GENERATION.md) |
| 模型资源 | vocabulary、embedding、processor 或预训练权重 | 由对应 baseline 准备 |
| 训练配置 | 数据路径、模型结构和优化参数 | `configs/baselines/` 或 VLM 自有配置 |

离线 trajectory 的基本结构为：

```text
trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

`annotations.json` 保存 Episode instruction、action sequence、frame index 和 logical scene
信息。每条轨迹以初始 observation 开始，并以 `STOP` action 结束。字段定义参阅
[数据格式 - 离线 trajectory](../dataset/DATASET_FORMAT.md#6-离线-trajectory)。

训练配置中的 Episode、scene、trajectory、checkpoint 和输出路径必须指向同一组实验数据。
本机路径应放入 Git ignored 的 `.local/env.sh` 或 `configs/local_*.yaml`，不要写入公共配置。

## 4. Tiny Example

仓库提供一套可直接运行的训练资源：

```text
applications/resources/
├── map.tif
├── satnav_example_episodes.json
└── satnav_example_task.yaml
```

其中包含 2 个示例 Episode 和一张程序生成的合成 GeoTIFF。Quickstart 会根据这两个 Episode
生成离线 trajectory，再使用以下配置训练模型：

```text
configs/baselines/seq2seq_offline_train.yaml
configs/baselines/cma_offline_train.yaml
```

Tiny example 的作用是确认以下链路能够在当前环境中正常工作：

- vocabulary 和 instruction embedding 构建；
- SatSim trajectory 生成与 RGB 写入；
- offline dataset 加载；
- 前向传播、反向传播和 optimizer update；
- checkpoint 保存与重新加载；
- Seq2Seq、CMA 在线 rollout 和指标输出。

示例数据规模很小，只用于检查训练链路。其 loss 和导航指标不代表模型在
SatNav-v0.1 benchmark 上的性能。

## 5. 准备 Quickstart 环境

在 SatNav 仓库根目录创建并激活 classic 环境：

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic,applications]'
```

如果环境已经按照[安装指南](../getting-started/INSTALLATION.md)准备完成，不需要重复创建。使用其他 CUDA
版本时，请安装与本机驱动匹配的 PyTorch 和 TorchVision。

Quickstart 还需要：

- `curl` 或 `wget`：下载 GloVe；
- `unzip`：解压 `glove.6B.zip`；
- 可访问 TorchVision 权重下载地址的网络，或已经缓存 ResNet50 ImageNet 权重。

运行以下命令检查关键依赖：

```bash
python - <<'PY'
import torch
import torchvision
from satnav.core.env import Env
from satnav.training.offline_trainer import OfflineTrainer

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("SatNav training import OK")
PY

command -v unzip
command -v curl || command -v wget
```

## 6. 运行 Tiny Example Quickstart

在仓库根目录执行：

```bash
bash scripts/quickstart_models.sh
```

脚本依次完成：

1. 根据示例指令构建 vocabulary；
2. 下载或复用 GloVe 6B 50d；
3. 为示例 vocabulary 构建 GloVe embedding；
4. 下载或复用 TorchVision ResNet50 ImageNet 权重；
5. 使用合成 GeoTIFF 生成两个 Episode 的离线 trajectory；
6. 训练并评测 Seq2Seq；
7. 训练并评测 CMA。

默认输出位于：

```text
output/quickstart_baselines/
├── artifacts/
│   ├── vocab/
│   │   └── satnav_example_vocab.json
│   ├── glove/
│   └── embeddings/
│       └── satnav_example_glove50d.json.gz
├── trajectory_data/
│   ├── annotations.json
│   ├── summary.json
│   └── images/
├── seq2seq/
│   ├── checkpoints/latest/best.pth
│   └── results/latest/
└── cma/
    ├── checkpoints/latest/best.pth
    └── results/latest/
```

运行成功后，终端最后会显示：

```text
Done. Outputs are under output/quickstart_baselines/
```

可以进一步确认两个 checkpoint 和评测结果已经生成：

```bash
test -f output/quickstart_baselines/seq2seq/checkpoints/latest/best.pth
test -f output/quickstart_baselines/cma/checkpoints/latest/best.pth

find output/quickstart_baselines/seq2seq/results/latest -type f
find output/quickstart_baselines/cma/results/latest -type f
```

如果本机已经有 `glove.6B.50d.txt`，可以避免重复下载：

```bash
LOCAL_GLOVE_TXT=/path/to/glove.6B.50d.txt \
  bash scripts/quickstart_models.sh
```

只运行一个模型时，可以关闭另一项：

```bash
# 只运行 Seq2Seq
RUN_CMA=0 bash scripts/quickstart_models.sh

# 只运行 CMA
RUN_SEQ2SEQ=0 bash scripts/quickstart_models.sh
```

如只需检查数据准备和训练，不运行训练后的在线评测：

```bash
RUN_EVAL=0 bash scripts/quickstart_models.sh
```

## 7. 常见问题

### 为什么 GloVe 下载失败？

确认当前机器可以访问 GloVe 下载地址，并且已经安装 `curl` 或 `wget`。如果其他机器已经
下载 `glove.6B.50d.txt`，通过 `LOCAL_GLOVE_TXT` 指向该文件即可。

### 为什么 ResNet50 权重下载失败？

Quickstart 使用 TorchVision 的 ImageNet 预训练 ResNet50。确认当前环境可以访问 PyTorch
模型下载地址，或预先在相同用户的 Torch cache 中准备对应权重。

### 为什么重新运行后仍会覆盖 checkpoint？

Tiny example 使用固定的 `output/quickstart_baselines/` 配置路径，适合重复执行链路检查。
需要保留某次输出时，请先将该目录移动到新的实验目录，再重新运行 Quickstart。

### 为什么 tiny example 的 Success 或 SPL 很低？

示例只有两个 Episode，训练样本和 optimizer update 数量都很少。它用于验证代码、数据、
checkpoint 和 rollout 链路，不用于衡量模型质量。

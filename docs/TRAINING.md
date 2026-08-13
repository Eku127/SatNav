# SatNav 模型训练

本文介绍 SatNav 的通用训练流程，并使用仓库内置的 tiny example 完成第一次端到端训练。
Quickstart 会生成离线轨迹、训练 Seq2Seq 和 CMA、保存 checkpoint，并在 SatSim 中运行评测。

开始前建议先阅读[环境安装](INSTALLATION.md)和[数据格式](DATASET_FORMAT.md)。如果只需要将
已有模型接入在线评测，请直接阅读[模型接入](MODEL_INTEGRATION.md)。

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
四个 VLM 的依赖互不兼容，安装方式参阅对应 baseline 文档。

使用完整 SatNav-v0.1 数据训练 Seq2Seq 和 CMA 的配置与命令参阅
[Classic Baselines](BASELINE_CLASSIC.md)。
StreamVLN 的独立环境、模型下载、数据校验、训练和评测流程参阅
[StreamVLN Baseline](BASELINE_STREAMVLN.md)。
NaVILA 的独立环境、上游代码、模型下载、训练和评测流程参阅
[NaVILA Baseline](BASELINE_NAVILA.md)。
Uni-NaVid 的独立环境、模型准备、训练和评测流程参阅
[Uni-NaVid Baseline](BASELINE_UNINAVID.md)。
OpenFly 的独立环境、模型准备、训练和评测流程参阅
[OpenFly Baseline](BASELINE_OPENFLY.md)。

## 2. 训练输入

一次完整训练通常需要以下输入：

| 输入 | 作用 | 准备方式 |
| --- | --- | --- |
| Episode JSON | 提供指令、起点、目标和 reference path | [Episode 数据下载](DATA_DOWNLOAD.md) |
| GeoTIFF scenes | 渲染 expert trajectory 的 RGB observation | [卫星场景下载](APPLICATION_MAP_DOWNLOAD.md) |
| Offline trajectory | 提供 `annotations.json`、RGB frame 和 action | [轨迹数据生成](APPLICATION_TRAJ_GENERATION.md) |
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
[数据格式 - 离线 trajectory](DATASET_FORMAT.md#6-离线-trajectory)。

训练配置中的 Episode、scene、trajectory、checkpoint 和输出路径必须指向同一组实验数据。
本机路径应放入 Git ignored 的 `.local/env.sh` 或 `configs/local_*.yaml`，不要写入公共配置。

## 3. Tiny Example

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

## 4. 准备 Quickstart 环境

在 SatNav 仓库根目录创建并激活 classic 环境：

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav

python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic,applications]'
```

如果环境已经按照[安装指南](INSTALLATION.md)准备完成，不需要重复创建。使用其他 CUDA
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

## 5. 运行 Tiny Example Quickstart

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

## 6. 常见问题

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

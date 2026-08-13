# SatNav 安装指南

本文介绍从源码安装 SatNav。Core、classic baseline 与各 VLM baseline 的依赖边界不同，
请按实际用途选择环境。

## 1. 获取代码

```bash
git clone https://github.com/Eku127/SatNav.git
cd SatNav
```

## 2. 安装 Core

Core 包含 SatNav 数据集、SatSim、环境 API 和统一评测框架，不依赖 PyTorch。

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav
python -m pip install --upgrade pip
python -m pip install -e .
```

验证安装：

```bash
python -c "from satnav.core.env import Env; print('SatNav import OK')"
```

## 3. 安装可选组件

### 数据生产与可视化工具

安装地图下载、轨迹生成和视频相关依赖：

```bash
python -m pip install -e '.[applications]'
python -m applications.trajectory_generation.generate --help
```

如果需要下载地图、生成训练轨迹或导出视频，请安装该组件。

### Classic baselines

运行 classic baselines 需要额外安装 PyTorch。Random 和 ReferenceFollower 不需要模型权重；Seq2Seq 和 CMA 需要对应的训练权重。以下 CUDA 12.1 组合已经验证：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic]'
python -m baselines.classic --help
```

如果使用其他 CUDA 版本，请从 PyTorch 官方源选择匹配的 wheel。

安装完成后，可以按照[模型训练](../training/README.md)使用仓库 tiny example 运行 Seq2Seq 和 CMA
端到端训练。

## 4. VLM baselines

四套 VLM 的 PyTorch、Transformers 和 FlashAttention 版本互不兼容。每套 VLM 必须使用独立 Conda 环境，不能复用 Core 或其他 VLM 的环境。

| Baseline | Python | PyTorch | 安装说明 |
| --- | --- | --- | --- |
| StreamVLN | 3.9 | 2.5.1 | [StreamVLN Baseline](../training/vlm/STREAMVLN.md) |
| NaVILA | 3.10 | 2.3.0 | [NaVILA Baseline](../training/vlm/NAVILA.md) |
| Uni-NaVid | 3.9 | 2.5.1 | [Uni-NaVid Baseline](../training/vlm/UNINAVID.md) |
| OpenFly | 3.10 | 2.3.0 | [OpenFly Baseline](../training/vlm/OPENFLY.md) |

每个 VLM 环境都需要安装 SatNav。建议按对应 baseline 文档的顺序执行：

1. 创建该 baseline 的 Conda 环境；
2. 安装与 CUDA 匹配的 PyTorch；
3. 安装匹配 Python、PyTorch、CUDA 和 CXX11 ABI 的 FlashAttention wheel；
4. 安装该目录的 `requirements.txt`；
5. 回到 SatNav 仓库根目录，安装 SatNav：

```bash
python -m pip install -e .
```

这里的 `-e .` 指向 SatNav 仓库根目录，不是 `baselines/vlm/<name>` 目录。该命令只安装 SatNav Core，不会引入其他 VLM 的模型依赖。

OpenFly 提供一键环境脚本：

```bash
bash baselines/vlm/openfly/scripts/bootstrap_env.sh satnav-openfly
```

该脚本会创建独立环境并自动安装 SatNav；其他三套 VLM 需要按各自 README 手动完成上述步骤。

## 5. 本地配置

数据集、场景、模型和输出路径不要写入公共配置。复制本地模板后再修改：

```bash
mkdir -p .local
cp local.env.example .local/env.sh
```

Classic 和各 VLM 目录也提供各自的 `local.env.example`。使用某个 baseline 时，应在对应目录创建 `.local/env.sh`，例如：

```bash
mkdir -p baselines/vlm/streamvln/.local
cp baselines/vlm/streamvln/local.env.example \
  baselines/vlm/streamvln/.local/env.sh
```

所有 `.local/env.sh` 都已被 Git 忽略，仅用于保存本机路径。

## 6. 常见问题

### 为什么出现 `ModuleNotFoundError: satnav`？

确认当前终端位于 SatNav 仓库根目录，并且已经在当前 Python 环境中执行：

```bash
python -m pip install -e .
python -c "import satnav; print(satnav.__file__)"
```

如果使用 Conda，请先确认 `which python` 指向准备运行 SatNav 的环境。

### 为什么 CUDA 或模型算子无法加载？

检查 Python、PyTorch、CUDA 和 FlashAttention wheel 是否属于同一套兼容组合。先运行
`python -m pip check`，再按照对应 baseline README 中列出的版本重新安装。不要在一个
环境中混用多套 VLM 的依赖。

### 为什么安装一个 VLM 后另一个 VLM 无法运行？

四个 VLM baseline 的 PyTorch、Transformers 和 FlashAttention 版本不同。每个 baseline
应使用独立 Conda 环境；删除混用环境后，按照对应 baseline 的环境文件重新创建。

### 为什么 `rasterio` 或 `pyproj` 安装失败？

优先使用 `environments/satnav/conda.yml` 创建干净环境，让 Conda 安装底层地理空间库。
避免混用系统 Python、系统 GDAL 和来自不同渠道的二进制 wheel。

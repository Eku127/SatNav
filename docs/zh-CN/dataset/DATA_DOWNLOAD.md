# SatNav Episode 数据下载

本文介绍 SatNav Episode 元数据的下载、目录配置与完整性校验。卫星场景需单独准备，参阅[卫星场景下载](../applications/MAP_DOWNLOAD.md)。

## 1. 数据集说明

| 名称 | 数据 | 获取方式 | 内容 |
| --- | --- | --- | --- |
| SatNav-Episodes-v0.1 | Episode 元数据 | [Hugging Face](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1) | 指令、起点、目标、waypoint、reference path 和数据划分 |

Episodes 数据集包含 Episode JSON、train/evaluation 划分、场景范围 `scenes_list.yaml` 和数据说明。配套 GeoTIFF 有两种获取方式：在 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) 填表申请并下载，或使用自己的地图 API 凭据通过脚本生成。操作步骤见[卫星场景下载](../applications/MAP_DOWNLOAD.md)。

## 2. 下载与配置

从 Hugging Face 下载到本地数据目录：

```bash
pip install -U huggingface_hub
hf download Eku127/SatNav-Episodes-v0.1 --repo-type dataset \
  --local-dir data/satnav_datasets/SatNav-v0.1
```

访问私有仓库前，先运行 `hf auth login`，并使用已获授权的账户登录。
SatNav 读取 `episodes/` 下的 JSON；`data/` 下的 Parquet 用于 Hugging Face 数据预览。
关键文件结构如下：

```text
data/satnav_datasets/SatNav-v0.1/
├── episodes/
│   ├── train/all_episodes.json
│   └── eval/
│       ├── val_seen/all_episodes.json
│       └── val_unseen/all_episodes.json
├── scenes_list.yaml
└── SHA256SUMS
```

在 SatNav 仓库根目录设置数据路径；如数据位于其他磁盘，请替换为对应的绝对路径：

```bash
export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
```

## 3. 完整性校验

在数据根目录执行 SHA256 校验：

```bash
cd "$SATNAV_DATA_ROOT"
sha256sum -c SHA256SUMS
cd -
```

校验通过后，各标准划分的 Episode 数量应为：

| Split | Episodes |
| --- | ---: |
| train | 105,164 |
| val_seen | 4,574 |
| val_unseen | 8,756 |
| **总计** | **118,494** |

## 4. 数据许可

Episode JSON 和相关 benchmark 参数文件按 [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) 发布，数据说明文档按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。完整说明参阅[数据许可](../../../DATA_LICENSE.md)。

配套场景的使用条件见 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) 页面；API 生成场景遵循对应影像服务商的条款。

## 5. 下一步

按照[卫星场景下载](../applications/MAP_DOWNLOAD.md)准备 SatSim 所需的 GeoTIFF 场景。

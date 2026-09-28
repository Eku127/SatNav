# SatNav 卫星场景下载

SatSim 使用 59 个 GeoTIFF 场景。可以选择下载已准备好的场景，或使用自己的地图服务 API 凭据，通过 SatNav 脚本生成场景。

| 获取方式 | 步骤 |
| --- | --- |
| 方式一：下载现成场景 | 在 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) 填写申请表并同意使用条款，申请通过系统检查后下载 |
| 方式二：通过 API 生成 | 注册 Google Map Tiles API 或 Mapbox，配置凭据后运行地图下载脚本 |

场景范围定义在 Episodes 数据集的 `scenes_list.yaml` 中，见 [Episode 数据下载](../dataset/DATA_DOWNLOAD.md)。GeoTIFF 如何生成 RGB observation，见 [SatSim 观测原理](../concepts/SATSIM.md)。

## 方式一：申请并下载现成场景

打开 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1)，登录 Hugging Face，填写姓名、机构、机构邮箱和研究用途，勾选声明并提交。申请通过系统检查后，该账号将获得下载权限。

使用同一账号登录命令行，在 SatNav 仓库根目录下载场景和校验清单（共 64.79 GB）：

```bash
pip install -U huggingface_hub
hf auth login
hf download Eku127/SatNav-Scenes-v0.1 --repo-type dataset \
  --include "scenes/*.tif" --include SHA256SUMS \
  --local-dir "$PWD/data/satnav_datasets"
```

设置数据路径，并校验下载的 GeoTIFF：

```bash
export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
(cd "$PWD/data/satnav_datasets" && sha256sum -c SHA256SUMS)
```

完成后直接进入下方“校验数据配置”。

## 方式二：使用自己的 API 凭据生成场景

### 1. 准备环境

先完成[环境安装](../getting-started/INSTALLATION.md)，然后在 SatNav 仓库根目录安装 applications 依赖并设置数据路径：

```bash
python -m pip install -e '.[applications]'

export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
mkdir -p "$SATNAV_SCENES_DIR"
```

如数据位于其他磁盘，请将环境变量改为对应的绝对路径。

### 2. 配置服务凭据

选择实际使用的服务商。Google Map Tiles API 支持按场景清单批量生成；Mapbox 支持单场景生成。

#### Google Map Tiles API

1. 创建或选择已启用结算的 Google Cloud 项目。
2. 启用 [Map Tiles API](https://console.cloud.google.com/apis/library/tile.googleapis.com)。
3. 在[凭据页面](https://console.cloud.google.com/google/maps-apis/credentials)创建 API key。
4. 将 key 限制为仅可访问 Map Tiles API；条件允许时，再限制可信 IP。

```bash
export GOOGLE_MAPS_API_KEY="your-api-key"
```

参考 [Google 官方配置指南](https://developers.google.com/maps/documentation/tile/get-api-key)。

#### Mapbox

1. 创建 Mapbox 账户。
2. 在 [Access Tokens](https://console.mapbox.com/account/access-tokens/) 页面创建 token。
3. 仅授予读取地图瓦片所需的最小权限，例如 `styles:tiles`。

```bash
export MAPBOX_ACCESS_TOKEN="your-access-token"
```

参考 [Mapbox token 文档](https://docs.mapbox.com/accounts/guides/tokens/)。不要将真实凭据写入脚本或提交到 Git。

### 3. 批量生成 59 个场景

`--scene-config` 批量模式目前仅支持 Google。请先执行 dry run，检查场景配置和输出路径；该命令不会下载瓦片：

```bash
python -m applications.map_downloader google \
  --scene-config "$SATNAV_DATA_ROOT/scenes_list.yaml" \
  --output-dir "$SATNAV_SCENES_DIR" \
  --dry-run
```

确认无误后开始下载：

```bash
python -m applications.map_downloader google \
  --scene-config "$SATNAV_DATA_ROOT/scenes_list.yaml" \
  --output-dir "$SATNAV_SCENES_DIR"
```

已有 GeoTIFF 默认会被跳过，因此中断后可直接重新执行同一命令。常用选项包括：

- `--scene-id Geneva-1`：只处理指定场景，可重复传入。
- `--limit 2`：只处理配置中的前两个场景。
- `--overwrite`：重新生成已有文件。
- `--max-workers N`：设置并行下载数量。
- `--use-env-proxy`：使用系统代理环境变量。

下载前请确认 API 配额、费用和可用磁盘空间。

### 4. 下载单个场景

Google 和 Mapbox 均支持单场景下载。以下示例使用两个 WGS84 角点定义区域：

```bash
python -m applications.map_downloader google \
  --type corners \
  --lat1 46.161791698085 --lon1 6.082932204008 \
  --lat2 46.179030896969 --lon2 6.116151362658 \
  --zoom 19 \
  --output "$SATNAV_SCENES_DIR/Geneva-1.tif"
```

使用 Mapbox 时，将子命令 `google` 改为 `mapbox`，其余范围参数保持不变。也可以通过 `--type center` 配合 `--center-lat`、`--center-lon`、`--height-m` 和 `--width-m`，按中心点和实际尺寸定义区域。

SatNav 场景使用 zoom level `19`。下载器默认保留服务商要求的 attribution；Mapbox 还会生成对应的 `.attribution.txt` 文件。

## 5. 校验数据配置

完成下载后，在仓库根目录运行：

```bash
bash scripts/validation/data_validation.sh
```

脚本会检查：

- GeoTIFF 文件数量是否为 59；
- train、val_seen 和 val_unseen 的 Episode 数量；
- Episode 数据能否正常加载；
- 所有 Episode 引用的场景是否存在。

校验通过后，终端会输出：

```text
SatNav data configuration is complete.
```

## 6. 常见问题

### 为什么运行下载命令时出现 `ModuleNotFoundError`？

在 SatNav 仓库根目录为当前环境安装 applications 依赖：

```bash
python -m pip install -e '.[applications]'
```

### 为什么 API 鉴权失败？

确认凭据环境变量已经在当前终端生效，并检查对应 API 是否启用、账号结算状态、key 使用
限制和 token scope。不要将 key 或 token 写入仓库配置。

### 下载中断后如何继续？

重新执行相同命令即可续传。不要添加 `--overwrite`，否则已经完成的场景也会重新下载。

### 为什么无法连接 `tile.googleapis.com`？

如果当前网络需要代理，添加 `--use-env-proxy`，让下载器读取终端中的代理环境变量后重试。

### 为什么 Google satellite tiles 不可用？

除配额和区域覆盖外，绑定 EEA 账单地址的项目无法获取 2D satellite tiles。根据命令返回的
错误码检查 [Google 错误说明](https://developers.google.com/maps/documentation/tile/error_handling)。

## 7. 数据来源与使用条款

现成场景的申请和使用条件见 [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1)。通过 API 获取影像时，使用自己的账号凭据，并按服务商协议确定存储和研究使用范围。服务文档：[Google Map Tiles API 政策](https://developers.google.com/maps/documentation/tile/policies)、[计费说明](https://developers.google.com/maps/documentation/tile/usage-and-billing)、[Mapbox Raster Tiles API](https://docs.mapbox.com/api/maps/raster-tiles/)。

场景准备完成后，可使用 [SatSim Viewer](VIEWER.md)检查 GeoTIFF，或按照[轨迹数据生成](TRAJECTORY_GENERATION.md)生成离线训练数据。

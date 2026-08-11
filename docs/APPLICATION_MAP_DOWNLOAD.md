# SatNav 卫星场景下载

本文介绍如何使用 `applications/map_downloader` 将地图瓦片拼接为 SatSim 可读取的 GeoTIFF 场景。SatNav-Episodes-v0.1 所需的 59 个场景范围定义在 `scenes_list.yaml` 中；如尚未准备该文件，请先完成 [Episode 数据下载](DATA_DOWNLOAD.md)。

> 地图下载器只提供技术能力，不授予地图内容的下载、存储、分发或机器学习使用许可。请在使用前确认服务商的最新条款及你的授权范围。

## 1. 准备环境

先完成[环境安装](INSTALLATION.md)，然后在 SatNav 仓库根目录安装 applications 依赖并设置数据路径：

```bash
python -m pip install -e '.[applications]'

export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
mkdir -p "$SATNAV_SCENES_DIR"
```

如数据位于其他磁盘，请将环境变量改为对应的绝对路径。

## 2. 配置服务凭据

只需配置实际使用的服务商。建议使用google的api来进行tile的下载

### Google Map Tiles API

1. 创建或选择已启用结算的 Google Cloud 项目。
2. 启用 [Map Tiles API](https://console.cloud.google.com/apis/library/tile.googleapis.com)。
3. 在[凭据页面](https://console.cloud.google.com/google/maps-apis/credentials)创建 API key。
4. 将 key 限制为仅可访问 Map Tiles API；条件允许时，再限制可信 IP。

```bash
export GOOGLE_MAPS_API_KEY="your-api-key"
```

参考 [Google 官方配置指南](https://developers.google.com/maps/documentation/tile/get-api-key)。

### Mapbox

1. 创建 Mapbox 账户。
2. 在 [Access Tokens](https://console.mapbox.com/account/access-tokens/) 页面创建 token。
3. 仅授予读取地图瓦片所需的最小权限，例如 `styles:tiles`。

```bash
export MAPBOX_ACCESS_TOKEN="your-access-token"
```

参考 [Mapbox token 文档](https://docs.mapbox.com/accounts/guides/tokens/)。不要将真实凭据写入脚本或提交到 Git。

## 3. 批量生成 59 个场景

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

## 4. 下载单个场景

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

- `ModuleNotFoundError`：确认当前环境已执行 `python -m pip install -e '.[applications]'`。
- API 鉴权失败：检查环境变量、API 是否启用、结算状态、key 限制和 token scope。
- 下载中断：重新执行原命令即可续传，不要添加 `--overwrite`。
- 网络无法连接 `tile.googleapis.com`：添加 `--use-env-proxy` 后重试。
- Google satellite tiles 不可用：除配额和区域覆盖外，绑定 EEA 账单地址的项目无法获取 2D satellite tiles，参阅 [Google 错误说明](https://developers.google.com/maps/documentation/tile/error_handling)。

## 7. 使用条款

Google Map Tiles API 当前政策限制未经授权的预取、存储和离线使用，并将图像分析、机器解释等列为不可使用的非可视化场景。除非你的协议另有许可，否则不要将其输出用于离线训练或评测。使用前请阅读 [Google Map Tiles API 政策](https://developers.google.com/maps/documentation/tile/policies)和[计费说明](https://developers.google.com/maps/documentation/tile/usage-and-billing)。

Mapbox 用户请阅读 [Raster Tiles API 文档](https://docs.mapbox.com/api/maps/raster-tiles/)及对应服务条款。SatNav 不分发第三方卫星影像，也不替用户获得或转授地图内容许可。

场景准备完成后，可使用 [SatSim Viewer](APPLICATION_VIEWER.md)检查 GeoTIFF，或按照[轨迹数据生成](APPLICATION_TRAJ_GENERATION.md)生成离线训练数据。

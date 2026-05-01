# Map Downloader

`applications/map_downloader/` 统一管理 SatNav 的卫星图下载工具，当前包含两个 provider：

- `google_downloader/`：Google Map Tiles API XYZ。
- `mapbox_downloader/`：Mapbox Raster Tiles API。

顶层入口和其它 `applications/*` 工具保持一致：

```bash
python -m applications.map_downloader google --help
python -m applications.map_downloader mapbox --help
```

为了兼容旧用法，不传 provider 时默认使用 `google`：

```bash
python -m applications.map_downloader --help
```

Provider 也可以直接运行：

```bash
python -m applications.map_downloader.google_downloader --help
python -m applications.map_downloader.mapbox_downloader --help
```

## 目录结构

```text
applications/map_downloader/
  __init__.py
  __main__.py
  main.py
  config.yaml
  google_downloader/
    __init__.py
    __main__.py
    config.yaml
    downloader.py
    generate_geotiff.py
    assets/
  mapbox_downloader/
    __init__.py
    __main__.py
    config.yaml
    downloader.py
    generate_geotiff.py
    assets/
```

生成的 GeoTIFF 不应放在 `applications/map_downloader/data/`。建议输出到仓库根目录已忽略的 `output/`，或外部数据目录，例如：

```bash
--output-dir output/map_downloader/google
```

## Key 配置

优先使用环境变量，不要把真实 key 写进仓库配置：

```bash
export GOOGLE_MAPS_API_KEY="..."
export MAPBOX_ACCESS_TOKEN="..."
```

也可以写入 provider 配置文件的本地副本：

- `google_downloader/config.yaml` -> `GOOGLE.API_KEY`
- `mapbox_downloader/config.yaml` -> `MAPBOX.ACCESS_TOKEN`

## Google 示例

中心点 + 宽高：

```bash
python -m applications.map_downloader google \
  --type center \
  --center-lat 41.939165 \
  --center-lon 12.483188 \
  --height-m 500 \
  --width-m 500 \
  --zoom 19 \
  --output-dir output/map_downloader/google
```

两个角点：

```bash
python -m applications.map_downloader google \
  --type corners \
  --lat1 41.936900 \
  --lon1 12.480200 \
  --lat2 41.941400 \
  --lon2 12.486100 \
  --zoom 19 \
  --output output/map_downloader/google/google_corners.tif
```

## Mapbox 示例

中心点 + 宽高：

```bash
python -m applications.map_downloader mapbox \
  --type center \
  --center-lat 41.939165 \
  --center-lon 12.483188 \
  --height-m 500 \
  --width-m 500 \
  --zoom 19 \
  --output-dir output/map_downloader/mapbox
```

两个角点：

```bash
python -m applications.map_downloader mapbox \
  --type corners \
  --lat1 41.936900 \
  --lon1 12.480200 \
  --lat2 41.941400 \
  --lon2 12.486100 \
  --zoom 19 \
  --output output/map_downloader/mapbox/mapbox_corners.tif
```

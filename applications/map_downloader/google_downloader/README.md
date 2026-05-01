# Google XYZ Downloader

这个包只使用官方 Google Map Tiles API 的 XYZ 瓦片接口生成 GeoTIFF。旧的 Google Static Maps 测试分支已从 release 代码中移除。

## 使用

推荐先设置环境变量：

```bash
export GOOGLE_MAPS_API_KEY="你的 Google key"
```

按中心点下载：

```bash
python -m applications.map_downloader google \
  --type center \
  --center-lat 41.939165 \
  --center-lon 12.483188 \
  --height-m 500 \
  --width-m 500 \
  --zoom 19
```

如果需要走系统代理：

```bash
python -m applications.map_downloader google \
  --type center \
  --center-lat 41.939165 \
  --center-lon 12.483188 \
  --height-m 100 \
  --width-m 100 \
  --zoom 19 \
  --use-env-proxy
```

输出为 `EPSG:3857` GeoTIFF，默认在最终图像左下角添加 Google 返回的 attribution 文本。

# Concept figure sources and reproduction

## Camera examples / 相机示例

The camera figures use a local, user-supplied `Amsterdam-1.tif`. Only the illustrated crops are included in the documentation; the source GeoTIFF stays in the external scene collection.

| Item | Value |
| --- | --- |
| Scene | `Amsterdam-1.tif` |
| Raster | 12774 × 16375, EPSG:3857, bands=4 |
| SHA-256 | `c937756c89aa904e7e8546a61188122da2a75f07c8939a16f35f0b087831611d` |
| Center fraction (column, row) | `0.552, 0.653` in the EPSG:3857 raster |
| Center WGS84 (longitude, latitude) | `4.882231275, 52.362661288` |
| Center EPSG:3857 (x, y) | `543487.499451, 6865966.777350` |
| Output observation | 448 × 448 RGB |
| Context map | 290 × 290 ground meters, north up |
| Raster metadata | `{'AREA_OR_POINT': 'Area'}` |

`camera-coverage.*.png` holds three settings at heading 0°: (altitude 50 m, HFOV 90°), (100 m, 90°), and (50 m, 60°). `camera-heading.*.png` holds heading 0°, 45°, and 90° at altitude 75 m and HFOV 90°. Footprint outlines use the same plane geometry and local Mercator scale as the camera. In the heading figure, A and C footprints coincide; their line patterns distinguish the outlines.

中文：两组图分别比较高度与视场角、朝向。RGB 直接调用仓库相机渲染器生成；地图彩框按相同几何计算。朝向对比中 A 与 C 的方形覆盖框重合，观测图像方向不同。

From the repository root, in a Python environment with `rasterio`, `opencv-python-headless`, `pyproj`, `matplotlib`, `numpy`, and `Pillow`:

```bash
python scripts/docs/render_camera_examples.py --scene /path/to/Amsterdam-1.tif --center-fraction 0.552 0.653
```

The script uses Microsoft YaHei on Windows. On another platform, install Noto Sans CJK SC or pass `--font /path/to/cjk-font.ttf`.

## Editable diagrams / 可编辑机制图

Each of the eight diagrams has Chinese and English SVG and uncompressed `.drawio` versions. Their shapes, labels, and connectors share one source in `scripts/docs/generate_concept_diagrams.py`.

```bash
python scripts/docs/generate_concept_diagrams.py
```

Open `.drawio` files in diagrams.net to edit individual elements. Update the generator when maintaining reproducible SVG/draw.io pairs. Camera panels are rendered data figures; the editable camera geometry and coordinate diagrams are in `diagrams/`.

The mechanism pages were checked against SatNav code baseline `402027e`.

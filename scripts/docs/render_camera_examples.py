#!/usr/bin/env python3
"""Render documented camera comparisons from a user-supplied GeoTIFF.

Requires rasterio, opencv-python-headless, pyproj, matplotlib, numpy and Pillow.
Uses the repository's SatelliteCamera implementation; keeps source scenes external.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/assets/concepts"
spec = importlib.util.spec_from_file_location("satnav_camera", ROOT / "satnav/sims/satsim/camera.py")
camera_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_module)
SatelliteCamera = camera_module.SatelliteCamera
COLORS = ["#42BBFF", "#FFD05C", "#FD81B9"]


def footprint(h, hfov, heading):
    half = h * math.tan(math.radians(hfov / 2))
    corners = np.array([[-half, -half], [-half, half], [half, half], [half, -half]])
    angle = math.radians(heading)
    rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return corners @ rotation.T


def render_figure(scene, center, scale, settings, kind, locale, scene_name):
    zh = locale == "zh-CN"
    radius = 145.0
    cx, cy = center
    extent = (cx-radius*scale, cy-radius*scale, cx+radius*scale, cy+radius*scale)
    window = from_bounds(*extent, transform=scene.transform)
    context = scene.read([1, 2, 3], window=window, out_shape=(3, 800, 800), resampling=Resampling.bilinear).transpose(1, 2, 0)
    fig, axs = plt.subplots(2, 2, figsize=(10, 12), dpi=150)
    fig.patch.set_facecolor("white")
    # Reserve separate bands for the figure title, subtitle, and panel titles.
    fig.subplots_adjust(left=.065, right=.97, bottom=.055, top=.84, wspace=.17, hspace=.25)
    title = ("高度与 HFOV：同一位置的覆盖范围" if zh else "Altitude and HFOV: coverage at one position") if kind == "coverage" else ("Heading：同一位置的视野旋转" if zh else "Heading: rotating the view at one position")
    fig.suptitle(title, fontsize=21, fontweight="bold", color="#20364C", y=.97)
    fig.text(.5,.91,f"{scene_name}  ·  448 × 448 RGB  ·  SatelliteCamera.render_image()",ha="center",va="center",fontsize=11,color="#52677C")
    ax=axs.flat[0]
    ax.imshow(context, extent=(-radius,radius,-radius,radius))
    ax.set_title("地图与地面覆盖框" if zh else "Map and ground footprints", fontsize=15, pad=12)
    # Draw largest footprints first so the smaller ones remain visible.
    for i in sorted(range(3), key=lambda i: settings[i][0]*math.tan(math.radians(settings[i][1]/2)), reverse=True):
        h,hfov,heading=settings[i]
        poly=Polygon(footprint(h,hfov,heading),closed=True,fill=False,edgecolor=COLORS[i],linewidth=2.6,linestyle=["-","--",":"][i])
        ax.add_patch(poly)
    ax.plot(0,0,"o",markersize=5,color="white",markeredgecolor="#20364C")
    ax.annotate("N",xy=(122,121),xytext=(122,79),ha="center",color="white",fontsize=15,fontweight="bold",arrowprops={"arrowstyle":"-|>","color":"white","lw":2})
    ax.plot([-125,-75],[-125,-125],lw=4,color="white")
    ax.text(-100,-117,"50 m",ha="center",color="white",fontsize=11,fontweight="bold")
    ax.set_xticks([]);ax.set_yticks([])
    for i,(h,hfov,heading) in enumerate(settings):
        camera=SatelliteCamera(448,448,hfov)
        rgb=camera.render_image(scene,center,h,heading)
        assert rgb.shape==(448,448,3) and rgb.dtype==np.uint8
        ax=axs.flat[i+1]
        ax.imshow(rgb)
        ax.set_title(f"{'ABC'[i]}   h={h:g} m · HFOV={hfov:g}° · θ={heading:g}°",fontsize=12.5,pad=12,color="#20364C")
        ax.set_xticks([]);ax.set_yticks([])
        width=2*h*math.tan(math.radians(hfov/2))
        ax.set_xlabel((f"地面覆盖 {width:.1f} × {width:.1f} m" if zh else f"Ground coverage {width:.1f} × {width:.1f} m"),fontsize=12,labelpad=9,color="#52677C")
        for spine in ax.spines.values(): spine.set_edgecolor(COLORS[i]);spine.set_linewidth(3)
    fig.savefig(OUT/f"camera-{kind}.{locale}.png",dpi=150,facecolor="white")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--center-fraction",type=float,nargs=2,default=(.552,.653),metavar=("COL","ROW"),help="Fractional column and row in the EPSG:3857 raster")
    parser.add_argument("--font",type=Path,help="Optional CJK font file")
    args=parser.parse_args()
    if not all(0 < v < 1 for v in args.center_fraction):
        parser.error("center fractions must be between 0 and 1")
    font=args.font or Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.family"]=["Noto Sans CJK SC","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    OUT.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        source=stack.enter_context(rasterio.open(args.scene))
        scene=source if source.crs and source.crs.to_epsg()==3857 else stack.enter_context(WarpedVRT(source,crs="EPSG:3857"))
        fx,fy=args.center_fraction
        center=(scene.bounds.left+fx*(scene.bounds.right-scene.bounds.left),scene.bounds.top-fy*(scene.bounds.top-scene.bounds.bottom))
        lon,lat=Transformer.from_crs(3857,4326,always_xy=True).transform(*center)
        scale=1/math.cos(math.radians(lat))
        settings={"coverage":[(50,90,0),(100,90,0),(50,60,0)],"heading":[(75,90,0),(75,90,45),(75,90,90)]}
        for kind,values in settings.items():
            for locale in ["zh-CN","en-US"]: render_figure(scene,center,scale,values,kind,locale,args.scene.stem)
        source_summary=f"{source.width} × {source.height}, {source.crs}, bands={source.count}"
        tags=source.tags()
    digest=hashlib.sha256()
    with args.scene.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""): digest.update(chunk)
    text=f'''# Concept figure sources and reproduction

## Camera examples / 相机示例

The camera figures use a local, user-supplied `{args.scene.name}`. Only the illustrated crops are included in the documentation; the source GeoTIFF stays in the external scene collection.

| Item | Value |
| --- | --- |
| Scene | `{args.scene.name}` |
| Raster | {source_summary} |
| SHA-256 | `{digest.hexdigest()}` |
| Center fraction (column, row) | `{fx}, {fy}` in the EPSG:3857 raster |
| Center WGS84 (longitude, latitude) | `{lon:.9f}, {lat:.9f}` |
| Center EPSG:3857 (x, y) | `{center[0]:.6f}, {center[1]:.6f}` |
| Output observation | 448 × 448 RGB |
| Context map | 290 × 290 ground meters, north up |
| Raster metadata | `{tags}` |

`camera-coverage.*.png` holds three settings at heading 0°: (altitude 50 m, HFOV 90°), (100 m, 90°), and (50 m, 60°). `camera-heading.*.png` holds heading 0°, 45°, and 90° at altitude 75 m and HFOV 90°. Footprint outlines use the same plane geometry and local Mercator scale as the camera. In the heading figure, A and C footprints coincide; their line patterns distinguish the outlines.

中文：两组图分别比较高度与视场角、朝向。RGB 直接调用仓库相机渲染器生成；地图彩框按相同几何计算。朝向对比中 A 与 C 的方形覆盖框重合，观测图像方向不同。

From the repository root, in a Python environment with `rasterio`, `opencv-python-headless`, `pyproj`, `matplotlib`, `numpy`, and `Pillow`:

```bash
python scripts/docs/render_camera_examples.py --scene /path/to/{args.scene.name} --center-fraction {fx} {fy}
```

The script uses Microsoft YaHei on Windows. On another platform, install Noto Sans CJK SC or pass `--font /path/to/cjk-font.ttf`.

## Editable diagrams / 可编辑机制图

Each of the eight diagrams has Chinese and English SVG and uncompressed `.drawio` versions. Their shapes, labels, and connectors share one source in `scripts/docs/generate_concept_diagrams.py`.

```bash
python scripts/docs/generate_concept_diagrams.py
```

Open `.drawio` files in diagrams.net to edit individual elements. Update the generator when maintaining reproducible SVG/draw.io pairs. Camera panels are rendered data figures; the editable camera geometry and coordinate diagrams are in `diagrams/`.

The mechanism pages were checked against SatNav code baseline `402027e`.
'''
    (OUT/"README.md").write_text(text,encoding="utf-8")
    print(f"Rendered camera panels at WGS84 {lon:.9f}, {lat:.9f}; wrote provenance to {OUT / 'README.md'}")


if __name__=="__main__": main()

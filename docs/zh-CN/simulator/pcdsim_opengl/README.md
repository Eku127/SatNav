# PCDSim OpenGL 使用指南

PCDSim OpenGL 是 SatNav 推荐的点云渲染后端。它使用点云生成 RGB observation，同时保留
SatSim 的 GeoTIFF、移动、可通行性、导航指标和 `TOP_DOWN_MAP` 链路。

推荐直接使用配置：

[`configs/pcdsim_opengl_task.yaml`](../../../../configs/pcdsim_opengl_task.yaml)

仿真器参数已有默认值，正常使用不需要修改。运行前只需在配置的 `DATASET` 中核对三个数据路径：

```yaml
DATASET:
  DATA_PATH: F:/path/to/VLN_episodes.json
  SCENES_DIR: F:/path/to/geotiff-scenes
  PCD_PATH: F:/path/to/pointclouds
```

- `DATA_PATH`：本次实验使用的 Episode JSON；
- `SCENES_DIR`：GeoTIFF 场景目录；
- `PCD_PATH`：点云场景根目录。

`SCENES_DIR` 和 `PCD_PATH` 通常在首次部署时设置一次，之后更换实验数据时只需修改
`DATA_PATH`。

目前只支持两个场景：

- `Cambridge`
- `Birmingham`

Episode 的 `scene_id`、GeoTIFF 文件名和点云文件夹必须使用对应的完整场景名。

## 1. 依赖

先按照 SatNav 的[环境安装指南](../../getting-started/INSTALLATION.md)安装基础依赖，再安装
PCDSim OpenGL 额外使用的 ModernGL：

```bash
pip install "moderngl>=5.12,<6"
```

安装 `moderngl` 时会自动安装 `glcontext`。运行环境还需要：

- 支持 OpenGL 3.3 或更高版本的 GPU；
- 正确安装的显卡驱动；
- 足够的 GPU 显存。Cambridge 示例视野同时加载两个 PLY 块时约占用 2.98 GB。

可以用下面的命令检查 OpenGL 环境：

```bash
python -c "import moderngl; c=moderngl.create_standalone_context(require=330); print(c.info['GL_RENDERER']); c.release()"
```

## 2. 数据目录

推荐按照下面的结构放置数据：

```text
<dataset-root>/
├── VLN_episodes.json               # DATASET.DATA_PATH
├── scenes/                         # DATASET.SCENES_DIR
│   ├── Cambridge.tif
│   └── Birmingham.tif
└── pointclouds/                    # DATASET.PCD_PATH
    ├── Cambridge/
    │   ├── cambridge_block_0.ply
    │   ├── cambridge_block_1.ply
    │   ├── ...
    │   ├── pcdsim_index.json
    │   └── ground_height_grid_20m.json
    └── Birmingham/
        ├── birmingham_block_0.ply
        ├── birmingham_block_1.ply
        ├── ...
        ├── pcdsim_index.json
        └── ground_height_grid_20m.json
```

GeoTIFF 仍然是必需的。RGB 由点云渲染，但移动、地图与导航指标继续依赖 TIF。

## 3. 缓存文件

本指南旁边附带了两个场景的真实缓存文件：

| 场景 | PLY 空间索引 | 地面高度缓存 |
| --- | --- | --- |
| Cambridge | [`pcdsim_index.json`](Cambridge/pcdsim_index.json) | [`ground_height_grid_20m.json`](Cambridge/ground_height_grid_20m.json) |
| Birmingham | [`pcdsim_index.json`](Birmingham/pcdsim_index.json) | [`ground_height_grid_20m.json`](Birmingham/ground_height_grid_20m.json) |

使用时必须把它们放进对应场景的 PLY 根目录：

```text
DATASET.PCD_PATH/Cambridge/pcdsim_index.json
DATASET.PCD_PATH/Cambridge/ground_height_grid_20m.json
DATASET.PCD_PATH/Birmingham/pcdsim_index.json
DATASET.PCD_PATH/Birmingham/ground_height_grid_20m.json
```

不要修改文件名，也不要放到额外的 `cache/` 子目录中。缓存必须与实际 PLY 文件匹配；如果 PLY
文件名、大小或修改时间发生变化，空间索引会失效并触发重新扫描。

## 4. 渲染默认值

下面的 OpenGL 参数已经配置为默认值，无需修改：

```text
内部渲染：2048 × 2048
POINT_SIZE：3
POINT_SHAPE：circle
MSAA：4×
OpenGL：3.3
```

配置中的 RGB observation 默认为 `224×224`。渲染流程为：

```text
2048×2048 OpenGL 渲染
→ 小空洞填补
→ resize 到 224×224
→ 返回 RGB observation
```

## 5. 运行

使用 Path Follower 进行无视频测试：

```bash
python examples/satnav_path_follower_example.py \
  --config configs/pcdsim_opengl_task.yaml \
  --no-video
```

生成视频：

```bash
python examples/satnav_path_follower_example.py \
  --config configs/pcdsim_opengl_task.yaml \
  --output-dir output/pcdsim_opengl_path_follower
```

第一次进入新的点云区域时需要将相交的 PLY 上传到 GPU，因此首帧通常比后续帧慢。

## 6. 常见问题

- 找不到点云：检查 `PCD_PATH/<scene_id>/` 是否存在；
- 找不到 TIF：检查 `SCENES_DIR/<scene_id>.tif` 是否存在；
- 找不到高度缓存：确认 `ground_height_grid_20m.json` 与 PLY 位于同一目录；
- OpenGL context 创建失败：检查 ModernGL、显卡驱动和 OpenGL 3.3 支持；
- 显存不足：关闭其他 GPU 程序，或重新规划点云分块。

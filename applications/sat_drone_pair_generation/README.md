# Sat-Drone Pair Generation

`applications/sat_drone_pair_generation/` 统一承载 SatNav 内的 sat-drone pair 数据生产脚本。

设计原则：

- 按数据源拆目录，保留原脚本边界
- 不改各数据源的 build/export 逻辑
- 只增加统一入口和仓内包化运行支持

当前纳入的数据源：

- `denseuav`
- `gta_uav`
- `sues`
- `uavvisloc`

## 环境

使用仓库现有 `satnav` conda 环境即可，无需新增独立环境。

主流程依赖已由 SatNav 覆盖：

- `numpy`
- `Pillow`
- `scipy`

激活方式：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
```

## 统一入口

统一入口只做路由，不改各脚本 CLI 语义：

```bash
python -m applications.sat_drone_pair_generation [--config /path/to/config.yaml] <dataset> <command> [args...]
```

示例：

```bash
python -m applications.sat_drone_pair_generation denseuav build_pairs --help
python -m applications.sat_drone_pair_generation gta-uav build_pairs --help
python -m applications.sat_drone_pair_generation sues pipeline --help
python -m applications.sat_drone_pair_generation uavvisloc export_selected --help
```

其中：

- `gta-uav` 会映射到 `gta_uav`
- `uav-visloc` 会映射到 `uavvisloc`

默认会读取：

`applications/sat_drone_pair_generation/config.yaml`

如果你想把常用数据路径固定到配置里，编辑这个文件即可。统一入口会优先从配置中的 `input` / `output` / `args` 读取默认值，再用命令行显式参数覆盖。

配置示例：

```yaml
DATASETS:
  denseuav:
    build_pairs:
      input:
        dataset_root: /path/to/DenseUAV/DenseUAV
      output:
        output_dir: /path/to/output/denseuav
      args:
        workers: 16

  uavvisloc:
    export_selected:
      input:
        data_root: /path/to/UAV-VisLoc/data
        sat_bounds_csv: /path/to/satellite_coordinates_range.csv
      output:
        output_dir: /path/to/output/uavvisloc
      args:
        allow_missing_pose: true
```

例如：

```bash
python -m applications.sat_drone_pair_generation denseuav build_pairs
python -m applications.sat_drone_pair_generation uavvisloc export_selected --output-dir /tmp/uavvisloc_debug
```

上面第二条命令里，`--output-dir` 会覆盖配置中的 `output.output_dir`。

## 推荐主入口

各数据源继续沿用原来的推荐主入口：

- `denseuav`: `build_pairs.py`
- `gta_uav`: `build_pairs.py`
- `sues`: `pipeline.py`
- `uavvisloc`: `export_selected.py`

也可以直接按模块运行具体脚本，例如：

```bash
python -m applications.sat_drone_pair_generation.uavvisloc.export_selected --help
python -m applications.sat_drone_pair_generation.sues.pipeline --help
```

## 目录结构

```text
applications/sat_drone_pair_generation/
├── README.md
├── __main__.py
├── config.yaml
├── main.py
├── registry.py
├── denseuav/
├── gta_uav/
├── sues/
└── uavvisloc/
```

说明：

- `dev/` 调参脚本未纳入本应用
- `OrthoLoC` 不在当前范围内
- 各数据源输出结构保持原脚本定义，不强行统一为同一 schema

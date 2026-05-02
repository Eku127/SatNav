# SatNav Trajectory Generation

This tool generates trajectory training data from SatNav episodes. It follows
each episode's `reference_path`, renders SatSim RGB frames, and writes action
annotations.

## Quick Test

Shared example resources are stored under `applications/resources/`:

- `satnav_example_task.yaml`: task/simulator config for trajectory generation
- `satnav_example_episodes.json`: two tiny example episodes
- `map.tif`: tiny GeoTIFF scene used by the example episodes

Run the test generation from the repo root:

```bash
python -m applications.trajectory_generation.generate \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/trajectory_generation_test
```

Expected result:

```text
output/trajectory_generation_test/
├── annotations.json
├── summary.json
└── images/
    └── ...
```

`annotations.json` is the final JSON annotation file. `summary.json` is JSONL
and is used for resume checks.

## Parallel Generation

For larger datasets, use the parallel entrypoint:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/trajectory_generation_test_parallel \
  --num_workers 2
```

For real production runs, replace `DATASET.DATA_PATH` and `DATASET.SCENES_DIR`
in the YAML config with the target episode JSON and GeoTIFF scene directory.

## Processing SatNav Data

To generate trajectories from a SatNav dataset release, copy
`applications/resources/satnav_example_task.yaml` and update the `DATASET` fields:

```yaml
DATASET:
  TYPE: SatNav
  SPLIT: train
  DATA_PATH: /path/to/SatNav-Episodes-v0.1/episodes/train/all_episodes.json
  SCENES_DIR: /path/to/satnav_scenes
```

`DATA_PATH` should point to an episode JSON file. `SCENES_DIR` should contain
GeoTIFF files named by scene id, for example `Amsterdam-1.tif` for
`scene_id: Amsterdam-1`.

If the scene GeoTIFF files are not prepared yet, use the map downloader app to
download and build them first:

```bash
python -m applications.map_downloader
```

See `applications/map_downloader/README.md` for downloader configuration.

Then run:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config path/to/your_traj_gen_config.yaml \
  --output_dir output/trajectory_data_train \
  --num_workers 64
```

For a small sanity check before a full run, use the serial entrypoint with a
small episode-index file:

```json
{"episode_indices": [0, 1, 2]}
```

```bash
python -m applications.trajectory_generation.generate \
  --config path/to/your_traj_gen_config.yaml \
  --output_dir output/trajectory_data_smoke \
  --episode_indices_file path/to/episode_indices.json
```

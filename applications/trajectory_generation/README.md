# SatNav Trajectory Generation

This tool generates trajectory training data from SatNav episodes. It follows
each episode's `reference_path`, renders SatSim RGB frames, and writes action
annotations.

## Quick Test

Shared example resources are stored under `applications/resources/`:

- `satnav_example_task.yaml`: task/simulator config for trajectory generation
- `satnav_example_episodes.json`: two tiny example episodes
- `map.tif`: synthetic CC0 GeoTIFF scene used by the example episodes; see
  [resource provenance](../resources/README.md)

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

For real production runs, use
`applications/episode_processing/configs/trajectory_generation.yaml` and set
`SATNAV_TRAIN_EPISODES_PATH` plus `SATNAV_SCENES_DIR`. Its 3 m LandmarkSet
value is a waypoint-arrival radius for expert trajectory generation, not the
30 m online-evaluation success threshold.

## Processing SatNav Data

To generate trajectories from a SatNav dataset release, use the dedicated
production configuration and provide the dataset locations through its
environment variables:

```bash
export SATNAV_TRAIN_EPISODES_PATH=/path/to/SatNav-Episodes-v0.1/episodes/train/all_episodes.json
export SATNAV_SCENES_DIR=/path/to/satnav_scenes
```

`DATA_PATH` should point to an episode JSON file. `SCENES_DIR` should contain
GeoTIFF files named by scene id, for example `Amsterdam-1.tif` for
`scene_id: Amsterdam-1`.

Prepare the scene GeoTIFFs using either option:

- [Request and download SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1); submit the form and accept the terms to receive access once your request passes the system checks.
- Register for an imagery API and generate scenes with your own credentials using `applications/map_downloader`.

See [Satellite Scene Download](../../docs/en-US/applications/MAP_DOWNLOAD.md) for both workflows and path configuration.

Then run:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir output/trajectory_data_train \
  --num_workers 64
```

After generation or resume, validate the complete output tree against the
same source/config/scenes contract. The validator rejects unreferenced episode
directories and unexpected sidecars in addition to checking every action and
frame; `--decode-images` fully loads every JPEG through RGB conversion and
records an ordered content digest. The public CLI is a release gate: source
episodes, generation config, scenes, and `--decode-images` are mandatory so an
empty or structural-only check cannot be reported as a complete validation.

```bash
python scripts/validation/validate_trajectory_output.py \
  --annotations output/trajectory_data_train/annotations.json \
  --output-root output/trajectory_data_train \
  --source-episodes /path/to/SatNav-Episodes-v0.1/episodes/train/all_episodes.json \
  --generation-config path/to/your_traj_gen_config.yaml \
  --scenes-dir /path/to/satnav_scenes \
  --decode-images \
  --report output/trajectory_data_train.validation.json
```

The report must be outside the validated output tree and must not overlap the
source episode, generation config, or scene inputs, so writing the report
cannot alter an artifact that was just validated.

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

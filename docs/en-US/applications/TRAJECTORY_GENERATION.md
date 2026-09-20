# Generate SatNav Trajectories

This guide uses `applications/trajectory_generation` to produce offline
training data for VLN models. The generator drives SatSim along each episode's
`reference_path` and saves RGB frames, actions, and instructions.

For the path-following algorithm, waypoint switching, and frame/action alignment, see [expert trajectories](../concepts/EXPERT_TRAJECTORIES.md).

## 1. Prepare the environment

Complete [Installation](../getting-started/INSTALLATION.md),
[Episode Download](../dataset/DATA_DOWNLOAD.md), and
[Satellite Scene Download](MAP_DOWNLOAD.md), then run from the repository root:

```bash
python -m pip install -e '.[applications]'

export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
export SATNAV_TRAIN_EPISODES_PATH="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
export SATNAV_SCENES_DIR="$PWD/data/satnav_datasets/scenes"
export SATNAV_TRAJECTORY_DIR="$SATNAV_DATA_ROOT/trajectory_data"
```

> A complete train export currently occupies about 233 GB. Reserve at least
> 250 GB before starting.

Verify the episodes and all 59 GeoTIFFs:

```bash
bash scripts/validation/data_validation.sh
```

## 2. Quick test

The repository provides two episodes and a synthetic GeoTIFF for testing
without the real dataset:

```bash
python -m applications.trajectory_generation.generate \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/trajectory_generation_test
```

Successful completion prints `Trajectory generation completed successfully`
and writes `annotations.json`, `summary.json`, and RGB images.

## 3. Production configuration

SatNav-v0.1 trajectories use:

```text
applications/episode_processing/configs/trajectory_generation.yaml
```

The main settings are a 10 m forward step, 15° turns, 448 × 448 RGB, 90° HFOV,
and at most 500 steps. Waypoint arrival radii are:

| Trajectory type | Arrival radius |
| --- | ---: |
| Boundary | 10 m |
| LandmarkSet | 3 m |
| Road | 10 m |

The 3 m LandmarkSet value is only the expert-generation waypoint tolerance;
it improves generated training paths. Do not change the production
configuration if reproducibility is required.

## 4. Generate the complete train split

Use the parallel entrypoint:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir "$SATNAV_TRAJECTORY_DIR"
```

The generator chooses workers from the scene count and CPU capacity, grouping
episodes by scene to reduce GeoTIFF memory use. Override with
`--num_workers N`; lower it when memory is limited. The parallel entrypoint is
recommended for the full release.

## 5. Resume

After an interruption, rerun the exact parallel command. No additional option
is required. Resume must use the same output directory as the initial run.

## 6. Output layout

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <scene>_satnav_<episode-index>/
        ├── .done
        ├── .annotation.json
        └── rgb/
            ├── 001.jpg
            └── ...
```

- `annotations.json`: final annotations consumed by training;
- `summary.json`: per-episode summaries in JSONL format;
- `images/`: RGB frames aligned with actions plus resume state.

Do not manually edit `.done` or `.annotation.json`; they are used for resume.

## 7. Confirm completion

The complete SatNav-v0.1 train split should end with:

```text
Success (incl. cached): 105164
Discarded (max steps): 0
Failed: 0
Generated annotations: 105164 / 105164 episodes
```

If counts are lower, rerun the same command to retry incomplete episodes. If
the problem persists, inspect the reported failure reason, scene files, and
production configuration.

## 8. Troubleshooting

### Why does generation raise `ModuleNotFoundError`?

Install application dependencies from the repository root:

```bash
python -m pip install -e '.[applications]'
```

### Why does parallel generation use too much memory?

Reduce concurrency with `--num_workers N`. Every worker owns a simulator,
scene cache, and image encoder; concurrency must fit memory and local disk
throughput.

### Why is generation slow?

Keep episodes, GeoTIFFs, and output on fast local storage. Preserve default
scene affinity so workers process the same scene consecutively and avoid
rebuilding GeoTIFF caches.

### Why can the generator not find episodes or scenes?

Reset `SATNAV_TRAIN_EPISODES_PATH` and `SATNAV_SCENES_DIR`, then run:

```bash
bash scripts/validation/data_validation.sh
```

Every logical `scene_id` must resolve to a matching GeoTIFF.

### Why were some episodes regenerated after a rerun?

The episode, generation configuration, or GeoTIFF content changed, so the old
cache could not be reused. Verify that all local paths still refer to the same
inputs. Regeneration is expected when inputs have genuinely changed.

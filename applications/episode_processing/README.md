# SatNav Episode Processing (Legacy)

> **Legacy application:** These Episode-processing commands are retained only
> for compatibility with the original per-city data-production workflow. New
> users should download the published SatNav-Episodes data directly and should
> not run this application. See [Episode data download](../../docs/DATA_DOWNLOAD.md),
> [map download](../../docs/APPLICATION_MAP_DOWNLOAD.md), and
> [trajectory generation](../../docs/APPLICATION_TRAJ_GENERATION.md).

The production configuration under `configs/trajectory_generation.yaml`
remains in use by `applications/trajectory_generation`; only the Episode
processing CLI described below is legacy.

This application inspects per-city `VLN_episodes.json` files and builds the
canonical train, `val_seen`, and `val_unseen` episode JSON files used by SatNav
consumers. It is data-production tooling and is not imported by the SatNav
simulator runtime.

From the SatNav repository root:

```bash
export SATNAV_DATA_ROOT=/path/to/satnav_datasets
python -m applications.episode_processing.inspect_data SatNav-v0.1
python -m applications.episode_processing.process_episodes SatNav-v0.1
python -m applications.episode_processing.run_all SatNav-v0.1
```

The expected source layout is
`$SATNAV_DATA_ROOT/<version>/data/<city>/VLN_episodes.json`. Generated files
use the canonical layout:

```text
episodes/train/all_episodes.json
episodes/eval/val_seen/all_episodes.json
episodes/eval/val_unseen/all_episodes.json
```

The production trajectory config is
`applications/episode_processing/configs/trajectory_generation.yaml`. Set
`SATNAV_TRAIN_EPISODES_PATH` and `SATNAV_SCENES_DIR`, then pass it to the
existing trajectory-generation application:

```bash
python -m applications.trajectory_generation.generate_parallel \
  --config applications/episode_processing/configs/trajectory_generation.yaml \
  --output_dir /path/to/trajectory_data \
  --num_workers 64
```

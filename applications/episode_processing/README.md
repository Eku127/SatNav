# SatNav Episode Processing

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
`$SATNAV_DATA_ROOT/<version>/data/<city>/VLN_episodes.json`; generated splits
are written below `$SATNAV_DATA_ROOT/<version>/episodes/`.

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

`SWIFTVLN_SATNAV_DATA_ROOT` remains accepted as a temporary compatibility
fallback for the dataset root; new SatNav workflows should use
`SATNAV_DATA_ROOT`.

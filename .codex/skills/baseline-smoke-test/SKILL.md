---
name: baseline-smoke-test
description: "Prepare and run SatNav baseline smoke tests on ver_260317 (0317) data. Use when the user wants a quick real-data validation of Seq2Seq or CMA training wiring, including correct dataset paths, vocab handling, tiny smoke subsets, and output locations."
---

# Baseline Smoke Test

Run a small but real SatNav baseline check on `ver_260317` data before launching a full training job.

Use this skill when the user asks to:

- smoke test `seq2seq` or `cma`
- verify that baseline training is wired to `0317` data instead of debug data
- generate a tiny real-data subset for a fast training sanity check

## Validated Findings

- `run.py` loads `BASE_TASK_CONFIG_PATH` before applying command-line `opts`. Do not rely on `BASE_TASK_CONFIG_PATH configs/satnav_task.yaml` as a CLI override; it is too late.
- `run.py` only resolves one `_base_` layer. A smoke config cannot inherit `configs/baselines/seq2seq.yaml` and expect `configs/default.yaml` to be merged automatically.
- The first validated smoke run with a `32`-episode subset only extracted `2` usable trajectories. That still trained, but it is too small to be a reliable smoke baseline.
- The validated repo-local smoke run with a `128`-episode subset extracted `26` usable trajectories and completed successfully.
- In the validated `128`-episode run, `2` episodes were skipped because the start position was too close to the map edge. This warning is acceptable for smoke testing.
- The first model run downloads `torchvision` ResNet-50 pretrained weights to the local torch cache. Expect a one-time startup delay.

## Repo Facts

- Baseline experiment files inherit [`configs/default.yaml`](/mnt/data1/home/jiangjiajun/workspace/SatNav/configs/default.yaml), which points to debug task config by default.
- Real-data smoke tests must override `BASE_TASK_CONFIG_PATH` to `configs/satnav_task.yaml`.
- `ver_260317` train data lives at `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/train/all_episodes.json`.
- Scene TIFs live at `/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes`.
- The 0317 dataset does not provide `instruction_vocab`, so smoke tests should build or reuse a vocab explicitly.
- Fastest smoke path: disable pretrained embeddings and use the full-train vocab size `5395`.

## Workflow

1. Read `README.md` and `.codex/CODEX_CONTEXT.md` if present.
2. Activate the runtime environment:

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
```

3. Build or reuse the full-train vocab once:

```bash
mkdir -p output/seq2seq_offline/artifacts/vocab

python -m satnav.utils.build_vocab \
  --dataset /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/train/all_episodes.json \
  --output /mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json
```

4. Create a tiny smoke subset from 0317 train data. Prefer the bundled script so the subset stays deterministic and scene-diverse. Use `128` episodes as the validated default, not `32`.

```bash
python /mnt/data1/home/jiangjiajun/workspace/SatNav/.codex/skills/baseline-smoke-test/scripts/make_smoke_subset.py \
  --src /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/train/all_episodes.json \
  --dst /mnt/data1/home/jiangjiajun/workspace/SatNav/output/smoke_260317/seq2seq_train_128.json \
  --count 128 \
  --per-scene 2
```

5. Run the model-specific smoke test through the validated smoke config file:

```bash
python run.py \
  --exp-config /mnt/data1/home/jiangjiajun/workspace/SatNav/.codex/skills/baseline-smoke-test/references/seq2seq-smoke-260317.yaml \
  --run-type train
```

## Seq2Seq Smoke Test

Use the validated config in [`references/seq2seq-smoke-260317.yaml`](/mnt/data1/home/jiangjiajun/workspace/SatNav/.codex/skills/baseline-smoke-test/references/seq2seq-smoke-260317.yaml). The goal is not quality. The goal is to verify:

- config inheritance is correct
- 0317 data and scenes load correctly
- vocab wiring works
- trainer reaches the first epoch and writes a checkpoint
- the run is actually based on `configs/satnav_task.yaml`, not `configs/debug_vln_task.yaml`

Smoke config contents are fixed in the YAML file because `BASE_TASK_CONFIG_PATH` must be present before `run.py` loads task config. The file sets:

- base experiment: `configs/default.yaml` plus the Seq2Seq-specific smoke overrides
- task config: `configs/satnav_task.yaml`
- smoke subset path: `output/smoke_260317/seq2seq_train_128.json`
- vocab path: `output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json`
- random instruction embeddings
- `IL.batch_size=1`, `IL.epochs=1`, `preload_size=2`
- smoke output directories under `output/*/seq2seq_smoke_260317`

Expected success signals:

- startup log shows `Loading task config: configs/satnav_task.yaml`
- `Creating dataset...` finishes on the small subset
- `Extracted ... trajectories` is comfortably above the `2` seen with the too-small 32-episode subset
- policy initializes successfully
- epoch `1/1` runs
- `output/checkpoints/seq2seq_smoke_260317/best.pth` 写出（在线 smoke 输出路径）

Validated real-run notes:

- A first-ever run may spend about a minute downloading `resnet50-0676ba61.pth`.
- On the original `32`-episode subset, training succeeded but only `2` trajectories were usable after GT extraction. Prefer `128` episodes for smoke.
- On the validated repo-local `128`-episode subset, the run logged `Extracted 26 trajectories`, trained `26` batches for one epoch, finished in about `39s`, and wrote `output/checkpoints/seq2seq_smoke_260317/best.pth` (online trainer output).
- `Warning: Episode 0/1 start position too close to map edge, skipping` appeared during dataset creation. This did not block the smoke run.

If the run is too slow even on the subset:

- reduce the subset only after confirming extracted trajectory count stays reasonable
- keep `IL.batch_size 1`
- keep `IL.RECOLLECT_TRAINER.preload_size 1` or `2`

If the run fails on embeddings:

- confirm `MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings false`
- confirm `DATASET.vocab_file` exists
- confirm `MODEL.INSTRUCTION_ENCODER.vocab_size 5395`

If startup still prints `Loading task config: configs/debug_vln_task.yaml`:

- do not try to fix it with CLI `BASE_TASK_CONFIG_PATH`
- run through the validated smoke YAML file in `references/`
- if a different task config is needed, create a dedicated experiment YAML instead of relying on `opts`

If startup errors with `TRAINER_NAME not specified in config`:

- check whether the smoke YAML has a `TRAINER_NAME` key
- if missing, add `TRAINER_NAME: recollect_trainer` (online smoke) or `TRAINER_NAME: offline_trainer` (offline smoke)
- keep the Seq2Seq-specific fields locally in the smoke YAML

## CMA

Reserved for a future smoke-test procedure.

When filling this section later:

- reuse the same full-train vocab
- reuse the same smoke subset creation flow
- switch `--exp-config` to `configs/baselines/cma.yaml`
- write outputs to `output/cma/checkpoints/cma_smoke_260317` and related directories

# StreamVLN baseline

This is SatNav's maintained StreamVLN integration. Training consumes SatNav
`trajectory_data`; evaluation implements `satnav.evaluation.PolicyAdapter` and
delegates deterministic episode selection, strided multi-rank sharding,
append-only JSONL, crash resume, and aggregation to the common
evaluator.

For the complete environment, SatNav-v0.1 data preparation, training, and
single/multi-GPU evaluation workflow, see the
[StreamVLN baseline guide](../../../docs/BASELINE_STREAMVLN.md).

The model implementation and weights remain external. The runtime is pinned
to
[`Eku127/StreamVLN@60476e81f4c01b29f1a51a7469f1cb4addbc1d62`](https://github.com/Eku127/StreamVLN/commit/60476e81f4c01b29f1a51a7469f1cb4addbc1d62).
The fork has no standalone `LICENSE` at that commit; its README states CC
BY-NC-SA 4.0. Read [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) before use.

## What is maintained here

```text
baselines/vlm/streamvln/
├── adapter.py             # PolicyAdapter, action chunks, KV/window state
├── dataset.py             # SatNav trajectory training dataset + validator
├── trainer.py             # thin wrapper around the pinned upstream trainer
├── evaluate.py            # common evaluator entrypoint
├── actions.py, history.py # dependency-light behavior primitives
├── configs/               # SatNav task, train defaults, DeepSpeed ZeRO-2
├── environment/conda.yml  # independent Python environment definition
├── requirements.txt       # pinned non-Torch Python dependencies
└── scripts/                # download, train, eval launchers
```

Importing `satnav` or `baselines.vlm` does not import Torch, Transformers, or
StreamVLN. Those dependencies are loaded only by this baseline's runtime
entrypoints.

This integration is supported from a SatNav source checkout installed with
`pip install -e .`. A standalone SatNav wheel does not contain the repository-
level launcher support files used by these commands.

## Fresh environment gate

The versions below preserve the migrated working stack; a new environment
still needs the explicit fresh-environment validation listed at the end.

```bash
conda env create -f baselines/vlm/streamvln/environment/conda.yml
conda activate satnav-streamvln

# Choose the PyTorch wheel matching the target CUDA runtime.
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install a FlashAttention wheel matching Python 3.9, Torch 2.5, CUDA, and ABI.
# Example: export STREAMVLN_FLASH_ATTN_WHEEL=/absolute/path/to/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
test -f "$STREAMVLN_FLASH_ATTN_WHEEL"
pip install "$STREAMVLN_FLASH_ATTN_WHEEL"

pip install -r baselines/vlm/streamvln/requirements.txt
# The official PyPI Linux x86_64 decord 0.6.0 wheel is named as Python-3
# compatible, but its inner WHEEL metadata incorrectly retains a CPython-3.6
# tag and its RECORD has a stale top_level.txt hash. This fail-closed helper
# verifies the exact official payload, dependency metadata, platform, WHEEL,
# and RECORD before correcting those two known internal metadata defects.
# It does not alter the installed top_level.txt or decoder payload bytes.
python baselines/vlm/streamvln/scripts/normalize_decord_wheel.py
pip install -e .
python -m pip check

# Keep the environment's console tools discoverable when launchers receive an
# absolute STREAMVLN_PYTHON path.  ZeRO-2 uses torch.optim.AdamW, so training
# does not JIT-compile FusedAdam against an unpinned host CUDA toolkit.
command -v ninja
```

Prepare the exact upstream source:

```bash
git clone https://github.com/Eku127/StreamVLN.git ../StreamVLN
git -C ../StreamVLN checkout 60476e81f4c01b29f1a51a7469f1cb4addbc1d62
```

Machine paths belong in the ignored overlay:

```bash
mkdir -p baselines/vlm/streamvln/.local
cp baselines/vlm/streamvln/local.env.example \
  baselines/vlm/streamvln/.local/env.sh
${EDITOR:-vi} baselines/vlm/streamvln/.local/env.sh
```

An explicitly exported variable wins over the value in `.local/env.sh`.
Tracked files contain no workstation path, credential, model, dataset, output,
or conda environment.

## Models

Download the official continue-training checkpoint:

```bash
bash baselines/vlm/streamvln/scripts/download.sh
```

Download the scratch base and local SigLIP tower (ModelScope is optional and
requires `pip install modelscope`):

```bash
bash baselines/vlm/streamvln/scripts/download.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2 --source modelscope
bash baselines/vlm/streamvln/scripts/download.sh \
  --repo google/siglip-so400m-patch14-384 --source modelscope
```

Use `--endpoint https://hf-mirror.com` when a Hugging Face mirror is needed.
`model/` is ignored and must never be committed.

## Dataset and loader validation

Training expects the public trajectory export:

```text
trajectory_data/
├── annotations.json
└── images/<episode>/rgb/*.jpg
```

Current annotations use `actions=[-1, ..., 0]`, where `-1` is INIT and the
terminal STOP is already present. The loader also accepts archived annotations
without INIT or STOP. Before canonical SatNav-v0.1 training, first run the
release-wide validator. Its CLI requires source episodes, generation config,
scenes, and full JPEG decoding. It checks all 105,164 annotations, source-episode
coverage, path/action/frame alignment, the complete `images/` tree for orphan
directories or unexpected files, fully loads every JPEG as RGB, and records an ordered
content digest:

```bash
python scripts/validation/validate_trajectory_output.py \
  --annotations /path/to/trajectory_data/annotations.json \
  --output-root /path/to/trajectory_data \
  --source-episodes /path/to/SatNav-v0.1/episodes/train/all_episodes.json \
  --generation-config configs/satnav_task.yaml \
  --scenes-dir /path/to/satnav_scenes \
  --expected-count 105164 --decode-images \
  --report output/baselines/vlm/streamvln/data_validation.json
```

Keep `--report` outside the validated trajectory root and outside every source,
config, and scenes input. This prevents the reporting step from modifying any
artifact whose digest or tree inventory was just checked.

Then validate the StreamVLN-specific chunking and 32-frame sampling contract
without loading weights:

```bash
python -m baselines.vlm.streamvln.dataset /path/to/trajectory_data \
  --num-frames 32 --strict-frames
```

Both historical launcher conventions work: passing `trajectory_data` with an
annotation video `images/<episode>`, or passing `trajectory_data/images` from an
older wrapper. Resolution is anchored to the configured data root, never to
the current working directory. Absolute or out-of-root annotation video paths
are rejected by default; archived data that truly requires them must opt in
explicitly with `--allow-external-video-paths` (validator) or
`SATNAV_ALLOW_EXTERNAL_VIDEO_PATHS=1` (training).

## Training

Continue from the official checkpoint (the default):

```bash
bash baselines/vlm/streamvln/scripts/train.sh continue \
  --trajectory-root /path/to/trajectory_data \
  --gpus 8
```

Start from LLaVA-Video:

```bash
bash baselines/vlm/streamvln/scripts/train.sh scratch \
  --trajectory-root /path/to/trajectory_data \
  --gpus 8
```

`configs/train.yaml` retains the migrated defaults: 32 frames, eight history
samples, four future actions, per-device batch 3, gradient accumulation 2,
one epoch, and learning rate `2e-5`. Common overrides are launcher options.
`configs/zero2.json` keeps ZeRO stage 2 while selecting PyTorch AdamW explicitly;
this avoids an otherwise implicit dependency on the host `nvcc` version.
For a bounded real-environment gate, add `--max-steps N`; this repository
migration intentionally does not run that GPU job automatically. The bounded
gate may also use `--dataloader-workers 0 --no-data-augmentation
--no-torch-compile` to remove worker/compile startup overhead without changing
the model, optimizer, or supervised sample.
For an explicitly capped dataset, `SATNAV_MAX_SAMPLES` must be at least one
global microbatch (`--gpus` × `--batch-size`) because training drops incomplete
batches. A two-step four-GPU, batch-one gate can use
`SATNAV_MAX_EPISODES=4 SATNAV_MAX_SAMPLES=8`.

The trainer imports the pinned upstream training implementation and replaces
only its supervised data module. It also overrides stale checkpoint vision
tower paths with the explicit local `--vision-tower`, without modifying the
checkpoint on disk.

Acceptance requires a fresh-process, non-dry-run model/evaluation load from the
saved directory and an inventory of its weight index, shards, optimizer, and
trainer state. This integration does not expose a standalone tensor-audit CLI,
so describe this as a full model reload, not as a tensor-by-tensor strict audit.

## Evaluation

Single GPU, canonical val-seen episodes:

```bash
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path /path/to/checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --vision-tower /path/to/siglip-so400m-patch14-384 \
  --episodes /path/to/SatNav-v0.1/episodes/eval/val_seen/all_episodes.json \
  --scenes-dir /path/to/scenes \
  --split val_seen \
  --gpus 1
```

Eight-rank evaluation uses the same command with `--gpus 8`. Do not pre-shard
the input. Each rank loads the complete episode list; the common evaluator
sorts stable `split::scene::episode` keys, applies `--offset/--limit`, then
assigns `selected[rank::world_size]`.

Use `--max-steps 5` to cap every episode at five primitive steps and select a
small deterministic subset:

```bash
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path /path/to/checkpoint \
  --episodes /path/to/SatNav-v0.1/episodes/eval/val_seen/all_episodes.json \
  --scenes-dir /path/to/scenes \
  --split val_seen --limit 5 --max-steps 5 --gpus 1
```

The default output is:

```text
output/baselines/vlm/streamvln/eval/<checkpoint>/<split>/
├── rank_00000/episodes.jsonl
├── rank_00000/done.json
└── summary.json
```

Add `--resume` after interruption. Resume skips Episode keys already present
in the current rank JSONL, including completed error records. It does not
compare the previous and current model, data, selection, seed, or world size;
use a new output directory whenever any of those conditions changes.

## Preserved model behavior and intentional integration changes

| Behavior | Maintained implementation |
| --- | --- |
| Symbol actions | `STOP`, `↑`, `←`, `→`; empty generation falls back to STOP |
| Action chunks | A generated chunk is queued and returned one primitive action per evaluator call |
| KV cache | Reused inside a window; cleared at the 32-frame boundary |
| Environment state | `model.reset_for_env(rank)` at episode reset and every window boundary |
| Cross-boundary leftovers | Queued actions survive the boundary; later history remains anchored at that boundary |
| History | Legacy global-frame sampling with eight history samples and current RGB |
| SatNav motion | 10 m forward, 15 degree turns |
| Evaluation storage | Replaced ad-hoc shared `result.jsonl` with rank-local durable JSONL, done markers, aggregation, and resume |
| Episode assignment | Replaced scene-local slicing with common stable-key strided sharding |
| Source/data paths | Replaced cwd-relative imports and `images/images` joins with validated roots |

The per-episode conjunction choice is seeded from SatNav's stable episode seed;
it is therefore independent of rank assignment. Model generation remains
greedy (`do_sample=False`, one beam), as in the migration input.

## Dependency-light checks

These checks require neither weights nor a GPU:

```bash
python -m pytest baselines/vlm/streamvln/tests -q
# Run only for the official PyPI Linux x86_64 decord 0.6.0 artifact. The
# helper deliberately rejects other platforms, versions, and repackaged payloads.
python baselines/vlm/streamvln/scripts/normalize_decord_wheel.py
python -m pip check
python -m baselines.vlm.streamvln.dataset --help
python -m baselines.vlm.streamvln.trainer --help
python -m baselines.vlm.streamvln.evaluate --help
bash baselines/vlm/streamvln/scripts/download.sh --help
bash baselines/vlm/streamvln/scripts/train.sh --help
bash baselines/vlm/streamvln/scripts/eval.sh --help
```

Still required before declaring the baseline freshly reproduced:

- build the independent Python 3.9 environment from scratch;
- import Torch, FlashAttention, DeepSpeed, Transformers, StreamVLN, and SatNav;
- run the dataset validator against the intended production trajectory export;
- perform a bounded optimizer-update training run and reload its saved model;
- perform single-GPU and multi-GPU online rollout against real scenes;
- verify the final aggregate with `scripts/evaluation/aggregate.py`.

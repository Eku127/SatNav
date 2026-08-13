# Uni-NaVid on SatNav

This is SatNav's maintained Uni-NaVid integration. It keeps model code and
weights external, adapts canonical `trajectory_data` without MP4 conversion,
and implements `satnav.evaluation.PolicyAdapter`. Importing `satnav` does not
import Torch or Uni-NaVid.

For the complete environment, SatNav-v0.1 data preparation, model download,
training, and single/multi-GPU evaluation workflow, see the
[Uni-NaVid baseline guide](../../../docs/BASELINE_UNINAVID.md).

The runtime is pinned to
[`jzhzhang/Uni-NaVid@79ef5ea3fea14c205342d1ab070563d84c7a966a`](https://github.com/jzhzhang/Uni-NaVid/commit/79ef5ea3fea14c205342d1ab070563d84c7a966a).
Accepted runs require a clean checkout at that exact revision. Read
[UPSTREAM.md](UPSTREAM.md), [NOTICE](NOTICE), and [LICENSE.upstream](LICENSE.upstream)
before redistributing code or weights.

## Fresh environment

Run from the SatNav repository root. First create and edit the ignored machine
overlay. Its fallback assignments preserve already exported shell values:

```bash
set -euo pipefail
mkdir -p baselines/vlm/uninavid/.local
cp baselines/vlm/uninavid/local.env.example \
  baselines/vlm/uninavid/.local/env.sh
${EDITOR:-vi} baselines/vlm/uninavid/.local/env.sh
source baselines/vlm/uninavid/scripts/_common.sh
```

Then create the independent Python 3.9 environment. Override the environment
name for a clean-room retry instead of replacing an existing environment:

```bash
export UNINAVID_ENV_NAME="${UNINAVID_ENV_NAME:-satnav-uninavid}"
conda env create --name "$UNINAVID_ENV_NAME" \
  --file baselines/vlm/uninavid/environment/conda.yml
conda activate "$UNINAVID_ENV_NAME"
export PYTHONNOUSERSITE=1
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$UNINAVID_PYTHON")"

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# The wheel must match CPython 3.9, Torch 2.5, CUDA 12, and the host ABI.
python -m pip install "$UNINAVID_FLASH_ATTN_WHEEL"
python -m pip install -r baselines/vlm/uninavid/requirements.txt
python -m pip install -e .
```

Do not install the upstream project as a Python distribution. The adapter
adds only the explicitly configured clean checkout to `sys.path`. It also
provides a narrow in-memory `decord` import stub because SatNav replaces the
upstream MP4 loader with a JPEG loader; attempting to instantiate the stubbed
`VideoReader` fails. This avoids the obsolete CPython-3.6-only PyPI metadata
and keeps `pip check` clean.

Explicitly exported variables take precedence over `.local/env.sh`, which
takes precedence over repository-relative defaults. No machine path belongs
in a tracked file.

Verify a fresh environment before using weights:

```bash
python -m pip check
python - <<'PY'
import accelerate, cv2, deepspeed, flash_attn, torch, transformers
from importlib import metadata
from satnav.evaluation import Evaluator, PolicyAdapter

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert cv2.__version__ == "4.13.0"
opencv_distributions = sorted(
    dist.metadata["Name"].lower()
    for dist in metadata.distributions()
    if dist.metadata["Name"].lower().startswith("opencv-python")
)
assert opencv_distributions == ["opencv-python"], opencv_distributions
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
print(Evaluator, PolicyAdapter)
PY
python - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16,
                requires_grad=True)
out = flash_attn_func(q, torch.randn_like(q), torch.randn_like(q))
out.float().sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(out).all() and torch.count_nonzero(q.grad)
print(torch.cuda.get_device_name(0), tuple(out.shape))
PY
```

## Source and model acquisition

The download helper creates or validates the pinned source checkout, resolves
the requested Hugging Face model revision to an immutable commit, and accepts
the official EVA file only at SHA-256
`99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f`:

```bash
bash baselines/vlm/uninavid/scripts/download.sh \
  --source-dir "$UNINAVID_REPO" \
  --model-root "$UNINAVID_MODEL_ROOT" \
  --model-revision main
```

Now verify that the acquired source is exactly pinned and clean:

```bash
python - <<'PY'
import os
from baselines.vlm.uninavid.bootstrap import bootstrap_uninavid

print(bootstrap_uninavid(os.environ["UNINAVID_REPO"]))
PY
```

The official model repository may require Hugging Face authentication. Model
snapshots, EVA, datasets, source checkouts, and outputs remain untracked.

Full checkpoints include language, multimodal projector, and embedded vision
weights. Check strict consumption in a fresh process:

```bash
python -m baselines.vlm.uninavid.checkpoint \
  --model-path /path/to/full-checkpoint \
  --uninavid-repo "$UNINAVID_REPO" \
  --eva-path "$UNINAVID_EVA" \
  --processor-path "$UNINAVID_PROCESSOR" \
  --device cuda:0
```

The command constructs the vision tower before Hugging Face reads the shards,
requires all 806 indexed tensors (291 language, 4 projector, 511 vision),
rejects missing/unexpected/mismatched keys, validates tokenizer IDs and
FlashAttention, and records exact checkpoint/EVA/processor identities. Stale
paths in an old checkpoint config are runtime-overridden with the configured
verified EVA and the processor from the pinned clean checkout; checkpoint
files are not edited.

## Canonical trajectory validation and training

Training consumes `trajectory_data/annotations.json` plus
`images/<episode>/rgb/*.jpg`. The launcher validates all canonical v0.1
annotations, exact window count, contained paths, required frames, sampled
JPEG decoding, and a full relative-path/size/content digest on rank 0 before
any model load:

```bash
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root /path/to/SatNav-v0.1/trajectory_data \
  --model-path /path/to/full-checkpoint \
  --eva-path /path/to/eva_vit_g.pth \
  --output-dir output/baselines/vlm/uninavid/train/v0.1
```

The default trains the language model and projector while keeping the vision
tower frozen. It preserves non-overlapping four-action windows, INIT removal,
STOP padding, all historical frames through each window start, the exact
numbered target, and upstream navigation augmentation. Missing/corrupt samples
and missing response boundaries fail; samples are never silently replaced.
Trainer owns the AdamW optimizer and declared cosine schedule; DeepSpeed's
ZeRO-1 configuration partitions that supplied optimizer without replacing
either component.

A deterministic two-update smoke can use:

```bash
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root /path/to/SatNav-v0.1/trajectory_data \
  --model-path /path/to/full-checkpoint \
  --eva-path /path/to/eva_vit_g.pth \
  --output-dir output/baselines/vlm/uninavid/train/smoke \
  --max-steps 2 --save-steps 1 --batch-size 1 \
  --gradient-accumulation 1 --warmup-ratio 0 \
  --dataloader-workers 0 --max-samples 16 --disable-augmentation
```

`training_manifest.json` fixes the complete source checkpoint, upstream
revision, verified external assets, full data snapshot, integration/config
files, distributed/global-batch shape, and sampling settings. Pass `--resume`
only for an existing matching run. Model, data, config, world-size, or sampling
changes are rejected before optimizer work. After training, the launcher
strictly reloads the complete saved checkpoint and exits nonzero unless at
least one language and projector tensor changed while every frozen vision
tensor remained bitwise identical to the source checkpoint.

## Evaluation

Single GPU five-step smoke:

```bash
bash baselines/vlm/uninavid/scripts/eval.sh \
  --gpus 1 \
  --model-path /path/to/trained-checkpoint \
  --eva-path /path/to/eva_vit_g.pth \
  --episodes /path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json \
  --scenes-dir /path/to/scenes \
  --split val_seen --limit 4 --max-steps 5 \
  --output-dir output/baselines/vlm/uninavid/eval/single \
  --fail-on-episode-error
```

Repeat with `--gpus 2` and a different output directory to validate strided
multi-rank execution. The selected episode union, stable seeds, action traces,
steps, and metrics must match the single-GPU run. Repeating an identical
command with `--resume` leaves completed rank JSONL bytes and mtimes unchanged.
Resume only skips Episode keys already recorded by that rank; use a new output
directory if checkpoint, EVA/processor, task config, episodes, generation,
selection, seed, or world size changes.

The policy preserves the `vicuna_v1` prompt, navigation-token injection order,
greedy generation, four-action word parser, STOP fallback, per-episode feature
cache reset, incremental `new_frames`, and queued primitive actions. SatNav's
common evaluator exclusively owns simulator stepping, durable results,
sharding, resume, and aggregation.

## Dependency-light checks

```bash
python -m pytest baselines/vlm/uninavid/tests -q
python -m baselines.vlm.uninavid.dataset --help
python -m baselines.vlm.uninavid.trainer --help
python -m baselines.vlm.uninavid.evaluate --help
python -m baselines.vlm.uninavid.checkpoint --help
bash baselines/vlm/uninavid/scripts/download.sh --help
bash baselines/vlm/uninavid/scripts/train.sh --help
bash baselines/vlm/uninavid/scripts/eval.sh --help
```

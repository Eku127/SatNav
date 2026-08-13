# NaVILA on SatNav

This directory is a small integration layer for NaVILA: it owns the SatNav
trajectory adapter, train/eval launchers, checkpoint identity checks, and a
`PolicyAdapter`. The model implementation and weights stay in an independent
environment and external checkout, so importing `satnav` or
`satnav.evaluation` never imports Torch, VILA, or NaVILA.

For the complete SatNav-v0.1 data preparation, model download, training, and
single-/multi-GPU evaluation workflow, see the
[NaVILA baseline guide](../../../docs/BASELINE_NAVILA.md).

The pinned model source is
[`AnjieCheng/NaVILA@76b98f233dd0fff05dfcd69435eec6740febff9d`](https://github.com/AnjieCheng/NaVILA/commit/76b98f233dd0fff05dfcd69435eec6740febff9d).
See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) before redistribution.

## Fresh environment (tested workflow)

Run these commands from the SatNav repository root. Use a new environment name
for every clean-room retry; do not reuse an existing NaVILA environment as a
fresh-install result.

```bash
export NAVILA_ENV_NAME="${NAVILA_ENV_NAME:-satnav-navila}"
conda env create \
  --name "${NAVILA_ENV_NAME}" \
  --file baselines/vlm/navila/environment/conda.yml
conda activate "${NAVILA_ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Use a trusted local wheel if the public URL is unavailable. Its Python,
# PyTorch, CUDA, CXX11 ABI, and architecture tags must match this environment.
export NAVILA_FLASH_ATTN_WHEEL="${NAVILA_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
python -m pip install "${NAVILA_FLASH_ATTN_WHEEL}"
python -m pip install -r baselines/vlm/navila/requirements.txt

export NAVILA_REPO="${NAVILA_REPO:-$(dirname "$PWD")/NaVILA}"
git clone https://github.com/AnjieCheng/NaVILA.git "${NAVILA_REPO}"
git -C "${NAVILA_REPO}" checkout --detach \
  76b98f233dd0fff05dfcd69435eec6740febff9d
git -C "${NAVILA_REPO}" rev-parse HEAD
# Expected: 76b98f233dd0fff05dfcd69435eec6740febff9d
python -m pip install --no-deps -e "${NAVILA_REPO}"
python -m pip install -e .
bash baselines/vlm/navila/scripts/patch_environment.sh
```

The FlashAttention 2.5.8 wheel above was built for CPython 3.10, Torch 2.3,
CUDA 12.x, and the old CXX11 ABI. A differently tagged local wheel is not an
equivalent installation. The patch script copies only the replacements shipped
by the pinned checkout into this environment's installed Transformers and
DeepSpeed packages; it does not modify the checkout.

Verify the environment before loading weights:

```bash
python - <<'PY'
import accelerate, deepspeed, flash_attn, torch, transformers
from baselines.vlm.navila.bootstrap import bootstrap_navila
from satnav.evaluation import Evaluator, PolicyAdapter

bootstrap_navila()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("accelerate", accelerate.__version__)
print("flash_attn", flash_attn.__version__)
print(Evaluator, PolicyAdapter)
PY
python - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(
    2, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True
)
out = flash_attn_func(q, torch.randn_like(q), torch.randn_like(q))
out.float().sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(out).all() and torch.count_nonzero(q.grad)
print(torch.cuda.get_device_name(0), tuple(out.shape))
PY
python -m pip check
python -m baselines.vlm.navila.dataset --help
python -m baselines.vlm.navila.trainer --help
python -m baselines.vlm.navila.evaluate --help
python -m baselines.vlm.navila.checkpoint --help
bash baselines/vlm/navila/scripts/train.sh --help
bash baselines/vlm/navila/scripts/eval.sh --help
```

To prove that a local composite checkpoint is complete, load the root plus all
LLM shards, tokenizer artifacts, vision tower, and multimodal projector in a
new process:

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path /path/to/navila-checkpoint \
  --navila-repo "${NAVILA_REPO}" \
  --device cuda:0
```

The command fails if any component has missing, unexpected, or mismatched
tensors. It also emits a root-relative, exact-content manifest that covers
every regular file at the model root and in the three checkpoint components.

## Local configuration and model acquisition

Copy the template to the ignored component overlay:

```bash
mkdir -p baselines/vlm/navila/.local
cp baselines/vlm/navila/local.env.example \
  baselines/vlm/navila/.local/env.sh
${EDITOR:-vi} baselines/vlm/navila/.local/env.sh
```

Precedence is explicit shell environment > component `.local/env.sh` > public
repository-relative defaults. Do not put checkpoint, dataset, scene, cache, or
conda paths in tracked files.

Official model snapshots are available only from Hugging Face:

- `a8cheng/navila-siglip-llama3-8b-v1.5-pretrain` (SFT starting point)
- `a8cheng/navila-llama3-8b-8f` (upstream evaluation model)

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo a8cheng/navila-llama3-8b-8f
```

## Canonical trajectory validation

NaVILA consumes `trajectory_data/annotations.json` and its `images/` tree.
Before training on SatNav-v0.1, run the complete schema/count/path preflight:

```bash
python -m baselines.vlm.navila.dataset /path/to/trajectory_data \
  --expected-episodes 105164 \
  --strict-frames \
  --hash-frame-content \
  --decode-samples 64 \
  --report /path/to/navila-data-validation.json
```

The validator checks all 105,164 annotations, INIT/action/STOP and `steps`
mapping, every expected RGB filename, and containment of every resolved path.
It decodes a deterministic sample of images and reports the exact annotations
SHA-256 plus a full relative-path/size/content SHA-256 identity for the
validated image tree. Absolute paths, traversal, video-directory symlink
escapes, and frame
symlink escapes are rejected. `--allow-external-paths` (validator) or
`SATNAV_ALLOW_EXTERNAL_TRAJECTORY_PATHS=1` (training) is an explicit legacy
opt-in and must not be used for canonical data.

## Training

```bash
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root /path/to/trajectory_data \
  --model-path /path/to/navila-llama3-8b-8f \
  --output-dir output/baselines/vlm/navila/train/continue-v0.1
```

The public default trains all model components with the pinned upstream
trainer. For a bounded hardware smoke that still performs a real nonzero
optimizer update and writes a complete checkpoint, use a zero warmup and tune
the projector:

```bash
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root /path/to/trajectory_data \
  --model-path /path/to/navila-llama3-8b-8f \
  --output-dir output/baselines/vlm/navila/train/smoke \
  --max-steps 2 --save-steps 1 --batch-size 1 \
  --dataloader-workers 0 --warmup-ratio 0 \
  --train-components projector --max-samples 16
```

`training_manifest.json` fixes the complete source checkpoint, upstream
revision, strict data-preflight artifact/image-tree identity, configs,
distributed/global-batch facts, arguments, sampling facts, and external-path
security setting.
Before every trainer launch, rank 0 freshly validates and content-hashes the
entire trajectory tree; other ranks wait for that launch's run-id-bound report.
An old preflight report is never accepted as the current training snapshot.
An existing output directory refuses resume if any immutable fact differs.
After a smoke, prove both strict reload and a concrete parameter change:

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path output/baselines/vlm/navila/train/smoke \
  --compare-model /path/to/navila-llama3-8b-8f \
  --navila-repo "${NAVILA_REPO}" \
  --device cuda:0
```

## Evaluation

All rollout state is model-owned, while episode selection, stable seed,
single/multi-rank sharding, result writing, resume, done markers, and
aggregation come from `satnav.evaluation`.

```bash
bash baselines/vlm/navila/scripts/eval.sh \
  --gpus 1 \
  --model-path /path/to/trained-navila \
  --split val_seen \
  --episodes /path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json \
  --scenes-dir /path/to/scenes \
  --output-dir output/baselines/vlm/navila/eval/single \
  --limit 4 --fail-on-episode-error
```

When testing the distributed launcher, repeat with `--gpus 2` and a different
output directory. The selected episode union, stable seeds, action traces,
steps, and metrics must equal the single-GPU run. Repeating the exact command
with `--resume` leaves completed JSONL bytes and mtimes unchanged. Changing a
model component, task YAML, episode artifact, generation option, world size,
selection, or seed requires a new output directory because resume only skips
Episode keys already recorded by that rank.

NaVILA's retained behavior is documented in [UPSTREAM.md](UPSTREAM.md): eight
sampled/padded frames, Llama-3 conversation formatting, deterministic greedy
generation, natural-language parsing, and queued 10 m forward / 15 degree turn
primitives. `compact` asks for one action word; `sentence` preserves the legacy
free-form response and distance/angle queue parsing.

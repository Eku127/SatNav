# OpenFly baseline

This directory is a self-contained OpenFly integration for SatNav. Model code
is loaded lazily from `baselines.vlm.openfly`; importing `satnav` does not import
Torch, Transformers, OpenFly, or an external checkout.

## Frozen source and supported contract

The bundled HF/Prismatic implementation is derived from OpenFly-Platform at
`c075075497a7122bad82f5b76b9be926ad5a81b3` (MIT) and behavior-audited against
the read-only SwiftVLN adapter at
`7b996303b05ca62791a2e5bf8f5fda729d6fdc07`. See [UPSTREAM.md](UPSTREAM.md),
[NOTICE](NOTICE), and [LICENSE.upstream](LICENSE.upstream).

The maintained model contract is intentionally narrow and fail-closed:

- `dinosiglip-vit-so-224px`, fused DINO/SigLIP, `grid_size=16`;
- exactly three RGB frames ordered current, previous, previous-2;
- Llama-2 prompt with bounded primitive-action history;
- `compact` (`stop|forward|left|right`) or original eight-dimensional action
  tokens, as declared by checkpoint metadata;
- local safetensors snapshots with complete content manifests and strict tensor
  reload; native `.pt` initialization is converted with collision checks into a
  content-addressed HF cache.

Directory names are never checkpoint identities. In particular, do not infer
scratch/continue provenance from legacy `openfly-satnav-*` basenames; the
integration hashes content and validates `config.json`, `backend_meta.json`,
`dataset_statistics.json`, tokenizer files, and every referenced weight shard.

## Fresh environment

Create a new environment explicitly; train/eval scripts never alter one:

```bash
bash baselines/vlm/openfly/scripts/bootstrap_env.sh satnav-openfly
mkdir -p baselines/vlm/openfly/.local
cp baselines/vlm/openfly/local.env.example \
  baselines/vlm/openfly/.local/env.sh
```

Edit only the ignored `.local/env.sh`. The frozen stack is Python 3.10,
Torch 2.3.0/cu121, torchvision 0.18.0/cu121, FlashAttention 2.5.8,
Transformers 4.48.1, Accelerate 0.33.0, DeepSpeed 0.14.4, timm 0.9.16,
tokenizers 0.21.1, and NumPy 1.26.4. The bootstrap ends with `pip check` and
version assertions. It ignores user-level Python packages and pip index
configuration, uses explicit package indexes, and builds the editable SatNav
package with the already-pinned build tools. Existing `openfly-baseline`/VILA
environments are not fresh-environment evidence.

## Canonical training data

Training requires both `trajectory_data/annotations.json` and the exact train
episode metadata. The join key is
`(logical_scene, trajectory_id, normalized_instruction)`; trajectory ID alone
is ambiguous and is rejected. Absolute paths, `..`, symlink escapes, missing or
extra frames, invalid INIT/STOP action schemas, duplicate joins, and unmatched
episodes all fail before model loading.

Validate the entire release, including every frame byte:

```bash
bash baselines/vlm/openfly/scripts/validate_data.sh \
  /data/SatNav-v0.1/trajectory_data /tmp/openfly_data_validation.json
```

Rank 0 repeats this full-content validation for every train launch. Other ranks
wait for the same run ID. Validation also writes a sibling
`data_validation.json.frames.sha256` index containing one raw 32-byte SHA-256
digest per frame, ordered first by annotation record and then by ascending frame
index. The report binds the index count, byte size, and SHA-256 identity.
Training snapshots that verified index in memory and hashes the exact JPEG bytes
that it decodes, so a frame-tree or index mutation after preflight fails closed.
`--max-episodes` consumes the corresponding prefix of the full-release index;
it does not weaken full-release validation. Sampling preserves the frozen
`hk7/fs3/stopx2/tail5` policy; caps require the explicit bounded-data path in
`scripts/train.sh`.

The 2026-08-10 CPU source-data gate joined all 105,164 records exactly and
produced 3,845,263 samples: STOP 210,328; forward 2,230,037; left 729,949;
right 674,949. Sample-level type counts were Boundary 1,312,126, LandmarkSet
1,199,694, and Road 1,333,443. The annotation SHA-256 was
`36a61c30c9dc6ee7a2dc378a07d1444554a47edf89c5889e1a89278777d2227a`.

## Train and resume

Continue from a complete HF snapshot:

```bash
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path /models/openfly-hf \
  --trajectory-root /data/SatNav-v0.1/trajectory_data \
  --gpus 1
```

Initialize from a native OpenFly run:

```bash
bash baselines/vlm/openfly/scripts/train.sh \
  --backend scratch \
  --model-path /runs/openfly-native \
  --processor-path /models/openfly-processor \
  --trajectory-root /data/SatNav-v0.1/trajectory_data \
  --gpus 2
```

A deterministic bounded smoke must save a real optimizer checkpoint:

```bash
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue --model-path /models/openfly-hf \
  --trajectory-root /data/SatNav-v0.1/trajectory_data \
  --max-episodes 2 --max-samples 4 --max-steps 1 --gpus 1
```

`--max-steps` defaults `save_steps` to 1 in the wrapper. Successful training
requires a new optimizer step, a finite nonzero parameter delta, a complete
`checkpoint-N` with scheduler/RNG/all DeepSpeed optimizer shards, strict final
reload, and immutable manifests. Resume rejects changed model/data/config/code,
world size, sampling policy, action format, empty optimizer/RNG payloads, or
incomplete checkpoint state:

```bash
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue --model-path /models/openfly-hf \
  --trajectory-root /data/SatNav-v0.1/trajectory_data \
  --output-dir /output/openfly/train/run-id --resume --gpus 2
```

The frozen default is full fine-tuning: vision backbone, language model, and
projector parameters must all remain trainable. Strict reload requires the
indexed and runtime state to have the same 982 keys and dtypes, split exactly
as vision 685, language 291, and projector 6, with no meta tensors. Both scratch
and continue training stream every tensor from the preserved source and trained
safetensors and require a finite nonzero update in each of the three components.

## Evaluate

All rollouts use `satnav.evaluation`: rank-local JSONL files, deterministic
sharding/seeds, done markers, Episode-key resume, and common aggregation.

```bash
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path /output/openfly/train/run-id \
  --split val_seen --gpus 1 --fail-on-episode-error
```

The canonical SatNav-v0.1 evaluation threshold is `LandmarkSet: 30.0`
(Boundary/Road 10.0).  The tighter `LandmarkSet: 3.0` radius belongs only to
offline expert-trajectory generation and must not be used for online success,
oracle-success, or SPL metrics. Model, tokenizer, processor, or parse failures
become explicit episode errors; they are never converted to STOP.

## CPU checks before GPU use

```bash
OPENFLY_PYTHON=/path/to/satnav-openfly/bin/python
"$OPENFLY_PYTHON" -m pytest -q baselines/vlm/openfly/tests
"$OPENFLY_PYTHON" -m ruff check baselines/vlm/openfly
bash -n baselines/vlm/openfly/scripts/*.sh
"$OPENFLY_PYTHON" -m baselines.vlm.openfly.trainer --help
"$OPENFLY_PYTHON" -m baselines.vlm.openfly.evaluate --help
"$OPENFLY_PYTHON" -m baselines.vlm.openfly.checkpoint --help
```

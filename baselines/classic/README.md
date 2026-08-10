# Classic baselines

SatNav maintains Random, ReferenceFollower, Seq2Seq, and CMA here.  Every
method uses the same public `satnav.Env` contract and generic evaluator result
format.  Importing `satnav` or `satnav.evaluation` does not import PyTorch;
Seq2Seq/CMA dependencies are loaded only when their packages are imported.

Maintained training/evaluation commands are source-checkout workflows: install
this repository in editable mode so top-level `configs/` and `scripts/` remain
available. A standalone SatNav wheel supports core imports and packaged example
resources, not these repository-level launchers.

## Local paths

Public configs use repository-relative example data.  For canonical v0.1 runs,
copy the template and edit the ignored file:

```bash
mkdir -p baselines/classic/.local
cp baselines/classic/local.env.example baselines/classic/.local/env.sh
```

An explicitly exported variable overrides the same variable in `.local/env.sh`.
No model, dataset, conda, or output absolute path belongs in tracked files.

## Methods and validation

| Method | Train | Eval |
| --- | --- | --- |
| Random | N/A | generic evaluator |
| ReferenceFollower | N/A | generic evaluator |
| Seq2Seq | offline imitation learning | generic evaluator |
| CMA | offline imitation learning | generic evaluator |

The repository quickstart configs remain runnable with example data.  The
canonical smoke workflow creates a deterministic v0.1 offline subset, performs
real optimizer updates, reloads the checkpoint strictly, and then evaluates a
fixed episode selection.  See `scripts/classic/` for the maintained commands.

Legacy imports under `satnav.models.baselines` and legacy checkpoints remain
supported.  Their module paths are compatibility surfaces, not core imports.

## Unified evaluation

All four methods use `python -m baselines.classic`, which delegates episode
selection, strided sharding, JSONL, resume, done markers, manifests, and
aggregation to `satnav.evaluation`.  The model-specific code only constructs a
policy adapter.

After configuring `baselines/classic/.local/env.sh`, canonical smoke commands
are:

```bash
bash scripts/classic/eval.sh random val_seen 8
bash scripts/classic/eval.sh reference_follower val_seen 8
bash scripts/classic/eval_parallel.sh random val_seen 2 0,1 8
```

Seq2Seq and CMA additionally require matching checkpoint, vocabulary, and
evaluation config variables from `local.env.example`.  Existing commands under
`scripts/seq2seq/eval*.sh` and `scripts/cma/eval*.sh` retain their arguments but
now forward to the same generic evaluator.

# OpenFly baseline

This directory provides SatNav's self-contained OpenFly integration. It owns
the pinned model runtime, trajectory adapter, training and evaluation
launchers, checkpoint conversion/loading, and the `PolicyAdapter` used by the
shared evaluator. Importing SatNav Core does not import Torch, Transformers,
or OpenFly model code.

For environment setup, model preparation, data validation, training, resume,
and single-/multi-GPU evaluation, see
[SatNav OpenFly Baseline](../../../docs/en-US/training/vlm/OPENFLY.md).

## Released checkpoints

- [Scratch](https://huggingface.co/Eku127/openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5)
- [Continue](https://huggingface.co/Eku127/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5)

Both checkpoints are part of the
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo).

## Upstream boundary

- Project: [`SHAILAB-IPEC/OpenFly-Platform`](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
- Commit: `c075075497a7122bad82f5b76b9be926ad5a81b3`
- Runtime: the adapted HF/Prismatic implementation lives in this directory;
  no external OpenFly checkout is required.

See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) for provenance and the
adaptation scope. A copy of the upstream MIT license is provided in
[LICENSE.upstream](LICENSE.upstream).

## Directory responsibilities

- `openfly_core/`, `native_core/`, and `backends/`: pinned runtime and native
  checkpoint conversion;
- `adapter.py`: three-frame observations, action history, and action conversion;
- `dataset.py`, `trainer.py`, and `checkpoint.py`: offline training and strict
  model validation;
- `evaluate.py`: shared-evaluator integration;
- `scripts/`: isolated environment, data validation, training, and evaluation;
- `local.env.example`: template for Git-ignored local paths.

## Developer entry points

```bash
python -m baselines.vlm.openfly.dataset --help
python -m baselines.vlm.openfly.trainer --help
python -m baselines.vlm.openfly.evaluate --help
python -m baselines.vlm.openfly.checkpoint --help
bash -n baselines/vlm/openfly/scripts/*.sh
```

User workflows are maintained only in the full baseline guide linked above.

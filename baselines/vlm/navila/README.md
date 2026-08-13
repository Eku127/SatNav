# NaVILA baseline

This directory is the integration layer between NaVILA and SatNav. It owns the
trajectory adapter, training and evaluation launchers, complete-checkpoint
loading, and the `PolicyAdapter` used by the shared evaluator. The NaVILA/VILA
model implementation and weights are not bundled with SatNav; runtime uses an
isolated environment and an external source checkout.

For environment setup, upstream checkout, model download, data validation,
training, and single-/multi-GPU evaluation, see
[SatNav NaVILA Baseline](../../../docs/en-US/training/vlm/NAVILA.md).

## Released checkpoints

- [Scratch](https://huggingface.co/Eku127/navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4)
- [Continue](https://huggingface.co/Eku127/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4)

Both checkpoints are part of the
[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo).

## Upstream boundary

- Repository: [`AnjieCheng/NaVILA`](https://github.com/AnjieCheng/NaVILA)
- Commit: `76b98f233dd0fff05dfcd69435eec6740febff9d`
- Runtime: `NAVILA_REPO` points to a separate checkout; model dependencies
  never enter SatNav Core.

See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) for provenance and retained
behavior. A copy of the upstream Apache-2.0 license is provided in
[LICENSE.upstream](LICENSE.upstream).

## Directory responsibilities

- `bootstrap.py` and `adapter.py`: pinned upstream loading and rollout state;
- `dataset.py`, `trainer.py`, and `checkpoint.py`: offline training and strict
  model validation;
- `evaluate.py`: shared-evaluator integration;
- `scripts/`: environment patching, downloads, training, and evaluation;
- `environment/` and `requirements.txt`: isolated dependencies;
- `local.env.example`: template for Git-ignored local paths.

## Developer entry points

```bash
python -m baselines.vlm.navila.dataset --help
python -m baselines.vlm.navila.trainer --help
python -m baselines.vlm.navila.evaluate --help
python -m baselines.vlm.navila.checkpoint --help
bash -n baselines/vlm/navila/scripts/*.sh
```

User workflows are maintained only in the full baseline guide linked above.

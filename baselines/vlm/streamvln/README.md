# StreamVLN baseline

This directory is the integration layer between StreamVLN and SatNav. It owns
the trajectory adapter, training and evaluation launchers, model loading, and
the `PolicyAdapter` used by the shared evaluator. The StreamVLN model
implementation and weights are not bundled with SatNav; runtime uses an
isolated environment and an external source checkout.

For environment setup, upstream checkout, model download, data validation,
training, and single-/multi-GPU evaluation, see
[SatNav StreamVLN Baseline](../../../docs/en-US/training/vlm/STREAMVLN.md).

## Upstream boundary

- Repository: [`Eku127/StreamVLN`](https://github.com/Eku127/StreamVLN)
- Commit: `60476e81f4c01b29f1a51a7469f1cb4addbc1d62`
- Runtime: `STREAMVLN_REPO` points to a separate checkout; model dependencies
  never enter SatNav Core.

See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) for provenance, retained
behavior, and licensing notes. The pinned upstream revision has no standalone
`LICENSE` file; verify its repository terms before use or redistribution.

## Directory responsibilities

- `adapter.py`: StreamVLN rollout state and SatNav action conversion;
- `dataset.py` and `trainer.py`: offline training and model saving;
- `evaluate.py`: shared-evaluator integration;
- `scripts/`: environment helpers, downloads, training, and evaluation;
- `environment/` and `requirements.txt`: isolated dependencies;
- `local.env.example`: template for Git-ignored local paths.

## Developer entry points

```bash
python -m baselines.vlm.streamvln.dataset --help
python -m baselines.vlm.streamvln.trainer --help
python -m baselines.vlm.streamvln.evaluate --help
bash -n baselines/vlm/streamvln/scripts/*.sh
```

User workflows are maintained only in the full baseline guide linked above.

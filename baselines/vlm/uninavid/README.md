# Uni-NaVid baseline

This directory is the integration layer between Uni-NaVid and SatNav. It owns
the trajectory adapter, training and evaluation launchers,
complete-checkpoint loading, and the `PolicyAdapter` used by the shared
evaluator. The Uni-NaVid model implementation and weights are not bundled with
SatNav; runtime uses an isolated environment and an external source checkout.

For environment setup, upstream checkout, model preparation, data validation,
training, and single-/multi-GPU evaluation, see
[SatNav Uni-NaVid Baseline](../../../docs/en-US/training/vlm/UNINAVID.md).

## Upstream boundary

- Repository: [`jzhzhang/Uni-NaVid`](https://github.com/jzhzhang/Uni-NaVid)
- Commit: `79ef5ea3fea14c205342d1ab070563d84c7a966a`
- Runtime: `UNINAVID_REPO` points to a separate checkout; model dependencies
  never enter SatNav Core.

See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) for provenance and retained
behavior. A copy of the upstream license is provided in
[LICENSE.upstream](LICENSE.upstream).

## Directory responsibilities

- `bootstrap.py` and `adapter.py`: pinned upstream loading and incremental
  navigation state;
- `dataset.py`, `trainer.py`, and `checkpoint.py`: offline training and strict
  model validation;
- `evaluate.py`: shared-evaluator integration;
- `scripts/`: downloads, training, and evaluation launchers;
- `environment/` and `requirements.txt`: isolated dependencies;
- `local.env.example`: template for Git-ignored local paths.

## Developer entry points

```bash
python -m baselines.vlm.uninavid.dataset --help
python -m baselines.vlm.uninavid.trainer --help
python -m baselines.vlm.uninavid.evaluate --help
python -m baselines.vlm.uninavid.checkpoint --help
bash -n baselines/vlm/uninavid/scripts/*.sh
```

User workflows are maintained only in the full baseline guide linked above.

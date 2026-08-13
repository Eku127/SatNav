# Classic baselines

This directory provides SatNav's Random, ReferenceFollower, Seq2Seq, and CMA
baselines. All four methods use the public `Env` API and the shared
`satnav.evaluation` interface. Only Seq2Seq and CMA require PyTorch, offline
training data, and checkpoints.

For complete data preparation, training, checkpoint validation, and
single-/multi-GPU evaluation instructions, see
[SatNav Classic Baselines](../../docs/en-US/training/CLASSIC.md). The general
training guide and tiny example are documented in
[Model Training](../../docs/en-US/training/README.md).

## Directory responsibilities

- `agents/`: online Random and ReferenceFollower policies;
- `seq2seq/` and `cma/`: checkpoint-to-`PolicyAdapter` factories;
- `common/`: shared checkpoint, visual, and instruction processing;
- `__main__.py`: common evaluation CLI for all four methods;
- `local.env.example`: template for Git-ignored local paths.

Public configurations use repository examples and relative paths. Store local
datasets, models, environments, and output paths in the Git-ignored
`baselines/classic/.local/env.sh`; never commit them to the repository.

## Developer entry points

```bash
python -m baselines.classic --help
bash scripts/quickstart_models.sh
python -m pytest -q tests/test_classic_adapters.py tests/test_classic_integration.py
```

Run training and evaluation from a SatNav source checkout installed in editable
mode. See the full baseline guide for compatibility details.

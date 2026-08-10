# External VLM baselines

StreamVLN, NaVILA, Uni-NaVid, and OpenFly use the same public SatNav Env and
evaluation result contract, but keep independent conda environments and model
code.  Nothing in this directory is imported by `satnav` itself.

All maintained VLM commands run from a SatNav source checkout installed in
editable mode. A standalone SatNav wheel does not install repository-level
benchmark manifests or shared launcher scripts and is not a supported VLM
train/eval surface.

Each baseline contains:

- the exact upstream repository and commit used by the adapter;
- an environment definition and a README tested from a fresh environment;
- `local.env.example` for untracked model/data/upstream-clone paths;
- model-specific dataset/training and `PolicyAdapter` code;
- train/eval scripts that delegate rollout, sharding, resume, result writing,
  and aggregation to `satnav.evaluation`.

Model weights, datasets, existing conda environments, `.local/`, `output/`, and
`results/` are never tracked.  A shell variable exported by the caller takes
precedence over the matching value loaded from a component `.local/env.sh`.

The four environments are intentionally not unified: their PyTorch,
Transformers, FlashAttention, and upstream-package constraints are mutually
different.  Core users should install SatNav without any VLM dependency.

Maintained integrations:

- [StreamVLN](streamvln/README.md): pinned external model code, SatNav
  trajectory training adapter, and common-evaluator online rollout.
- [NaVILA](navila/README.md): pinned VILA/NaVILA runtime, safe lazy trajectory
  adapter, and natural-language action rollout through the common evaluator.
- [Uni-NaVid](uninavid/README.md): strict full-checkpoint loading, windowed
  JPEG trajectory training, and incremental navigation-cache rollout.
- [OpenFly](openfly/README.md): bundled pinned Prismatic runtime, exact
  composite-key trajectory training, content-addressed checkpoints, and common
  evaluator rollout.

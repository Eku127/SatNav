# OpenFly source provenance

- Project: `SHAILAB-IPEC/OpenFly-Platform`
- URL: <https://github.com/SHAILAB-IPEC/OpenFly-Platform>
- Frozen revision: `c075075497a7122bad82f5b76b9be926ad5a81b3`
- Revision verified as both `HEAD` and `refs/heads/main` during the 2026-08-10
  source review.
- License: MIT; see `LICENSE.upstream`.
- Read-only adapter reference: SwiftVLN revision
  `7b996303b05ca62791a2e5bf8f5fda729d6fdc07`.

Bundled files `openfly_core/configuration_prismatic.py`,
`openfly_core/modeling_prismatic.py`, and
`openfly_core/processing_prismatic.py` originate from upstream
`train/extern/hf/`. `action_tokenizer.py` originates from
`train/model/action_tokenizer.py`. They are adapted for strict local loading,
the frozen three-frame SatNav contract, explicit supported-shape checks, and
safe checkpoint handling.

Runtime code imports only `baselines.vlm.openfly`; an external OpenFly checkout
is not added to `PYTHONPATH`. `OPENFLY_PLATFORM_REPO` or
`--comparison-repo` is optional and, when supplied, is accepted only at the
exact clean revision above.

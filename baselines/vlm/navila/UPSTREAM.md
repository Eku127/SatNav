# NaVILA upstream and adapter provenance

This directory maintains SatNav integration code; it does not vendor the
NaVILA/VILA implementation or model weights.

## Pinned model upstream

- Repository: [`AnjieCheng/NaVILA`](https://github.com/AnjieCheng/NaVILA)
- Commit: [`76b98f233dd0fff05dfcd69435eec6740febff9d`](https://github.com/AnjieCheng/NaVILA/commit/76b98f233dd0fff05dfcd69435eec6740febff9d)
- License: Apache License 2.0, reproduced in [LICENSE.upstream](LICENSE.upstream).
- Runtime relationship: `NAVILA_REPO` points at a separate pinned checkout.

## Adapter lineage

The migration input is
[`Eku127/SwiftVLN@7b996303b05ca62791a2e5bf8f5fda729d6fdc07`](https://github.com/Eku127/SwiftVLN/commit/7b996303b05ca62791a2e5bf8f5fda729d6fdc07),
directory `baseline/navila`. That adapter declared
`AnjieCheng/NaVILA@76b98f233dd0fff05dfcd69435eec6740febff9d` and
`Eku127/SatNav@c0c0e72ea4575b36d74a5e8f777942172978938e` as its
integration references.

The migration retains the model-specific eight-frame prompt, black-frame
padding/history sampling, natural-language action parsing, and 10 m/15 degree
primitive queues. Repository wiring, dataset path validation, model identity,
and rollout persistence were rewritten around SatNav's current public
`Env`/`PolicyAdapter` contract. See `NOTICE` and the pinned upstream license.

# StreamVLN upstream and adapter provenance

This directory maintains SatNav integration code; it does not vendor the
StreamVLN model implementation or model weights.

## Pinned model upstream

- Repository: [`Eku127/StreamVLN`](https://github.com/Eku127/StreamVLN)
- Commit: [`60476e81f4c01b29f1a51a7469f1cb4addbc1d62`](https://github.com/Eku127/StreamVLN/commit/60476e81f4c01b29f1a51a7469f1cb4addbc1d62)
- Runtime relationship: the checkout is loaded through `STREAMVLN_REPO`; its
  model implementation is not copied into SatNav.

At the pinned commit, the Eku127 fork has no standalone `LICENSE` file. Its
README states that the work is under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc-sa/4.0/). SatNav therefore
does not invent or copy a nonexistent upstream license file. Consult the
pinned repository and its history before redistribution or commercial use.

## SatNav adapter lineage

The maintained implementation was migrated from
[`Eku127/SwiftVLN@7b996303b05ca62791a2e5bf8f5fda729d6fdc07`](https://github.com/Eku127/SwiftVLN/commit/7b996303b05ca62791a2e5bf8f5fda729d6fdc07),
directory `baseline/streamvln`, whose declared SatNav integration reference was:

- Repository: [`Eku127/SatNav`](https://github.com/Eku127/SatNav)
- Commit: [`c0c0e72ea4575b36d74a5e8f777942172978938e`](https://github.com/Eku127/SatNav/commit/c0c0e72ea4575b36d74a5e8f777942172978938e)

The migration rewrites repository wiring around SatNav's current public
`Env`/`PolicyAdapter` contract. Model-specific behavior intentionally retained
from that adapter and the pinned StreamVLN code includes:

- `STOP`, `↑`, `←`, `→` symbolic action generation and chunked execution;
- Qwen conversation formatting and image/memory tokens;
- KV-cache reuse within a streaming window;
- `model.reset_for_env()` at episode and 32-frame window boundaries;
- eight-sample global observation history and four-action future chunks;
- SatNav trajectory annotation adaptation for 10 m forward / 15 degree turns.

See `NOTICE` for the concise third-party notice. This provenance document is
not a substitute for the upstream license terms.

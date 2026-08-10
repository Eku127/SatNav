# Upstream and retained behavior

The runtime is pinned to `jzhzhang/Uni-NaVid` revision
`79ef5ea3fea14c205342d1ab070563d84c7a966a`. Accepted runs require that exact,
clean Git checkout. The SatNav adapter was migrated from frozen SwiftVLN
revision `7b996303b05ca62791a2e5bf8f5fda729d6fdc07`.

Retained training semantics are non-overlapping four-action windows, dropping
the initial `-1`, STOP-padding the final window, historical frames from one
through `window_start + 1`, the exact numbered action target and prompt, and
the upstream navigation augmentation. Validation/smoke runs can explicitly
disable augmentation. Corrupt samples and missing label boundaries now fail
instead of being silently replaced or fully masked.

Retained rollout semantics are the `vicuna_v1` conversation, exact navigation
special-token injection order, deterministic greedy generation, four-action
regex queue, STOP fallback, and incremental `new_frames`/navigation-feature
cache. Episode selection, sharding, persistence, resume, and aggregation now
come from `satnav.evaluation`.

The frozen loader discarded 511 vision tensors embedded in a full checkpoint
and then loaded an external EVA file with `strict=False`. This integration
constructs the vision tower before Hugging Face loads checkpoint shards, so a
strict reload consumes all checkpoint tensors. The external EVA file is still
required to define/initialize the tower and its exact digest is part of every
run identity; accepted strict reloads reject missing, unexpected, or mismatched
checkpoint tensors.

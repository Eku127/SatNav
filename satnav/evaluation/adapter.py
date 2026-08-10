"""Small policy boundary used by the platform evaluator.

The evaluator deliberately knows nothing about model frameworks, tokenizers, or
recurrent state.  Those concerns belong in a policy adapter implementing the
three methods in :class:`PolicyAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class EpisodeContext:
    """Stable, per-episode information supplied to a policy adapter.

    ``seed`` is derived from the run seed and the stable episode key.  It is
    therefore independent of rank assignment and process launch order.
    ``environment`` is intentionally typed as ``Any`` so importing this module
    does not initialize SatNav's simulator stack.
    """

    episode: Any
    episode_key: str
    split: str
    episode_index: int
    rank: int
    world_size: int
    max_steps: int
    seed: int
    environment: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def simulator(self) -> Any:
        """Return the environment's public simulator handle."""

        return self.environment.simulator

    @property
    def agent_state(self) -> Any:
        """Return the environment's current public agent state."""

        return self.environment.agent_state


@dataclass(frozen=True)
class PolicyStep:
    """One primitive action selected by a policy.

    ``info`` is optional JSON-serializable policy diagnostics.  The evaluator
    never interprets it; it merely stores it with the action trace.
    """

    action: Any
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyAdapter(Protocol):
    """The complete interface required by the generic evaluator."""

    def reset(self, context: EpisodeContext) -> None:
        """Reset policy state for one episode."""

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        """Return the next primitive action."""

    def close(self) -> None:
        """Release policy-owned resources."""

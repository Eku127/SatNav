"""Deterministic random baseline for the generic SatNav evaluator."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Sequence

from satnav.evaluation import EpisodeContext, PolicyStep
from satnav.task.actions import Action


class RandomAdapter:
    """Sample primitive actions from an episode-local RNG.

    The evaluator derives ``EpisodeContext.seed`` from the run seed and stable
    episode key.  Reinitializing the RNG on every reset makes a given episode's
    trace independent of rank count, shard assignment, and resume boundaries.
    """

    def __init__(
        self,
        action_probabilities: Sequence[float] = (0.02, 0.68, 0.15, 0.15),
    ) -> None:
        probabilities = tuple(float(value) for value in action_probabilities)
        if len(probabilities) != len(Action.ALL_ACTIONS):
            raise ValueError(
                "action_probabilities must contain STOP, MOVE_FORWARD, "
                "TURN_LEFT, and TURN_RIGHT"
            )
        if any(value < 0.0 for value in probabilities):
            raise ValueError("action probabilities cannot be negative")
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"action probabilities must sum to 1.0, got {total:.12g}"
            )
        self.action_probabilities = probabilities
        self._rng: Optional[random.Random] = None
        self._episode_seed: Optional[int] = None

    def reset(self, context: EpisodeContext) -> None:
        self._episode_seed = int(context.seed)
        self._rng = random.Random(self._episode_seed)

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        del observation
        if self._rng is None:
            raise RuntimeError("reset() must be called before act()")
        action = self._rng.choices(
            Action.ALL_ACTIONS,
            weights=self.action_probabilities,
            k=1,
        )[0]
        return PolicyStep(
            action=action,
            info={"policy": "random", "episode_seed": self._episode_seed},
        )

    def close(self) -> None:
        self._rng = None
        self._episode_seed = None

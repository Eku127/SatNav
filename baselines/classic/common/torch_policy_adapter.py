"""Adapter for the legacy recurrent imitation-learning policy contract.

This module is deliberately outside :mod:`satnav`; importing it explicitly is
the opt-in boundary for PyTorch and classic model dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch

from satnav.evaluation import EpisodeContext, PolicyStep
from satnav.task.actions import Action


ObservationTransform = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class TorchILPolicyAdapter:
    """Run a SatNav ``ILPolicy`` inside the framework-neutral evaluator."""

    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        device: str = "cuda:0",
        observation_transform: Optional[ObservationTransform] = None,
        deterministic: bool = True,
        min_stop_steps: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.policy.eval()
        self.observation_transform = observation_transform
        self.deterministic = bool(deterministic)
        self.min_stop_steps = int(min_stop_steps)
        if self.min_stop_steps < 0:
            raise ValueError("min_stop_steps cannot be negative")

        self._rnn_state: Optional[torch.Tensor] = None
        self._previous_action: Optional[torch.Tensor] = None
        self._not_done_mask: Optional[torch.Tensor] = None
        self._steps = 0

    def reset(self, context: EpisodeContext) -> None:
        del context
        self._rnn_state = self.policy.net.get_initial_state(1, self.device)
        self._previous_action = torch.zeros(
            1, 1, device=self.device, dtype=torch.long
        )
        self._not_done_mask = torch.ones(
            1, 1, device=self.device, dtype=torch.uint8
        )
        self._steps = 0

    def _batch(self, observation: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        values = (
            self.observation_transform(observation)
            if self.observation_transform is not None
            else observation
        )
        batch: Dict[str, torch.Tensor] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                tensor = value
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                continue
            batch[name] = tensor.unsqueeze(0).to(self.device)
        return batch

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        if (
            self._rnn_state is None
            or self._previous_action is None
            or self._not_done_mask is None
        ):
            raise RuntimeError("reset() must be called before act()")

        with torch.no_grad():
            action, self._rnn_state = self.policy.act(
                self._batch(observation),
                self._rnn_state,
                self._previous_action,
                self._not_done_mask,
                deterministic=self.deterministic,
            )

        action_index = int(action.reshape(-1)[0].item())
        suppressed_stop = (
            action_index == Action.get_action_index(Action.STOP)
            and self._steps < self.min_stop_steps
        )
        if suppressed_stop:
            action_index = Action.get_action_index(Action.MOVE_FORWARD)

        self._previous_action = torch.tensor(
            [[action_index]], device=self.device, dtype=torch.long
        )
        self._steps += 1
        return PolicyStep(
            action=Action.get_action_from_index(action_index),
            info={
                "policy_action_index": action_index,
                "suppressed_stop": suppressed_stop,
            },
        )

    def close(self) -> None:
        self._rnn_state = None
        self._previous_action = None
        self._not_done_mask = None

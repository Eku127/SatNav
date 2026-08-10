"""CMA checkpoint/config/vocabulary to evaluator adapter factory."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from baselines.classic.common.neural_factory import (
    NeuralAdapterBundle,
    build_torch_il_adapter,
)
from satnav.models.baselines.cma_policy import CMAPolicy


def build_cma_adapter(
    config: DictConfig,
    *,
    observation_space: Any,
    action_space: Any,
    checkpoint_path: str,
    vocab_path: str,
    device: str = "cuda:0",
    deterministic: bool = True,
    min_stop_steps: int = 0,
) -> NeuralAdapterBundle:
    return build_torch_il_adapter(
        policy_class=CMAPolicy,
        method="cma",
        config=config,
        observation_space=observation_space,
        action_space=action_space,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
        deterministic=deterministic,
        min_stop_steps=min_stop_steps,
    )

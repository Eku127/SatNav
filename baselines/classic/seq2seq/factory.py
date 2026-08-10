"""Seq2Seq checkpoint/config/vocabulary to evaluator adapter factory."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from baselines.classic.common.neural_factory import (
    NeuralAdapterBundle,
    build_torch_il_adapter,
)
from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy


def build_seq2seq_adapter(
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
        policy_class=Seq2SeqPolicy,
        method="seq2seq",
        config=config,
        observation_space=observation_space,
        action_space=action_space,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
        deterministic=deterministic,
        min_stop_steps=min_stop_steps,
    )

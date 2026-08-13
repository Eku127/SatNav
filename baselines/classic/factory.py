"""Lazy four-method adapter factory for the generic SatNav evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from baselines.classic.agents import RandomAdapter, ReferenceFollowerAdapter


METHOD_ALIASES = {
    "random": "random",
    "reference": "reference_follower",
    "reference_follower": "reference_follower",
    "seq2seq": "seq2seq",
    "cma": "cma",
}


@dataclass(frozen=True)
class ClassicAdapterBundle:
    adapter: Any
    policy_id: str
    config: DictConfig


def normalize_method(method: str) -> str:
    try:
        return METHOD_ALIASES[str(method).lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown classic method {method!r}; choose from {sorted(METHOD_ALIASES)}"
        ) from error


def _success_distances(config: DictConfig) -> Mapping[str, float]:
    value = OmegaConf.select(config, "TASK.SUCCESS_DISTANCE", default={}) or {}
    if isinstance(value, (int, float)):
        return {"DEFAULT": float(value)}
    resolved = OmegaConf.to_container(value, resolve=True)
    return {str(name): float(distance) for name, distance in resolved.items()}


def build_classic_adapter(
    method: str,
    config: DictConfig,
    *,
    observation_space: Any = None,
    action_space: Any = None,
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    device: str = "cuda:0",
    deterministic: bool = True,
    min_stop_steps: int = 0,
) -> ClassicAdapterBundle:
    """Build one of the four adapters without eagerly importing Torch."""

    normalized = normalize_method(method)
    configured_method = OmegaConf.select(
        config, "MODEL.policy_name", default=None
    )
    if configured_method is not None:
        try:
            configured_method = normalize_method(str(configured_method))
        except ValueError as error:
            raise ValueError(
                f"config MODEL.policy_name={configured_method!r} is not a classic method"
            ) from error
        if configured_method != normalized:
            raise ValueError(
                f"config policy_name={configured_method!r} does not match {normalized!r}"
            )
    if normalized == "random":
        probabilities = OmegaConf.select(
            config,
            "MODEL.RANDOM_AGENT.action_probs",
            default=(0.02, 0.68, 0.15, 0.15),
        )
        adapter = RandomAdapter(probabilities)
        return ClassicAdapterBundle(
            adapter=adapter,
            policy_id="random",
            config=config,
        )

    if normalized == "reference_follower":
        turn_angle = float(
            OmegaConf.select(config, "SIMULATOR.TURN_ANGLE", default=15.0)
        )
        distances = _success_distances(config)
        adapter = ReferenceFollowerAdapter(
            goal_radius=float(distances.get("DEFAULT", 10.0)),
            turn_angle=turn_angle,
            success_distances=distances,
        )
        return ClassicAdapterBundle(
            adapter=adapter,
            policy_id="reference_follower",
            config=config,
        )

    if observation_space is None or action_space is None:
        raise ValueError(f"{normalized} requires observation_space and action_space")
    resolved_checkpoint = checkpoint_path or OmegaConf.select(
        config, "EVAL.CKPT_PATH", default=None
    )
    resolved_vocab = vocab_path or OmegaConf.select(
        config, "DATASET.vocab_file", default=None
    )
    if not resolved_checkpoint:
        raise ValueError(f"{normalized} requires a checkpoint path")
    if not resolved_vocab:
        raise ValueError(f"{normalized} requires a vocabulary path")

    if normalized == "seq2seq":
        from baselines.classic.seq2seq.factory import build_seq2seq_adapter

        neural = build_seq2seq_adapter(
            config,
            observation_space=observation_space,
            action_space=action_space,
            checkpoint_path=str(resolved_checkpoint),
            vocab_path=str(resolved_vocab),
            device=device,
            deterministic=deterministic,
            min_stop_steps=min_stop_steps,
        )
    else:
        from baselines.classic.cma.factory import build_cma_adapter

        neural = build_cma_adapter(
            config,
            observation_space=observation_space,
            action_space=action_space,
            checkpoint_path=str(resolved_checkpoint),
            vocab_path=str(resolved_vocab),
            device=device,
            deterministic=deterministic,
            min_stop_steps=min_stop_steps,
        )
    return ClassicAdapterBundle(
        adapter=neural.adapter,
        policy_id=normalized,
        config=neural.config,
    )

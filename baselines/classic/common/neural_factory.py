"""Strict construction shared by Seq2Seq and CMA evaluator adapters."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Type

from omegaconf import DictConfig, OmegaConf

from baselines.classic.common.checkpoints import load_training_checkpoint
from baselines.classic.common.config import resolve_path
from baselines.classic.common.observations import InstructionObservationTransform
from baselines.classic.common.torch_policy_adapter import TorchILPolicyAdapter
from satnav.utils.build_vocab import VocabDict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_config(config: DictConfig, vocab_size: int) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    configured_size = OmegaConf.select(
        copied, "MODEL.INSTRUCTION_ENCODER.vocab_size", default=None
    )
    if configured_size is None:
        OmegaConf.update(
            copied,
            "MODEL.INSTRUCTION_ENCODER.vocab_size",
            int(vocab_size),
            merge=False,
        )
    elif int(configured_size) != int(vocab_size):
        raise ValueError(
            "vocabulary size does not match MODEL.INSTRUCTION_ENCODER.vocab_size: "
            f"{vocab_size} != {configured_size}"
        )
    return copied


def _validate_pretrained_embeddings(config: DictConfig, vocab_size: int) -> None:
    encoder = config.MODEL.INSTRUCTION_ENCODER
    if not bool(getattr(encoder, "use_pretrained_embeddings", False)):
        return
    embedding_path = resolve_path(str(encoder.embedding_file))
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"pretrained embedding file does not exist: {embedding_path}"
        )
    with gzip.open(str(embedding_path), "rt", encoding="utf-8") as handle:
        embeddings = json.load(handle)
    if not isinstance(embeddings, list) or len(embeddings) != vocab_size:
        raise ValueError(
            "pretrained embedding row count does not match vocabulary size: "
            f"{len(embeddings) if isinstance(embeddings, list) else 'invalid'} "
            f"!= {vocab_size}"
        )
    embedding_size = int(encoder.embedding_size)
    if any(not isinstance(row, list) or len(row) != embedding_size for row in embeddings):
        raise ValueError(
            f"pretrained embeddings must have exactly {embedding_size} columns"
        )
    # Make model construction independent of the caller's current directory.
    encoder.embedding_file = str(embedding_path)


@dataclass(frozen=True)
class NeuralAdapterBundle:
    adapter: Any
    policy_metadata: Mapping[str, Any]
    config: DictConfig


def build_torch_il_adapter(
    *,
    policy_class: Type[Any],
    method: str,
    config: DictConfig,
    observation_space: Any,
    action_space: Any,
    checkpoint_path: str,
    vocab_path: str,
    device: str,
    deterministic: bool = True,
    min_stop_steps: int = 0,
) -> NeuralAdapterBundle:
    """Instantiate a legacy policy, strictly reload it, and adapt it."""

    resolved_checkpoint = resolve_path(checkpoint_path)
    resolved_vocab = resolve_path(vocab_path)
    if not resolved_vocab.is_file():
        raise FileNotFoundError(f"vocabulary file does not exist: {resolved_vocab}")
    vocab = VocabDict.load(str(resolved_vocab))
    resolved_config = _validated_config(config, len(vocab))
    configured_method = str(
        OmegaConf.select(resolved_config, "MODEL.policy_name", default=method)
    ).lower()
    if configured_method != method:
        raise ValueError(
            f"config policy_name={configured_method!r} does not match {method!r}"
        )
    _validate_pretrained_embeddings(resolved_config, len(vocab))

    policy = policy_class.from_config(
        resolved_config, observation_space, action_space
    )
    checkpoint = load_training_checkpoint(
        str(resolved_checkpoint), policy, device=device
    )
    max_instruction_length = OmegaConf.select(
        resolved_config, "EVAL.MAX_INSTRUCTION_LENGTH", default=None
    )
    transform = InstructionObservationTransform(
        vocab,
        max_length=(
            None
            if max_instruction_length is None
            else int(max_instruction_length)
        ),
    )
    adapter = TorchILPolicyAdapter(
        policy,
        device=device,
        observation_transform=transform,
        deterministic=deterministic,
        min_stop_steps=min_stop_steps,
    )
    metadata = {
        "method": method,
        "checkpoint_sha256": file_sha256(resolved_checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_step_id": int(checkpoint["step_id"]),
        "checkpoint_loss": float(checkpoint["loss"]),
        "vocab_sha256": file_sha256(resolved_vocab),
        "vocab_size": len(vocab),
    }
    return NeuralAdapterBundle(
        adapter=adapter, policy_metadata=metadata, config=resolved_config
    )

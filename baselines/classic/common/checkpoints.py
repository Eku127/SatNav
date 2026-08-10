"""Strict checkpoint helpers used by classic smoke and evaluation commands."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch


REQUIRED_TRAIN_CHECKPOINT_KEYS = {
    "config",
    "epoch",
    "loss",
    "optim_state",
    "state_dict",
    "step_id",
}


def read_training_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Read and validate the complete trainer checkpoint envelope."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # Older supported Torch releases lack weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} is not a mapping")
    checkpoint = dict(checkpoint)
    missing = REQUIRED_TRAIN_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} is missing keys: {sorted(missing)}"
        )
    if not isinstance(checkpoint["state_dict"], Mapping):
        raise ValueError("checkpoint state_dict is not a mapping")
    if not isinstance(checkpoint["optim_state"], Mapping):
        raise ValueError("checkpoint optim_state is not a mapping")
    if not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config is not a mapping")
    try:
        step_id = int(checkpoint["step_id"])
        epoch = int(checkpoint["epoch"])
        loss = float(checkpoint["loss"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint epoch/step_id/loss metadata is invalid") from error
    if step_id < 1:
        raise ValueError("training checkpoint contains no optimizer step")
    if epoch < 0:
        raise ValueError("training checkpoint epoch cannot be negative")
    if not torch.isfinite(torch.as_tensor(loss)):
        raise ValueError("training checkpoint loss is not finite")
    return checkpoint


def load_training_checkpoint(
    path: str,
    policy: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a complete training checkpoint and reject partial artifacts."""
    checkpoint = read_training_checkpoint(path, device=device)
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim_state"])
    return checkpoint

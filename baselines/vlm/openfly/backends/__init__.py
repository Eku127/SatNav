from __future__ import annotations

from baselines.vlm.openfly.backends.base import (
    CONTINUE_BACKEND,
    SCRATCH_BACKEND,
    HF_BACKEND,
    NATIVE_BACKEND,
    TrainBackendArtifacts,
    patch_accelerate_optimizer_train_eval,
    resolve_backend,
)
from baselines.vlm.openfly.backends.hf_backend import build_hf_train_backend
from baselines.vlm.openfly.backends.native_backend import build_native_train_backend


def build_train_backend(model_args, data_args, training_args) -> TrainBackendArtifacts:
    backend = resolve_backend(model_args.backend)
    if backend == CONTINUE_BACKEND:
        return build_hf_train_backend(model_args, data_args, training_args)
    if backend == SCRATCH_BACKEND:
        return build_native_train_backend(model_args, data_args, training_args)
    raise ValueError(f"Unsupported OpenFly backend: {backend}")


__all__ = [
    "CONTINUE_BACKEND",
    "SCRATCH_BACKEND",
    "HF_BACKEND",
    "NATIVE_BACKEND",
    "TrainBackendArtifacts",
    "build_train_backend",
    "patch_accelerate_optimizer_train_eval",
    "resolve_backend",
]

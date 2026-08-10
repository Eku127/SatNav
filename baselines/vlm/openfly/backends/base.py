from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from baselines.vlm.openfly.artifacts import (
    openfly_model_identity,
    processor_identity,
)


CONTINUE_BACKEND = "continue"
SCRATCH_BACKEND = "scratch"
SUPPORTED_BACKENDS = {CONTINUE_BACKEND, SCRATCH_BACKEND}

# Legacy aliases kept for any external references
HF_BACKEND = CONTINUE_BACKEND
NATIVE_BACKEND = SCRATCH_BACKEND
LEGACY_BACKEND_ALIASES = {
    "hf": CONTINUE_BACKEND,
    "native": SCRATCH_BACKEND,
    HF_BACKEND: CONTINUE_BACKEND,
    NATIVE_BACKEND: SCRATCH_BACKEND,
}


def resolve_backend(requested: str) -> str:
    requested = LEGACY_BACKEND_ALIASES.get(requested, requested)
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported OpenFly backend: {requested}")
    return requested


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            return
        except Exception:
            pass
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(
        language_model, "gradient_checkpointing_enable"
    ):
        language_model.gradient_checkpointing_enable()


def enforce_full_finetune_contract(model) -> dict[str, Any]:
    """Fail unless every model parameter belongs to and trains with one component."""

    prefixes = {
        "vision": "vision_backbone.",
        "language": "language_model.",
        "projector": "projector.",
    }
    report = {
        component: {
            "parameter_tensors": 0,
            "trainable_parameter_tensors": 0,
            "parameters": 0,
            "trainable_parameters": 0,
        }
        for component in prefixes
    }
    frozen = []
    unknown = []
    for name, parameter in model.named_parameters():
        component = next(
            (
                candidate
                for candidate, prefix in prefixes.items()
                if name.startswith(prefix)
            ),
            None,
        )
        if component is None:
            unknown.append(name)
            continue
        values = report[component]
        values["parameter_tensors"] += 1
        values["parameters"] += int(parameter.numel())
        if parameter.requires_grad:
            values["trainable_parameter_tensors"] += 1
            values["trainable_parameters"] += int(parameter.numel())
        else:
            frozen.append(name)
    empty_components = [
        component
        for component, values in report.items()
        if values["parameter_tensors"] == 0
    ]
    if frozen or unknown or empty_components:
        raise RuntimeError(
            "OpenFly full-finetune parameter contract failed: "
            + json.dumps(
                {
                    "frozen_parameters": frozen,
                    "unknown_parameters": unknown,
                    "empty_components": empty_components,
                    "components": report,
                },
                sort_keys=True,
            )
        )
    return {
        "contract": "full-finetune",
        "components": report,
        "frozen_parameters": [],
        "unknown_parameters": [],
    }


def reject_non_strict_loading(info: dict[str, Any], context: str) -> None:
    errors = {
        name: list(info.get(name, ()))
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if info.get(name)
    }
    if errors:
        raise RuntimeError(f"{context} did not load strictly: {errors}")


def patch_accelerate_optimizer_train_eval() -> None:
    import accelerate.optimizer

    def _safe_train(self):
        train_fn = getattr(self.optimizer, "train", None)
        if callable(train_fn):
            return train_fn()
        return None

    def _safe_eval(self):
        eval_fn = getattr(self.optimizer, "eval", None)
        if callable(eval_fn):
            return eval_fn()
        return None

    accelerate.optimizer.AcceleratedOptimizer.train = _safe_train
    accelerate.optimizer.AcceleratedOptimizer.eval = _safe_eval


@dataclass
class TrainBackendArtifacts:
    backend_name: str
    processor: Any
    model: Any
    data_collator: Any
    metadata: dict[str, Any]
    backend_meta: dict[str, Any] = field(default_factory=dict)
    comparison_model_path: str = ""


def build_default_backend_meta(
    backend_name: str,
    model_name_or_path: str,
    processor_source: str,
    action_format: str,
    grid_size: int,
) -> dict[str, Any]:
    model_identity = openfly_model_identity(Path(model_name_or_path))
    processor = processor_identity(Path(processor_source))
    return {
        "backend": resolve_backend(backend_name),
        "source_model_identity": model_identity,
        "comparison_model_identity": model_identity,
        "processor_identity": processor,
        "action_format": action_format,
        "grid_size": int(grid_size),
    }

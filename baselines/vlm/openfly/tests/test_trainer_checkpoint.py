from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from baselines.vlm.openfly.checkpoint import (
    EXPECTED_COMPONENT_TENSOR_COUNTS,
    EXPECTED_TENSOR_COUNT,
    _strict_state_audit,
    _weight_map,
    parameter_delta,
)
from baselines.vlm.openfly.artifacts import openfly_model_identity
from baselines.vlm.openfly.trainer import (
    _parameter_delta,
    _checkpoint_content_manifest,
    _stable_validation,
    _trainer_class,
    _validate_output_paths,
    _validate_output_mode,
    _validate_effective,
    audit_resumable_checkpoint,
    main as trainer_main,
    resolve_resume_checkpoint,
)
from satnav.evaluation.manifest import ensure_manifest


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model_files(
    root: Path, tensor: torch.Tensor | Mapping[str, torch.Tensor] | None = None
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _json(
        root / "config.json",
        {
            "model_type": "openvla",
            "architectures": ["OpenVLAForActionPrediction"],
            "action_format": "compact",
            "grid_size": 16,
            "use_cache": False,
        },
    )
    _json(root / "preprocessor_config.json", {})
    _json(root / "tokenizer_config.json", {})
    _json(root / "tokenizer.json", {})
    _json(
        root / "backend_meta.json", {"backend": "continue", "action_format": "compact"}
    )
    _json(
        root / "dataset_statistics.json", {"action_format": "compact", "num_samples": 1}
    )
    if tensor is None:
        (root / "model.safetensors").write_bytes(b"model")
    elif isinstance(tensor, Mapping):
        save_file(dict(tensor), root / "model.safetensors")
    else:
        save_file({"projector.weight": tensor}, root / "model.safetensors")


def _contract_tensors() -> dict[str, torch.Tensor]:
    prefixes = {
        "vision": "vision_backbone",
        "language": "language_model",
        "projector": "projector",
    }
    return {
        f"{prefixes[component]}.tensor_{index:04d}.weight": torch.zeros(1)
        for component, count in EXPECTED_COMPONENT_TENSOR_COUNTS.items()
        for index in range(count)
    }


def _add_lineage(before: Path, after: Path, *, source: Path | None = None) -> None:
    before_identity = openfly_model_identity(before)
    source_identity = openfly_model_identity(source or before)
    ensure_manifest(
        after / "training_manifest.json",
        {
            "manifest_type": "openfly_training",
            "source_model": source_identity,
            "comparison_model": before_identity,
            "configuration": {"distributed": {"world_size": 1}},
        },
    )


def _checkpoint(
    output: Path, step: int = 1, *, world_size: int = 1, deepspeed: bool = False
) -> Path:
    output.mkdir()
    payload = {
        "manifest_type": "openfly_training",
        "configuration": {"distributed": {"world_size": world_size}},
    }
    root_manifest = ensure_manifest(output / "training_manifest.json", payload)
    checkpoint = output / f"checkpoint-{step}"
    _model_files(checkpoint)
    ensure_manifest(checkpoint / "training_manifest.json", payload)
    _json(checkpoint / "trainer_state.json", {"global_step": step})
    torch.save(
        {"last_epoch": step, "_step_count": step + 1},
        checkpoint / "scheduler.pt",
    )
    if deepspeed:
        global_step = checkpoint / f"global_step{step}"
        global_step.mkdir()
        for rank in range(world_size):
            torch.save(
                {
                    "optimizer_state_dict": {
                        "zero_stage": 2,
                        "partition_count": world_size,
                        "base_optimizer_state": {
                            "state": {
                                0: {
                                    "step": torch.tensor(1.0),
                                    "exp_avg": torch.zeros(1),
                                }
                            },
                            "param_groups": [{"lr": 1e-5, "params": [0]}],
                        },
                        "single_partition_of_fp32_groups": [torch.zeros(1)],
                    }
                },
                global_step / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
            )
        torch.save(
            {
                "global_steps": step,
                "dp_world_size": world_size,
                "mp_world_size": 1,
                "module": {"projector.weight": torch.zeros(1)},
                "param_shapes": [{"projector.weight": torch.Size([1])}],
            },
            global_step / "mp_rank_00_model_states.pt",
        )
    else:
        torch.save(
            {
                "state": {
                    0: {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros(1),
                    }
                },
                "param_groups": [{"lr": 1e-5, "params": [0]}],
            },
            checkpoint / "optimizer.pt",
        )
    for rank in range(world_size):
        rng_name = "rng_state.pth" if world_size == 1 else f"rng_state_{rank}.pth"
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": torch.arange(8, dtype=torch.uint8),
            },
            checkpoint / rng_name,
        )
    _reseal_checkpoint(checkpoint, root_manifest["digest"], step)
    return checkpoint


def _reseal_checkpoint(checkpoint: Path, manifest_digest: str, step: int) -> None:
    sentinel = checkpoint / ".openfly_checkpoint_complete"
    sentinel.unlink(missing_ok=True)
    content = _checkpoint_content_manifest(checkpoint)
    _json(
        sentinel,
        {
            "status": "complete",
            "global_step": step,
            "training_manifest_digest": manifest_digest,
            "content_manifest": {
                "algorithm": content["algorithm"],
                "digest": content["digest"],
                "file_count": content["file_count"],
                "total_bytes": content["total_bytes"],
            },
        },
    )


def test_resumable_checkpoint_requires_complete_exact_state(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = _checkpoint(output)
    report = audit_resumable_checkpoint(checkpoint)
    assert report["global_step"] == 1
    assert resolve_resume_checkpoint(output, "latest") == checkpoint.resolve()
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="content|RNG"):
        audit_resumable_checkpoint(checkpoint)


def test_optimizer_directory_cannot_spoof_state(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = _checkpoint(output)
    (checkpoint / "optimizer.pt").unlink()
    (checkpoint / "global_step1" / "fake_optim_states.pt").mkdir(parents=True)
    with pytest.raises(ValueError, match="optimizer"):
        audit_resumable_checkpoint(checkpoint)


def test_resumable_checkpoint_loads_state_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    (checkpoint / "scheduler.pt").write_bytes(b"not a torch checkpoint")
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="cannot load scheduler"):
        audit_resumable_checkpoint(checkpoint)


@pytest.mark.parametrize("state_name", ["optimizer.pt", "rng_state.pth"])
@pytest.mark.parametrize("replacement", [b"", b"not a torch checkpoint"])
def test_resumable_checkpoint_rejects_empty_or_corrupt_torch_state(
    tmp_path: Path, state_name: str, replacement: bytes
) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    (checkpoint / state_name).write_bytes(replacement)
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="optimizer|RNG"):
        audit_resumable_checkpoint(checkpoint)


def test_resumable_checkpoint_rejects_semantically_empty_optimizer_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    torch.save(
        {"state": {}, "param_groups": [{"lr": 1e-5, "params": [0]}]},
        checkpoint / "optimizer.pt",
    )
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="semantically empty"):
        audit_resumable_checkpoint(checkpoint)


@pytest.mark.parametrize("rng_key", ["python", "numpy"])
def test_resumable_checkpoint_rejects_empty_python_or_numpy_rng(
    tmp_path: Path, rng_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    rng_path = checkpoint / "rng_state.pth"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng[rng_key] = ()
    torch.save(rng, rng_path)
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="Python/NumPy"):
        audit_resumable_checkpoint(checkpoint)


def test_deepspeed_checkpoint_rejects_empty_base_optimizer_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run", world_size=2, deepspeed=True)
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    shard = (
        checkpoint / "global_step1" / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    )
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    payload["optimizer_state_dict"]["base_optimizer_state"]["state"] = {}
    torch.save(payload, shard)
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="empty or invalid"):
        audit_resumable_checkpoint(checkpoint)


def test_deepspeed_checkpoint_requires_exact_rank_and_step_sets(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run", world_size=2, deepspeed=True)
    report = audit_resumable_checkpoint(checkpoint)
    assert report["world_size"] == 2
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    (checkpoint / "rng_state_2.pth").write_bytes(
        (checkpoint / "rng_state_1.pth").read_bytes()
    )
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="RNG rank set mismatch"):
        audit_resumable_checkpoint(checkpoint)


def test_deepspeed_checkpoint_rejects_extra_global_step_directory(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run", world_size=2, deepspeed=True)
    manifest_digest = json.loads(
        (checkpoint / ".openfly_checkpoint_complete").read_text(encoding="utf-8")
    )["training_manifest_digest"]
    (checkpoint / "global_step2").mkdir()
    _reseal_checkpoint(checkpoint, manifest_digest, 1)
    with pytest.raises(ValueError, match="global_step directory set mismatch"):
        audit_resumable_checkpoint(checkpoint)


def test_resumable_checkpoint_rejects_any_symlink(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (checkpoint / "linked.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        audit_resumable_checkpoint(checkpoint)


def test_resume_latest_never_resolves_checkpoint_directory_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    checkpoint = _checkpoint(output)
    link = output / "checkpoint-2"
    link.symlink_to(checkpoint, target_is_directory=True)
    assert resolve_resume_checkpoint(output, "latest") == checkpoint.resolve()
    moved = tmp_path / "outside" / "checkpoint-1"
    moved.parent.mkdir()
    checkpoint.rename(moved)
    link.unlink()
    link.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="resumable"):
        resolve_resume_checkpoint(output, "latest")


def test_fresh_output_rejects_nonempty_without_mutating_bytes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-9"
    checkpoint.mkdir(parents=True)
    old = checkpoint / "optimizer.pt"
    old.write_bytes(b"do-not-touch")
    before = old.read_bytes()
    with pytest.raises(ValueError, match="nonexistent or empty"):
        _validate_output_mode(output, resume_requested=False)
    assert old.read_bytes() == before
    _validate_output_mode(output, resume_requested=True)
    with pytest.raises(ValueError, match="existing"):
        _validate_output_mode(tmp_path / "missing", resume_requested=True)


def test_trainer_main_guards_fresh_output_before_other_validation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-9"
    checkpoint.mkdir(parents=True)
    old = checkpoint / "optimizer.pt"
    old.write_bytes(b"do-not-touch")
    before = old.read_bytes()
    with pytest.raises(ValueError, match="nonexistent or empty"):
        trainer_main(
            [
                "--backend",
                "continue",
                "--model-path",
                str(tmp_path / "missing-model"),
                "--trajectory-root",
                str(tmp_path / "missing-data"),
                "--output-dir",
                str(output),
                "--config",
                str(tmp_path / "missing-config.yaml"),
            ]
        )
    assert old.read_bytes() == before


def test_weighted_original_loss_is_finite_and_backpropagates() -> None:
    trainer_type = _trainer_class()
    trainer = object.__new__(trainer_type)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 5, 5))

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(logits=self.logits)

    model = Model()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[-100, 2, 3]]),
        "loss_weights": torch.tensor([[0.0, 1.0, 2.0]]),
    }
    loss = trainer_type.compute_loss(trainer, model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.logits.grad is not None


def test_parameter_probe_delta_rejects_nonfinite() -> None:
    model = torch.nn.Linear(1, 1)
    name, parameter = next(iter(model.named_parameters()))
    before = parameter.detach().float().cpu().clone()
    with torch.no_grad():
        parameter.add_(1)
    assert _parameter_delta(model, name, before)["changed_elements"] > 0
    with torch.no_grad():
        parameter.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="NaN"):
        _parameter_delta(model, name, before)


def test_safetensors_parameter_delta_uses_content(tmp_path: Path) -> None:
    before_tensors = _contract_tensors()
    after_tensors = {name: value.clone() for name, value in before_tensors.items()}
    changed = {
        "vision_backbone.tensor_0000.weight",
        "language_model.tensor_0000.weight",
        "projector.tensor_0000.weight",
    }
    for name in changed:
        after_tensors[name].add_(1)
    _model_files(tmp_path / "before", before_tensors)
    _model_files(tmp_path / "after", after_tensors)
    _add_lineage(tmp_path / "before", tmp_path / "after")
    report = parameter_delta(tmp_path / "before", tmp_path / "after")
    assert report["tensor_count"] == EXPECTED_TENSOR_COUNT == 982
    assert report["changed_elements"] == 3
    assert report["changed_tensor_count"] == 3
    assert report["structure_audit"]["key_sets"]["missing_keys"] == []
    assert report["structure_audit"]["key_sets"]["unexpected_keys"] == []
    assert report["structure_audit"]["mismatched_tensors"] == []
    for component, expected in EXPECTED_COMPONENT_TENSOR_COUNTS.items():
        component_report = report["components"][component]
        assert component_report["tensor_count"] == expected
        assert component_report["changed_tensor_count"] == 1
        assert component_report["unchanged_tensor_count"] == expected - 1
    assert report["before_digest"] != report["after_digest"]


def test_parameter_delta_uses_comparison_model_lineage(tmp_path: Path) -> None:
    before_tensors = _contract_tensors()
    after_tensors = {name: value.clone() for name, value in before_tensors.items()}
    for name in (
        "vision_backbone.tensor_0000.weight",
        "language_model.tensor_0000.weight",
        "projector.tensor_0000.weight",
    ):
        after_tensors[name].add_(1)
    _model_files(tmp_path / "native-source", {"projector.weight": torch.zeros(1)})
    _model_files(tmp_path / "comparison", before_tensors)
    _model_files(tmp_path / "trained", after_tensors)
    _add_lineage(
        tmp_path / "comparison",
        tmp_path / "trained",
        source=tmp_path / "native-source",
    )
    report = parameter_delta(tmp_path / "comparison", tmp_path / "trained")
    assert report["tensor_count"] == EXPECTED_TENSOR_COUNT
    with pytest.raises(ValueError, match="lineage"):
        parameter_delta(tmp_path / "native-source", tmp_path / "trained")


def test_strict_state_audit_requires_exact_contract_and_no_meta(tmp_path: Path) -> None:
    tensors = _contract_tensors()
    _model_files(tmp_path / "model", tensors)
    report = _strict_state_audit(
        _weight_map(tmp_path / "model"), tensors, loading_info={}
    )
    assert report["key_sets"]["expected_count"] == EXPECTED_TENSOR_COUNT
    assert report["meta_tensor_count"] == 0
    assert report["indexed_components"]["counts"] == dict(
        EXPECTED_COMPONENT_TENSOR_COUNTS
    )

    invalid = {name: value.clone() for name, value in tensors.items()}
    invalid["vision_backbone.tensor_0000.weight"] = torch.empty(1, device="meta")
    with pytest.raises(RuntimeError, match="meta_tensor_count"):
        _strict_state_audit(_weight_map(tmp_path / "model"), invalid, loading_info={})


def test_parameter_delta_rejects_key_shape_and_dtype_mismatch(tmp_path: Path) -> None:
    before_tensors = _contract_tensors()
    after_tensors = {name: value.clone() for name, value in before_tensors.items()}
    del after_tensors["vision_backbone.tensor_0000.weight"]
    after_tensors["vision_backbone.replacement.weight"] = torch.zeros(1)
    after_tensors["language_model.tensor_0000.weight"] = torch.zeros(2)
    after_tensors["projector.tensor_0000.weight"] = torch.zeros(1, dtype=torch.float64)
    _model_files(tmp_path / "before", before_tensors)
    _model_files(tmp_path / "after", after_tensors)
    _add_lineage(tmp_path / "before", tmp_path / "after")
    with pytest.raises(ValueError) as caught:
        parameter_delta(tmp_path / "before", tmp_path / "after")
    message = str(caught.value)
    assert "missing_keys" in message
    assert "unexpected_keys" in message
    assert '"field": "shape"' in message
    assert '"field": "dtype"' in message


def test_parameter_delta_requires_update_in_every_component(tmp_path: Path) -> None:
    before_tensors = _contract_tensors()
    after_tensors = {name: value.clone() for name, value in before_tensors.items()}
    after_tensors["vision_backbone.tensor_0000.weight"].add_(1)
    after_tensors["projector.tensor_0000.weight"].add_(1)
    _model_files(tmp_path / "before", before_tensors)
    _model_files(tmp_path / "after", after_tensors)
    _add_lineage(tmp_path / "before", tmp_path / "after")
    with pytest.raises(RuntimeError, match="language"):
        parameter_delta(tmp_path / "before", tmp_path / "after")


def test_stable_validation_and_output_overlap_gate(tmp_path: Path) -> None:
    left = _stable_validation(
        {"status": "passed", "validation_run_id": "a", "completed_at_ns": 1}
    )
    right = _stable_validation(
        {"status": "passed", "validation_run_id": "b", "completed_at_ns": 2}
    )
    assert left == right
    protected = tmp_path / "model"
    with pytest.raises(ValueError, match="overlaps"):
        _validate_output_paths(
            output=protected / "output",
            validation_report=tmp_path / "report.json",
            model=protected,
            processor=None,
            trajectory=tmp_path / "data",
        )


def test_effective_numeric_values_are_finite() -> None:
    effective = {
        "num_train_epochs": 1.0,
        "learning_rate": float("nan"),
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
        "logging_steps": 1,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "dataloader_num_workers": 0,
        "max_steps": 1,
        "grid_size": 16,
    }
    with pytest.raises(ValueError, match="finite"):
        _validate_effective(effective)

"""Strict OpenFly reload, optimizer-state, and parameter-delta audits."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from baselines.vlm.openfly.artifacts import openfly_model_identity
from baselines.vlm.openfly.actions import validate_tokenizer_model_contract
from baselines.vlm.openfly.bootstrap import register_openfly_classes
from satnav.training.manifest import read_manifest


EXPECTED_COMPONENT_TENSOR_COUNTS = {
    "vision": 685,
    "language": 291,
    "projector": 6,
}
EXPECTED_TENSOR_COUNT = sum(EXPECTED_COMPONENT_TENSOR_COUNTS.values())
_DELTA_CHUNK_ELEMENTS = 4 * 1024 * 1024
_TORCH_TO_SAFETENSORS_DTYPE = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}


def _loading_errors(info: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    )
    return {name: list(info.get(name, ())) for name in fields if info.get(name)}


def strict_reload(
    model_path: Path,
    *,
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> Mapping[str, Any]:
    identity = openfly_model_identity(model_path)
    register_openfly_classes()
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtypes:
        raise ValueError(f"unsupported strict-reload dtype: {dtype}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model, loading_info = AutoModelForVision2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtypes[dtype],
        output_loading_info=True,
    )
    declared = str(identity["facts"]["action_format"])
    if str(model.config.action_format) != declared:
        raise RuntimeError("loaded OpenFly config contradicts checkpoint identity")
    validate_tokenizer_model_contract(model, processor.tokenizer, declared)
    tensor_audit = _strict_state_audit(
        _weight_map(model_path), model.state_dict(), loading_info=loading_info
    )
    model.to(torch.device(device))
    model.eval()
    return {
        "status": "passed",
        "checkpoint_identity": identity,
        "loaded_parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "action_format": declared,
        "device": str(device),
        "dtype": dtype,
        "loading_info": {"errors": {}},
        "tensor_audit": tensor_audit,
    }


def _weight_map(root: Path) -> Mapping[str, Path]:
    root = Path(root).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        result = {}
        for name, filename in weight_map.items():
            path = (root / str(filename)).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("unsafe safetensors shard path") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            result[str(name)] = path
        return result
    single = root / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"no safetensors checkpoint under {root}")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {name: single for name in handle.keys()}


def _checkpoint_tensor_metadata(
    mapping: Mapping[str, Path],
) -> Mapping[str, Mapping[str, Any]]:
    from collections import defaultdict

    from safetensors import safe_open

    grouped = defaultdict(list)
    for name, path in mapping.items():
        grouped[path].append(name)
    metadata: dict[str, Mapping[str, Any]] = {}
    shard_errors = []
    for path, names in grouped.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            expected = set(names)
            actual = set(handle.keys())
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            if missing or unexpected:
                shard_errors.append(
                    {
                        "shard": path.name,
                        "missing_indexed_keys": missing,
                        "unindexed_keys": unexpected,
                    }
                )
            for name in sorted(expected & actual):
                tensor_slice = handle.get_slice(name)
                metadata[name] = {
                    "shape": tuple(tensor_slice.get_shape()),
                    "dtype": str(tensor_slice.get_dtype()),
                }
    if shard_errors or set(metadata) != set(mapping):
        raise ValueError(
            "safetensors index/shard mismatch: "
            + json.dumps(shard_errors, sort_keys=True)
        )
    return metadata


def _component_name(tensor_name: str) -> Optional[str]:
    if tensor_name.startswith("vision_backbone."):
        return "vision"
    if tensor_name.startswith("language_model."):
        return "language"
    if tensor_name.startswith("projector."):
        return "projector"
    return None


def _component_audit(keys: set[str]) -> Mapping[str, Any]:
    counts = {name: 0 for name in EXPECTED_COMPONENT_TENSOR_COUNTS}
    unknown = []
    for key in sorted(keys):
        component = _component_name(key)
        if component is None:
            unknown.append(key)
        else:
            counts[component] += 1
    mismatches = {
        name: {
            "expected": expected,
            "actual": counts[name],
        }
        for name, expected in EXPECTED_COMPONENT_TENSOR_COUNTS.items()
        if counts[name] != expected
    }
    return {
        "counts": counts,
        "unknown_keys": unknown,
        "count_mismatches": mismatches,
    }


def _key_set_audit(expected: set[str], actual: set[str]) -> Mapping[str, Any]:
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_keys": sorted(expected - actual),
        "unexpected_keys": sorted(actual - expected),
    }


def _strict_state_audit(
    indexed: Mapping[str, Path],
    runtime_state: Mapping[str, Any],
    *,
    loading_info: Mapping[str, Any],
) -> Mapping[str, Any]:
    indexed_metadata = _checkpoint_tensor_metadata(indexed)
    indexed_keys = set(indexed_metadata)
    runtime_keys = set(runtime_state)
    key_sets = _key_set_audit(indexed_keys, runtime_keys)
    indexed_components = _component_audit(indexed_keys)
    runtime_components = _component_audit(runtime_keys)
    meta_tensor_keys = sorted(
        name
        for name, tensor in runtime_state.items()
        if getattr(getattr(tensor, "device", None), "type", None) == "meta"
    )
    mismatched_tensors = []
    for name in sorted(indexed_keys & runtime_keys):
        indexed_shape = tuple(indexed_metadata[name]["shape"])
        runtime_shape = tuple(runtime_state[name].shape)
        if indexed_shape != runtime_shape:
            mismatched_tensors.append(
                {
                    "tensor": name,
                    "field": "shape",
                    "indexed": list(indexed_shape),
                    "runtime": list(runtime_shape),
                }
            )
        indexed_dtype = str(indexed_metadata[name]["dtype"])
        runtime_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(
            str(runtime_state[name].dtype), f"unsupported:{runtime_state[name].dtype}"
        )
        if indexed_dtype != runtime_dtype:
            mismatched_tensors.append(
                {
                    "tensor": name,
                    "field": "dtype",
                    "indexed": indexed_dtype,
                    "runtime": runtime_dtype,
                }
            )
    report = {
        "contract": "openfly-982-key-v1",
        "expected_tensor_count": EXPECTED_TENSOR_COUNT,
        "expected_component_counts": dict(EXPECTED_COMPONENT_TENSOR_COUNTS),
        "key_sets": key_sets,
        "indexed_components": indexed_components,
        "runtime_components": runtime_components,
        "meta_tensor_count": len(meta_tensor_keys),
        "meta_tensor_keys": meta_tensor_keys,
        "mismatched_tensors": mismatched_tensors,
        "loading_errors": _loading_errors(loading_info),
    }
    failed = any(
        (
            key_sets["missing_keys"],
            key_sets["unexpected_keys"],
            indexed_components["unknown_keys"],
            indexed_components["count_mismatches"],
            runtime_components["unknown_keys"],
            runtime_components["count_mismatches"],
            meta_tensor_keys,
            mismatched_tensors,
            report["loading_errors"],
        )
    )
    if failed:
        raise RuntimeError(
            "OpenFly checkpoint strict tensor audit failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def _delta_structure_audit(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    key_sets = _key_set_audit(before_keys, after_keys)
    mismatched_tensors = []
    for name in sorted(before_keys & after_keys):
        for field in ("shape", "dtype"):
            if before[name][field] != after[name][field]:
                left = before[name][field]
                right = after[name][field]
                mismatched_tensors.append(
                    {
                        "tensor": name,
                        "field": field,
                        "before": list(left) if field == "shape" else left,
                        "after": list(right) if field == "shape" else right,
                    }
                )
    return {
        "contract": "openfly-full-finetune-delta-v1",
        "expected_tensor_count": EXPECTED_TENSOR_COUNT,
        "expected_component_counts": dict(EXPECTED_COMPONENT_TENSOR_COUNTS),
        "key_sets": key_sets,
        "before_components": _component_audit(before_keys),
        "after_components": _component_audit(after_keys),
        "mismatched_tensors": mismatched_tensors,
    }


def _structure_failed(report: Mapping[str, Any]) -> bool:
    key_sets = report["key_sets"]
    return bool(
        key_sets["missing_keys"]
        or key_sets["unexpected_keys"]
        or report["before_components"]["unknown_keys"]
        or report["before_components"]["count_mismatches"]
        or report["after_components"]["unknown_keys"]
        or report["after_components"]["count_mismatches"]
        or report["mismatched_tensors"]
    )


def _tensor_chunks(left_handle, right_handle, name: str, shape: tuple[int, ...]):
    if not shape or shape[0] == 0:
        yield left_handle.get_tensor(name), right_handle.get_tensor(name)
        return
    row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, _DELTA_CHUNK_ELEMENTS // max(1, row_elements))
    left_slice = left_handle.get_slice(name)
    right_slice = right_handle.get_slice(name)
    for start in range(0, shape[0], rows):
        stop = min(shape[0], start + rows)
        yield left_slice[start:stop], right_slice[start:stop]


def _stream_tensor_delta(
    left_handle, right_handle, name: str, shape: tuple[int, ...]
) -> Mapping[str, Any]:
    import torch

    elements = 0
    changed_elements = 0
    max_abs_delta = 0.0
    squared_l2 = 0.0
    for left, right in _tensor_chunks(left_handle, right_handle, name, shape):
        if torch.is_complex(left) or torch.is_complex(right):
            raise RuntimeError(f"complex checkpoint tensor is unsupported: {name}")
        if torch.is_floating_point(left):
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise RuntimeError(
                    f"checkpoint tensor contains NaN or infinity: {name}"
                )
            work_dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
        else:
            work_dtype = torch.float64
        changed = left.ne(right)
        changed_elements += int(torch.count_nonzero(changed).item())
        elements += int(left.numel())
        delta = (right.to(work_dtype) - left.to(work_dtype)).abs()
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"parameter delta contains NaN or infinity: {name}")
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.max().item()))
            squared_l2 += float(torch.sum(delta * delta, dtype=torch.float64).item())
    return {
        "elements": elements,
        "changed_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
        "squared_l2": squared_l2,
    }


def parameter_delta(
    before: Path,
    after: Path,
    *,
    tensor_name: Optional[str] = None,
) -> Mapping[str, Any]:
    before_identity = openfly_model_identity(before)
    after_identity = openfly_model_identity(after)
    before_map = _weight_map(before)
    after_map = _weight_map(after)
    fact_mismatches = {
        fact: {
            "before": before_identity["facts"].get(fact),
            "after": after_identity["facts"].get(fact),
        }
        for fact in ("model_type", "architectures", "grid_size", "action_format")
        if before_identity["facts"].get(fact) != after_identity["facts"].get(fact)
    }
    if fact_mismatches:
        raise ValueError(
            "OpenFly checkpoint fact mismatch: "
            + json.dumps(fact_mismatches, sort_keys=True)
        )
    manifest_path = Path(after).expanduser().resolve() / "training_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("trained OpenFly checkpoint has no lineage manifest")
    manifest_payload = read_manifest(manifest_path)["payload"]
    lineage = manifest_payload.get(
        "comparison_model", manifest_payload.get("source_model", {})
    )
    if lineage.get("digest") != before_identity["digest"]:
        raise ValueError(
            "trained OpenFly checkpoint lineage does not match --compare-model"
        )
    before_metadata = _checkpoint_tensor_metadata(before_map)
    after_metadata = _checkpoint_tensor_metadata(after_map)
    structure = _delta_structure_audit(before_metadata, after_metadata)
    if tensor_name is not None and (
        tensor_name not in before_metadata or tensor_name not in after_metadata
    ):
        structure = dict(structure)
        structure["requested_tensor_error"] = tensor_name
    if _structure_failed(structure) or structure.get("requested_tensor_error"):
        raise ValueError(
            "OpenFly checkpoint delta structure mismatch: "
            + json.dumps(structure, sort_keys=True)
        )

    components: dict[str, dict[str, Any]] = {
        name: {
            "tensor_count": count,
            "element_count": 0,
            "changed_tensor_count": 0,
            "unchanged_tensor_count": 0,
            "changed_elements": 0,
            "max_abs_delta": 0.0,
            "squared_l2": 0.0,
            "changed_tensor_samples": [],
            "unchanged_tensor_samples": [],
        }
        for name, count in EXPECTED_COMPONENT_TENSOR_COUNTS.items()
    }
    requested = None
    from safetensors import safe_open

    with ExitStack() as stack:
        before_handles = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in set(before_map.values())
        }
        after_handles = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in set(after_map.values())
        }
        for name in sorted(before_metadata):
            component = _component_name(name)
            assert component is not None
            result = dict(
                _stream_tensor_delta(
                    before_handles[before_map[name]],
                    after_handles[after_map[name]],
                    name,
                    tuple(before_metadata[name]["shape"]),
                )
            )
            target = components[component]
            target["element_count"] += result["elements"]
            target["changed_elements"] += result["changed_elements"]
            target["max_abs_delta"] = max(
                target["max_abs_delta"], result["max_abs_delta"]
            )
            target["squared_l2"] += result["squared_l2"]
            changed_key = (
                "changed_tensor_count"
                if result["changed_elements"]
                else "unchanged_tensor_count"
            )
            sample_key = (
                "changed_tensor_samples"
                if result["changed_elements"]
                else "unchanged_tensor_samples"
            )
            target[changed_key] += 1
            if len(target[sample_key]) < 5:
                target[sample_key].append(name)
            if name == tensor_name:
                requested = {
                    "tensor": name,
                    "component": component,
                    "elements": result["elements"],
                    "changed_elements": result["changed_elements"],
                    "max_abs_delta": result["max_abs_delta"],
                    "l2_delta": math.sqrt(result["squared_l2"]),
                }

    unchanged_components = []
    for name, values in components.items():
        values["l2_delta"] = math.sqrt(values.pop("squared_l2"))
        if (
            values["changed_tensor_count"] <= 0
            or values["changed_elements"] <= 0
            or values["max_abs_delta"] <= 0
            or values["l2_delta"] <= 0
        ):
            unchanged_components.append(name)
    if unchanged_components:
        raise RuntimeError(
            "full-finetune produced no finite nonzero update in components: "
            + json.dumps(
                {
                    "unchanged_components": unchanged_components,
                    "components": components,
                },
                sort_keys=True,
            )
        )
    total_squared_l2 = sum(value["l2_delta"] ** 2 for value in components.values())
    return {
        "before_digest": before_identity["digest"],
        "after_digest": after_identity["digest"],
        "contract": "full-finetune",
        "tensor_count": EXPECTED_TENSOR_COUNT,
        "changed_tensor_count": sum(
            value["changed_tensor_count"] for value in components.values()
        ),
        "unchanged_tensor_count": sum(
            value["unchanged_tensor_count"] for value in components.values()
        ),
        "changed_elements": sum(
            value["changed_elements"] for value in components.values()
        ),
        "max_abs_delta": max(value["max_abs_delta"] for value in components.values()),
        "l2_delta": math.sqrt(total_squared_l2),
        "components": components,
        "structure_audit": structure,
        "requested_tensor": requested,
        "status": "passed",
    }


def optimizer_checkpoint_audit(path: Path) -> Mapping[str, Any]:
    from baselines.vlm.openfly.trainer import audit_resumable_checkpoint

    return audit_resumable_checkpoint(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly audit an OpenFly checkpoint")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument("--delta-tensor")
    parser.add_argument("--require-optimizer-state", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report is not None:
        report = Path(args.report).expanduser().resolve()
        protected = [Path(args.model_path).expanduser().resolve()]
        if args.compare_model is not None:
            protected.append(Path(args.compare_model).expanduser().resolve())
        for path in protected:
            if report == path or report in path.parents or path in report.parents:
                raise ValueError("checkpoint report path overlaps a model input")
    report = dict(strict_reload(args.model_path, device=args.device, dtype=args.dtype))
    if args.compare_model is not None:
        report["parameter_delta"] = parameter_delta(
            args.compare_model, args.model_path, tensor_name=args.delta_tensor
        )
    if args.require_optimizer_state:
        report["optimizer_checkpoint"] = optimizer_checkpoint_audit(args.model_path)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

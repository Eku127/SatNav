"""Strict Uni-NaVid checkpoint loading and tensor-consumption audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from unittest import mock

from baselines.vlm.uninavid.artifacts import (
    external_asset_identities,
    model_identity,
)
from baselines.vlm.uninavid.bootstrap import bootstrap_uninavid, source_revision


EXPECTED_ADDED_TOKENS = {
    "<video_special>": 32000,
    "</video_special>": 32001,
    "<image_special>": 32002,
    "</image_special>": 32003,
    "[Navigation]": 32004,
    "<image_sep>": 32005,
}


EXPECTED_COMPONENT_COUNTS = {"language": 291, "projector": 4, "vision": 511}


def _checkpoint_weight_map(model_path: Path) -> Mapping[str, str]:
    index_path = Path(model_path) / "pytorch_model.bin.index.json"
    if not index_path.is_file():
        index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "strict Uni-NaVid reload requires a sharded checkpoint index"
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = payload.get("weight_map")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"invalid checkpoint index: {index_path}")
    missing_shards = sorted(
        name for name in set(mapping.values()) if not (Path(model_path) / name).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"missing checkpoint shards: {missing_shards}")
    return {str(key): str(value) for key, value in mapping.items()}


def _checkpoint_keys(model_path: Path) -> set[str]:
    return set(_checkpoint_weight_map(model_path))


def _loading_entries(values: Any) -> list[Any]:
    result = []
    for value in values or ():
        if isinstance(value, Mapping):
            result.append(dict(value))
        elif isinstance(value, (tuple, list)):
            result.append([str(item) for item in value])
        else:
            result.append(str(value))
    return result


def checkpoint_load_report(
    model: Any,
    model_path: Path,
    loading_info: Mapping[str, Any],
    *,
    flash_attention: bool,
) -> Mapping[str, Any]:
    """Validate the full indexed state registered by a loaded model."""

    indexed = _checkpoint_keys(model_path)
    state = model.state_dict()
    loaded = set(state)
    missing_indexed = sorted(indexed - loaded)
    unindexed_state = sorted(loaded - indexed)
    meta_indexed = sorted(
        name
        for name in indexed & loaded
        if bool(getattr(state[name], "is_meta", False))
    )
    missing = sorted(str(value) for value in loading_info.get("missing_keys", ()))
    unexpected = sorted(
        str(value) for value in loading_info.get("unexpected_keys", ())
    )
    mismatched = _loading_entries(loading_info.get("mismatched_keys", ()))
    component_counts = {
        "language": sum(
            not key.startswith("model.vision_tower.")
            and not key.startswith("model.mm_projector.")
            for key in indexed
        ),
        "projector": sum(key.startswith("model.mm_projector.") for key in indexed),
        "vision": sum(key.startswith("model.vision_tower.") for key in indexed),
    }
    tensor_contract = [
        {
            "name": name,
            "shape": list(state[name].shape),
            "dtype": str(state[name].dtype),
            "is_meta": bool(getattr(state[name], "is_meta", False)),
        }
        for name in sorted(indexed & loaded)
    ]
    contract_digest = hashlib.sha256(
        json.dumps(tensor_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attention_class = type(model.model.layers[0].self_attn).__name__
    errors = []
    for name, values in (
        ("missing_keys", missing),
        ("unexpected_keys", unexpected),
        ("mismatched_keys", mismatched),
        ("missing_indexed_keys", missing_indexed),
        ("unindexed_state_keys", unindexed_state),
        ("meta_indexed_keys", meta_indexed),
    ):
        if values:
            errors.append(name)
    if component_counts != EXPECTED_COMPONENT_COUNTS:
        errors.append("component_tensor_counts")
    if flash_attention and "FlashAttention2" not in attention_class:
        errors.append("attention_class")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "indexed_tensor_count": len(indexed),
        "runtime_tensor_count": len(loaded),
        "component_tensor_counts": component_counts,
        "expected_component_tensor_counts": EXPECTED_COMPONENT_COUNTS,
        "runtime_tensor_contract": {
            "algorithm": "sha256-canonical-name-shape-dtype-is-meta-v2",
            "digest": contract_digest,
            "tensor_count": len(tensor_contract),
            "meta_tensor_count": len(meta_indexed),
        },
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "missing_indexed_keys": missing_indexed,
        "unindexed_state_keys": unindexed_state,
        "meta_indexed_tensor_count": len(meta_indexed),
        "meta_indexed_keys": meta_indexed,
        "attention_class": attention_class,
    }


def _component(name: str) -> str:
    if name.startswith("model.vision_tower."):
        return "vision"
    if name.startswith("model.mm_projector."):
        return "projector"
    return "language"


def _tensor_fingerprints(model_path: Path) -> Mapping[str, Mapping[str, Any]]:
    import torch

    root = Path(model_path).expanduser().resolve()
    weight_map = _checkpoint_weight_map(root)
    fingerprints = {}
    for shard_name in sorted(set(weight_map.values())):
        path = root / shard_name
        try:
            state = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except RuntimeError:
            state = torch.load(path, map_location="cpu", weights_only=True)
        expected = sorted(
            name for name, mapped_shard in weight_map.items() if mapped_shard == shard_name
        )
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"checkpoint shard/index mismatch for {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name in expected:
            tensor = state[name].detach().cpu().contiguous()
            byte_view = tensor.view(torch.uint8).reshape(-1).numpy()
            fingerprints[name] = {
                "component": _component(name),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(byte_view).hexdigest(),
            }
        del state
        gc.collect()
    return fingerprints


def compare_checkpoint_components(
    before: Path, after: Path, *, expected_train_components: str
) -> Mapping[str, Any]:
    before_fingerprints = _tensor_fingerprints(before)
    after_fingerprints = _tensor_fingerprints(after)
    before_keys = set(before_fingerprints)
    after_keys = set(after_fingerprints)
    missing = sorted(before_keys - after_keys)
    unexpected = sorted(after_keys - before_keys)
    shape_dtype_mismatches = []
    components = {}
    for component in ("language", "projector", "vision"):
        names = sorted(
            name
            for name in before_keys & after_keys
            if before_fingerprints[name]["component"] == component
        )
        changed = []
        unchanged = []
        for name in names:
            left = before_fingerprints[name]
            right = after_fingerprints[name]
            if (left["shape"], left["dtype"]) != (right["shape"], right["dtype"]):
                shape_dtype_mismatches.append(name)
            elif left["sha256"] == right["sha256"]:
                unchanged.append(name)
            else:
                changed.append(name)
        components[component] = {
            "tensor_count": len(names),
            "changed_tensor_count": len(changed),
            "unchanged_tensor_count": len(unchanged),
            "first_changed_tensor": changed[0] if changed else None,
        }
    errors = []
    if missing:
        errors.append("missing_keys")
    if unexpected:
        errors.append("unexpected_keys")
    if shape_dtype_mismatches:
        errors.append("shape_dtype_mismatches")
    expected_changes = {
        "all": {"language": True, "projector": True, "vision": False},
        "projector": {"language": False, "projector": True, "vision": False},
    }[expected_train_components]
    for component, should_change in expected_changes.items():
        changed = int(components[component]["changed_tensor_count"])
        if should_change and changed <= 0:
            errors.append(f"{component}_did_not_change")
        if not should_change and changed != 0:
            errors.append(f"{component}_changed")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "expected_train_components": expected_train_components,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_dtype_mismatches": shape_dtype_mismatches,
        "components": components,
        "before_identity": model_identity(before),
        "after_identity": model_identity(after),
    }


def tokenizer_report(tokenizer: Any, model: Any) -> Mapping[str, Any]:
    vocabulary = tokenizer.get_vocab()
    observed = {token: int(vocabulary.get(token, -1)) for token in EXPECTED_ADDED_TOKENS}
    errors = []
    if len(tokenizer) != 32006:
        errors.append(f"tokenizer length is {len(tokenizer)}, expected 32006")
    if observed != EXPECTED_ADDED_TOKENS:
        errors.append("navigation token ids differ from the accepted contract")
    input_shape = list(model.get_input_embeddings().weight.shape)
    output_shape = list(model.get_output_embeddings().weight.shape)
    if input_shape[0] != len(tokenizer) or output_shape[0] != len(tokenizer):
        errors.append("embedding rows do not match tokenizer length")
    return {
        "length": len(tokenizer),
        "added_token_ids": observed,
        "input_embedding_shape": input_shape,
        "output_embedding_shape": output_shape,
        "errors": errors,
    }


def load_strict_model(
    model_path: Path,
    *,
    uninavid_repo: Path,
    eva_path: Path,
    processor_path: Path,
    device: str = "cuda:0",
    dtype: str = "float16",
    flash_attention: bool = True,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Load every indexed tensor by eagerly registering the vision tower.

    Pinned upstream normally constructs the tower with ``delay_load=True``;
    Hugging Face therefore treats all embedded vision weights as unexpected.
    This scoped runtime patch only changes construction timing. It creates the
    same tower before shard loading, allowing the checkpoint to overwrite it.
    """

    repo = bootstrap_uninavid(uninavid_repo)
    model_path = Path(model_path).expanduser().resolve()
    eva_path = Path(eva_path).expanduser().resolve()
    processor_path = Path(processor_path).expanduser().resolve()
    if not eva_path.is_file():
        raise FileNotFoundError(eva_path)
    if not (processor_path / "preprocessor_config.json").is_file():
        raise FileNotFoundError(processor_path)
    if dtype != "float16":
        raise ValueError("the pinned Uni-NaVid evaluation stack requires float16")
    assets = external_asset_identities(eva_path, processor_path)

    import torch
    from transformers import AutoConfig, AutoTokenizer
    import uninavid.model.uninavid_arch as architecture
    from uninavid.model.language_model.llava_llama_vid import (
        LlavaLlamaAttForCausalLM,
    )

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    config.mm_vision_tower = str(eva_path)
    config.image_processor = str(processor_path)
    original_builder = architecture.build_vision_tower

    def eager_builder(value, **kwargs):
        kwargs["delay_load"] = False
        return original_builder(value, **kwargs)

    load_kwargs = {
        "config": config,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
        "device_map": {"": torch.device(device)},
        "output_loading_info": True,
    }
    if flash_attention:
        load_kwargs["use_flash_attention_2"] = True
    with mock.patch.object(
        architecture, "build_vision_tower", new=eager_builder
    ):
        model, loading_info = LlavaLlamaAttForCausalLM.from_pretrained(
            str(model_path), **load_kwargs
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    model.resize_token_embeddings(len(tokenizer))
    model.config.model_path = str(model_path)
    model.config.mm_vision_tower = str(eva_path)
    model.config.image_processor = str(processor_path)

    token_report = tokenizer_report(tokenizer, model)
    report = dict(
        checkpoint_load_report(
            model,
            model_path,
            loading_info,
            flash_attention=flash_attention,
        )
    )
    if token_report["errors"]:
        report["errors"].append("tokenizer")
        report["status"] = "failed"
    report.update({
        "tokenizer": token_report,
        "checkpoint_identity": model_identity(model_path),
        "external_assets": assets,
        "upstream_revision": source_revision(repo) or "unknown",
    })
    if report["errors"]:
        del model, tokenizer
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"strict Uni-NaVid reload failed: {report['errors']}"
        )
    image_processor = model.get_vision_tower().image_processor
    model.eval()
    return tokenizer, model, image_processor, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly reload Uni-NaVid")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--uninavid-repo", type=Path, required=True)
    parser.add_argument("--eva-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument(
        "--expected-train-components", choices=("all", "projector"), default="all"
    )
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    processor = args.processor_path or (
        args.uninavid_repo / "uninavid" / "processor" / "clip-patch14-224"
    )
    tokenizer, model, image_processor, report = load_strict_model(
        args.model_path,
        uninavid_repo=args.uninavid_repo,
        eva_path=args.eva_path,
        processor_path=processor,
        device=args.device,
        flash_attention=not args.no_flash_attention,
    )
    del tokenizer, model, image_processor
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if args.compare_model is not None:
        report = dict(report)
        report["parameter_delta"] = compare_checkpoint_components(
            args.compare_model,
            args.model_path,
            expected_train_components=args.expected_train_components,
        )
        if report["parameter_delta"]["status"] != "passed":
            report["status"] = "failed"
            report["errors"] = [
                *report["errors"],
                *[
                    f"parameter_delta.{value}"
                    for value in report["parameter_delta"]["errors"]
                ],
            ]
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

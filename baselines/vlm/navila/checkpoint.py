"""Strict NaVILA composite-checkpoint reload and parameter-delta audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from baselines.vlm.navila.artifacts import navila_model_identity
from baselines.vlm.navila.bootstrap import bootstrap_navila


def _weight_shapes(root: Path) -> Mapping[str, Tuple[int, ...]]:
    from safetensors import safe_open

    root = Path(root)
    index_path = root / "model.safetensors.index.json"
    files = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted({root / name for name in index["weight_map"].values()})
    else:
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {root}")
    shapes: Dict[str, Tuple[int, ...]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def _compare_component(
    model_state: Mapping[str, Any], expected: Mapping[str, Tuple[int, ...]]
):
    model_shapes = {name: tuple(value.shape) for name, value in model_state.items()}
    return {
        "missing_keys": sorted(set(expected) - set(model_shapes)),
        "unexpected_keys": sorted(set(model_shapes) - set(expected)),
        "mismatched_keys": sorted(
            name
            for name in set(model_shapes) & set(expected)
            if model_shapes[name] != expected[name]
        ),
        "loaded_tensor_count": len(expected),
    }


def strict_reload(
    model_path: Path, navila_repo: Path, device: str
) -> Mapping[str, Any]:
    bootstrap_navila(navila_repo)
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        Path(model_path).name,
        model_base=None,
        device_map={"": device},
        device=device,
    )
    del tokenizer, image_processor
    components = {
        "llm": _compare_component(
            model.get_llm().state_dict(), _weight_shapes(Path(model_path) / "llm")
        ),
        "vision_tower": _compare_component(
            model.get_vision_tower().vision_tower.state_dict(),
            _weight_shapes(Path(model_path) / "vision_tower"),
        ),
        "mm_projector": _compare_component(
            model.get_mm_projector().state_dict(),
            _weight_shapes(Path(model_path) / "mm_projector"),
        ),
    }
    errors = []
    for component, result in components.items():
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            if result[field]:
                errors.append(f"{component}.{field}")
    return {
        "status": "passed" if not errors else "failed",
        "components": components,
        "errors": errors,
        "checkpoint_identity": navila_model_identity(model_path),
    }


def projector_delta(before: Path, after: Path) -> Mapping[str, Any]:
    import torch
    from safetensors.torch import load_file

    before_state = load_file(str(Path(before) / "mm_projector" / "model.safetensors"))
    after_state = load_file(str(Path(after) / "mm_projector" / "model.safetensors"))
    common = sorted(set(before_state) & set(after_state))
    best = None
    for name in common:
        left = before_state[name].float()
        right = after_state[name].float()
        delta = (right - left).abs()
        changed = int(torch.count_nonzero(delta).item())
        candidate = {
            "tensor": name,
            "elements": int(delta.numel()),
            "changed_elements": changed,
            "max_abs_delta": float(delta.max().item()),
        }
        if best is None or candidate["changed_elements"] > best["changed_elements"]:
            best = candidate
    if best is None or best["changed_elements"] <= 0 or best["max_abs_delta"] <= 0:
        raise RuntimeError("no nonzero projector parameter update detected")
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly reload a NaVILA checkpoint")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--navila-repo", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = dict(strict_reload(args.model_path, args.navila_repo, args.device))
    if args.compare_model is not None:
        report["parameter_delta"] = projector_delta(args.compare_model, args.model_path)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

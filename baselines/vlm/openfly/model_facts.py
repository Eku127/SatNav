"""Read the OpenFly inference contract required by evaluation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from baselines.vlm.openfly.actions import SUPPORTED_ACTION_FORMATS


class OpenFlyArtifactError(ValueError):
    """Raised when a checkpoint is incomplete or internally contradictory."""


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenFlyArtifactError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise OpenFlyArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _safe_shard(root: Path, value: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or len(portable.parts) != 1:
        raise OpenFlyArtifactError(f"unsafe checkpoint shard path: {value!r}")
    if not portable.name.endswith(".safetensors"):
        raise OpenFlyArtifactError(f"checkpoint shard is not safetensors: {value!r}")
    path = root / portable.name
    if path.is_symlink():
        raise OpenFlyArtifactError(f"checkpoint shard must not be a symlink: {value!r}")
    return path


def _weight_layout(root: Path) -> tuple[Sequence[Path], int, int]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise OpenFlyArtifactError("safetensors index has no weight_map")
        shards = sorted(
            {_safe_shard(root, str(value)) for value in weight_map.values()}
        )
        missing = [path.name for path in shards if not path.is_file()]
        if missing:
            raise OpenFlyArtifactError(f"checkpoint is missing shards: {missing}")
        resolved_shards = {shard.resolve() for shard in shards}
        unexpected = sorted(
            path.name
            for path in root.glob("*.safetensors")
            if path.resolve() not in resolved_shards
        )
        if unexpected:
            raise OpenFlyArtifactError(
                f"checkpoint has unreferenced safetensors files: {unexpected}"
            )
        metadata = index.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise OpenFlyArtifactError("safetensors index metadata must be an object")
        declared = int(metadata.get("total_size", 0))
        if declared < 0:
            raise OpenFlyArtifactError("safetensors total_size must be non-negative")
        return (index_path, *shards), len(weight_map), declared
    single = root / "model.safetensors"
    if not single.is_file():
        raise OpenFlyArtifactError(
            f"OpenFly checkpoint needs model.safetensors or an index: {root}"
        )
    if single.is_symlink():
        raise OpenFlyArtifactError("model.safetensors must not be a symlink")
    unexpected = sorted(
        path.name for path in root.glob("*.safetensors") if path != single
    )
    if unexpected:
        raise OpenFlyArtifactError(
            f"checkpoint has ambiguous safetensors files: {unexpected}"
        )
    return (single,), -1, int(single.stat().st_size)


def _canonical_backend(value: Any) -> str:
    rendered = str(value or "").strip().lower()
    aliases = {
        "hf": "continue",
        "continue": "continue",
        "native": "scratch",
        "scratch": "scratch",
    }
    if rendered not in aliases:
        raise OpenFlyArtifactError(f"unknown OpenFly backend metadata: {rendered!r}")
    return aliases[rendered]


def _declared_facts(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(config.get("model_type")) != "openvla":
        raise OpenFlyArtifactError("OpenFly config.model_type must be 'openvla'")
    action_format = str(config.get("action_format", "")).strip().lower()
    if action_format not in SUPPORTED_ACTION_FORMATS:
        raise OpenFlyArtifactError(
            "OpenFly config.action_format must be compact or original"
        )
    facts: dict[str, Any] = {
        "action_format": action_format,
        "model_type": "openvla",
        "architectures": list(config.get("architectures", ())),
        "grid_size": int(config.get("grid_size", 16)),
        "torch_dtype": str(config.get("torch_dtype", "unknown")),
        "use_cache": bool(config.get("use_cache", False)),
        "backend": "unknown",
    }
    statistics_path = root / "dataset_statistics.json"
    if statistics_path.is_file():
        statistics = _json_object(statistics_path)
        declared = str(statistics.get("action_format", "")).strip().lower()
        if declared != action_format:
            raise OpenFlyArtifactError(
                "dataset_statistics.action_format contradicts config.action_format"
            )
        facts["training_samples"] = int(statistics.get("num_samples", -1))
    backend_path = root / "backend_meta.json"
    if backend_path.is_file():
        backend = _json_object(backend_path)
        facts["backend"] = _canonical_backend(backend.get("backend"))
        declared = backend.get("action_format")
        if declared is not None and str(declared).strip().lower() != action_format:
            raise OpenFlyArtifactError(
                "backend_meta.action_format contradicts config.action_format"
            )
    return facts


def openfly_model_facts(model_path: Path) -> Mapping[str, Any]:
    """Read the files and metadata required to construct an inference model."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = (
        root / "config.json",
        root / "preprocessor_config.json",
        root / "tokenizer_config.json",
        root / "tokenizer.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise OpenFlyArtifactError(f"OpenFly checkpoint is missing: {missing}")
    _weight_layout(root)
    return _declared_facts(root, _json_object(root / "config.json"))

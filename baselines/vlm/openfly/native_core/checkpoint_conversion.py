from __future__ import annotations

import copy
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_utils import no_init_weights

from baselines.vlm.openfly.actions import (
    ORIGINAL,
    ORIGINAL_UNNORM_KEY,
    get_original_norm_stats,
)
from baselines.vlm.openfly.artifacts import (
    artifact_manifest,
    native_source_identity,
    openfly_model_identity,
    processor_identity,
)
from baselines.vlm.openfly.openfly_core import (
    OpenVLAForActionPrediction,
    register_openfly_auto_classes,
)


_STEP_RE = re.compile(r"step-(\d+)")
_CACHE_COMPLETE_SENTINEL = ".cache_complete"
_DEFAULT_NATIVE_HF_CACHE_ROOT = ""
_CACHE_DISABLE_VALUES = {"", "0", "off", "false", "none", "disable", "disabled", "no"}


def _reject_non_strict_loading(info: Mapping[str, Any], context: str) -> None:
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


def _sort_key(path: Path) -> tuple[int, str]:
    match = _STEP_RE.search(path.name)
    step = int(match.group(1)) if match else -1
    return step, path.name


def _resolve_native_hf_cache_root(
    required_cache_root: str | None = None,
) -> Path | None:
    if required_cache_root is not None:
        configured = str(required_cache_root).strip()
        if not configured:
            raise ValueError("required native HF cache root must not be empty")
        return Path(configured).expanduser().resolve()
    env_value = os.getenv(
        "OPENFLY_NATIVE_HF_CACHE_DIR",
        os.getenv("OPENFLY_NATIVE_HF_CACHE_ROOT", _DEFAULT_NATIVE_HF_CACHE_ROOT),
    ).strip()
    if env_value.lower() in _CACHE_DISABLE_VALUES:
        return None
    return Path(env_value).expanduser().resolve()


def _build_native_hf_cache_key(
    *,
    checkpoint_path: str,
    processor_source: str,
    grid_size: int,
    unnorm_key: str,
    action_format: str,
    use_flash_attention_2: bool,
    torch_dtype: torch.dtype | None,
) -> tuple[str, Mapping[str, Any]]:
    source = native_source_identity(Path(checkpoint_path), Path(processor_source))
    payload = {
        "source_digest": source["digest"],
        "grid_size": int(grid_size),
        "unnorm_key": str(unnorm_key),
        "action_format": str(action_format),
        "use_flash_attention_2": bool(use_flash_attention_2),
        "torch_dtype": str(torch_dtype),
    }
    import hashlib

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, source


def _build_cached_dir_name(checkpoint_path: str, run_dir: str, cache_key: str) -> str:
    del checkpoint_path, run_dir
    return f"openfly-native-{cache_key[:24]}"


def resolve_native_checkpoint_path(model_name_or_path: str) -> tuple[str, str]:
    path = Path(model_name_or_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Native OpenFly checkpoint must be a .pt file: {path}")
        if path.parent.name != "checkpoints":
            raise ValueError(
                f"Expected native checkpoint under a checkpoints/ directory: {path}"
            )
        return str(path), str(path.parents[1])

    if not path.is_dir():
        raise FileNotFoundError(f"Native OpenFly model path not found: {path}")

    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Native OpenFly run dir missing checkpoints/: {path}")

    checkpoints = sorted(checkpoints_dir.glob("*.pt"), key=_sort_key)
    if not checkpoints:
        raise FileNotFoundError(
            f"No native OpenFly checkpoints found under: {checkpoints_dir}"
        )

    return str(checkpoints[-1]), str(path)


def resolve_native_processor_source(
    model_name_or_path: str, explicit: str | None = None
) -> str:
    configured = explicit or os.environ.get("OPENFLY_PROCESSOR_PATH", "").strip()
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"OpenFly processor path not found: {resolved}")
        return str(resolved)

    model_path = Path(model_name_or_path).expanduser().resolve()
    candidates = []
    if model_path.is_dir():
        candidates.extend([model_path, model_path.parent, model_path.parent.parent])
    elif model_path.is_file():
        candidates.extend(
            [
                model_path.parent,
                model_path.parent.parent,
                model_path.parent.parent.parent,
            ]
        )

    baseline_dir = Path(__file__).resolve().parents[1]
    candidates.append(baseline_dir / "model" / "openfly-agent-7b")

    for candidate in candidates:
        if (candidate / "preprocessor_config.json").exists() and (
            candidate / "tokenizer_config.json"
        ).exists():
            return str(candidate)

    raise FileNotFoundError(
        "Unable to resolve a processor/tokenizer source for native OpenFly backend. "
        "Set --processor-name-or-path / OPENFLY_PROCESSOR_PATH or place the model "
        "under baselines/vlm/openfly/model/openfly-agent-7b."
    )


def _rename_layer_scale(key: str) -> str:
    key = key.replace(".ls1.gamma", ".ls1.scale_factor")
    key = key.replace(".ls2.gamma", ".ls2.scale_factor")
    return key


def _map_native_vision_key(key: str) -> str:
    if key.startswith("dino_featurizer."):
        mapped = "vision_backbone.featurizer." + key.removeprefix("dino_featurizer.")
    elif key.startswith("siglip_featurizer."):
        mapped = "vision_backbone.fused_featurizer." + key.removeprefix(
            "siglip_featurizer."
        )
    else:
        raise KeyError(f"Unsupported native vision key: {key}")
    return _rename_layer_scale(mapped)


def _map_native_llm_key(key: str) -> str:
    if not key.startswith("llm."):
        raise KeyError(f"Unsupported native llm key: {key}")
    return "language_model." + key.removeprefix("llm.")


def _map_native_projector_key(key: str) -> str:
    mapping = {
        "projector.0.weight": "projector.fc1.weight",
        "projector.0.bias": "projector.fc1.bias",
        "projector.2.weight": "projector.fc2.weight",
        "projector.2.bias": "projector.fc2.bias",
        "projector.4.weight": "projector.fc3.weight",
        "projector.4.bias": "projector.fc3.bias",
    }
    if key not in mapping:
        raise KeyError(f"Unsupported native projector key: {key}")
    return mapping[key]


def convert_native_state_dict_to_hf(
    native_model_state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}

    def insert(source_key: str, target_key: str, value: Any) -> None:
        if target_key in mapped:
            raise ValueError(
                "native OpenFly keys collide after conversion: "
                f"source={source_key!r}, target={target_key!r}"
            )
        mapped[target_key] = value

    vision_state = native_model_state.get("vision_backbone")
    llm_state = native_model_state.get("llm_backbone")
    projector_state = native_model_state.get("projector")

    if (
        not isinstance(vision_state, dict)
        or not isinstance(llm_state, dict)
        or not isinstance(projector_state, dict)
    ):
        raise ValueError(
            "Native OpenFly checkpoint must contain vision_backbone, llm_backbone, and projector blocks"
        )

    for key, value in vision_state.items():
        insert(key, _map_native_vision_key(key), value)

    for key, value in llm_state.items():
        insert(key, _map_native_llm_key(key), value)

    for key, value in projector_state.items():
        insert(key, _map_native_projector_key(key), value)

    return mapped


def _filter_expected_missing_keys(missing_keys: list[str]) -> list[str]:
    filtered = []
    for key in missing_keys:
        if key.endswith(".rotary_emb.inv_freq"):
            continue
        filtered.append(key)
    return filtered


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype | None):
    if dtype is None:
        yield
        return
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def _load_native_model_into_memory(
    *,
    checkpoint_path: str,
    processor_source: str,
    cache_dir: str | None,
    grid_size: int,
    unnorm_key: str,
    action_format: str,
    use_flash_attention_2: bool,
    torch_dtype: torch.dtype | None,
) -> tuple[OpenVLAForActionPrediction, Any]:
    processor = AutoProcessor.from_pretrained(
        processor_source, cache_dir=cache_dir, local_files_only=True
    )

    config = copy.deepcopy(
        AutoConfig.from_pretrained(
            processor_source, cache_dir=cache_dir, local_files_only=True
        )
    )
    config.grid_size = int(grid_size)
    config.action_format = str(action_format)
    config.satnav_unnorm_key = unnorm_key
    config.norm_stats = get_original_norm_stats(unnorm_key)
    if use_flash_attention_2:
        setattr(config, "_attn_implementation", "flash_attention_2")

    with _temporary_default_dtype(torch_dtype):
        with no_init_weights():
            model = OpenVLAForActionPrediction(config)
    native_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    native_model_state = native_checkpoint.get("model")
    if not isinstance(native_model_state, dict):
        raise ValueError(f"Native checkpoint missing 'model' state: {checkpoint_path}")

    hf_state_dict = convert_native_state_dict_to_hf(native_model_state)
    load_result = model.load_state_dict(hf_state_dict, strict=False)
    missing_keys = _filter_expected_missing_keys(list(load_result.missing_keys))
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to map native OpenFly checkpoint into HF model.\n"
            f"checkpoint={checkpoint_path}\n"
            f"missing_keys={missing_keys}\n"
            f"unexpected_keys={unexpected_keys}"
        )
    return model, processor


def _ensure_cached_native_hf_checkpoint(
    *,
    checkpoint_path: str,
    run_dir: str,
    processor_source: str,
    cache_dir: str | None,
    grid_size: int,
    unnorm_key: str,
    action_format: str,
    use_flash_attention_2: bool,
    torch_dtype: torch.dtype | None,
    required_cache_root: str | None = None,
) -> tuple[str | None, Mapping[str, Any]]:
    cache_key, source_identity = _build_native_hf_cache_key(
        checkpoint_path=checkpoint_path,
        processor_source=processor_source,
        grid_size=grid_size,
        unnorm_key=unnorm_key,
        action_format=action_format,
        use_flash_attention_2=use_flash_attention_2,
        torch_dtype=torch_dtype,
    )
    cache_root = _resolve_native_hf_cache_root(required_cache_root)
    if cache_root is None:
        return None, source_identity

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir_path = cache_root / _build_cached_dir_name(
        checkpoint_path, run_dir, cache_key
    )
    sentinel_path = cache_dir_path / _CACHE_COMPLETE_SENTINEL

    def cache_is_valid() -> bool:
        try:
            payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, Mapping)
                or payload.get("cache_key") != cache_key
            ):
                return False
            expected = payload.get("output_manifest")
            files = tuple(
                path
                for path in cache_dir_path.rglob("*")
                if path.is_file() and path != sentinel_path
            )
            actual = artifact_manifest(cache_dir_path, files)
            if (
                not isinstance(expected, dict)
                or expected.get("digest") != actual["digest"]
            ):
                return False
            openfly_model_identity(cache_dir_path)
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    if sentinel_path.exists() and cache_is_valid():
        print(
            f"[OpenFly scratch] Using verified cached HF checkpoint: {cache_dir_path}",
            flush=True,
        )
        return str(cache_dir_path), source_identity

    lock_path = cache_root / f".{cache_dir_path.name}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        import fcntl

        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if sentinel_path.exists() and cache_is_valid():
                print(
                    f"[OpenFly scratch] Using verified cached HF checkpoint: {cache_dir_path}",
                    flush=True,
                )
                return str(cache_dir_path), source_identity

            if cache_dir_path.exists():
                shutil.rmtree(cache_dir_path)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            print(
                "[OpenFly scratch] Building shared HF cache from native checkpoint "
                f"{checkpoint_path} -> {cache_dir_path}",
                flush=True,
            )
            model, processor = _load_native_model_into_memory(
                checkpoint_path=checkpoint_path,
                processor_source=processor_source,
                cache_dir=cache_dir,
                grid_size=grid_size,
                unnorm_key=unnorm_key,
                action_format=action_format,
                use_flash_attention_2=use_flash_attention_2,
                torch_dtype=torch_dtype,
            )
            model.save_pretrained(
                cache_dir_path, safe_serialization=True, max_shard_size="5GB"
            )
            processor.save_pretrained(cache_dir_path)
            files = tuple(path for path in cache_dir_path.rglob("*") if path.is_file())
            output_manifest = artifact_manifest(cache_dir_path, files)
            with open(sentinel_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cache_key": cache_key,
                        "source_identity": source_identity,
                        "output_manifest": output_manifest,
                        "grid_size": int(grid_size),
                        "unnorm_key": unnorm_key,
                        "action_format": action_format,
                        "use_flash_attention_2": bool(use_flash_attention_2),
                        "torch_dtype": str(torch_dtype),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(
                f"[OpenFly scratch] Shared HF cache ready: {cache_dir_path}", flush=True
            )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return str(cache_dir_path), source_identity


def build_native_hf_model(
    *,
    model_name_or_path: str,
    processor_source: str,
    cache_dir: str | None,
    grid_size: int,
    unnorm_key: str = ORIGINAL_UNNORM_KEY,
    action_format: str = ORIGINAL,
    use_flash_attention_2: bool = False,
    torch_dtype: torch.dtype | None = None,
    required_cache_root: str | None = None,
) -> tuple[OpenVLAForActionPrediction, Any, dict[str, Any], str]:
    register_openfly_auto_classes()
    checkpoint_path, run_dir = resolve_native_checkpoint_path(model_name_or_path)
    cached_hf_dir, source_identity = _ensure_cached_native_hf_checkpoint(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        processor_source=processor_source,
        cache_dir=cache_dir,
        grid_size=grid_size,
        unnorm_key=unnorm_key,
        action_format=action_format,
        use_flash_attention_2=use_flash_attention_2,
        torch_dtype=torch_dtype,
        required_cache_root=required_cache_root,
    )
    processor = AutoProcessor.from_pretrained(
        processor_source, cache_dir=cache_dir, local_files_only=True
    )
    if cached_hf_dir is not None:
        load_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if use_flash_attention_2:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        model, loading_info = OpenVLAForActionPrediction.from_pretrained(
            cached_hf_dir, output_loading_info=True, **load_kwargs
        )
        _reject_non_strict_loading(loading_info, "OpenFly native HF cache")
    else:
        model, processor = _load_native_model_into_memory(
            checkpoint_path=checkpoint_path,
            processor_source=processor_source,
            cache_dir=cache_dir,
            grid_size=grid_size,
            unnorm_key=unnorm_key,
            action_format=action_format,
            use_flash_attention_2=use_flash_attention_2,
            torch_dtype=torch_dtype,
        )

    backend_meta = {
        "backend": "scratch",
        "source_identity": source_identity,
        "processor_identity": processor_identity(Path(processor_source)),
        "hf_cache_id": Path(cached_hf_dir).name if cached_hf_dir is not None else "",
        "action_format": str(action_format),
        "satnav_unnorm_key": unnorm_key,
        "grid_size": int(grid_size),
        "torch_dtype": str(torch_dtype).replace("torch.", "")
        if torch_dtype is not None
        else "default",
    }
    if cached_hf_dir is None:
        raise RuntimeError(
            "scratch training requires a preserved converted HF checkpoint for "
            "full-model parameter-delta validation"
        )
    backend_meta["comparison_model_identity"] = openfly_model_identity(
        Path(cached_hf_dir)
    )
    return model, processor, backend_meta, cached_hf_dir

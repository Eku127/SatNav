from __future__ import annotations

from baselines.vlm.openfly.actions import (
    COMPACT,
    ORIGINAL,
    get_original_action_templates,
    get_original_norm_stats,
    validate_tokenizer_model_contract,
)
from baselines.vlm.openfly.backends.base import (
    SCRATCH_BACKEND,
    TrainBackendArtifacts,
    enforce_full_finetune_contract,
    maybe_enable_gradient_checkpointing,
    resolve_dtype,
)
from baselines.vlm.openfly.dataset import OpenFlyDataCollator
from baselines.vlm.openfly.native_core import (
    build_native_hf_model,
    resolve_native_processor_source,
)


def build_native_train_backend(
    model_args, data_args, training_args
) -> TrainBackendArtifacts:
    # Both compact and original are supported.  The underlying Prismatic model generates
    # tokens via model.generate() regardless of format; compact simply decodes text tokens
    # while original decodes action-dimension tokens and unnormalizes them.
    processor_source = resolve_native_processor_source(
        model_args.model_name_or_path, model_args.processor_name_or_path
    )
    model, processor, backend_meta, comparison_model_path = build_native_hf_model(
        model_name_or_path=model_args.model_name_or_path,
        processor_source=processor_source,
        cache_dir=model_args.cache_dir,
        required_cache_root=model_args.comparison_cache_root,
        grid_size=model_args.grid_size,
        unnorm_key=data_args.unnorm_key,
        action_format=data_args.action_format,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=resolve_dtype(model_args.torch_dtype),
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("OpenFly tokenizer has no usable padding token")

    model.config.use_cache = False
    setattr(model, "grid_size", model_args.grid_size)
    setattr(model.config, "grid_size", model_args.grid_size)
    setattr(model.config, "action_format", data_args.action_format)
    setattr(model.config, "satnav_action_templates", get_original_action_templates())
    validate_tokenizer_model_contract(model, tokenizer, data_args.action_format)

    if data_args.action_format == ORIGINAL:
        setattr(model.config, "satnav_unnorm_key", data_args.unnorm_key)
        norm_stats = get_original_norm_stats(data_args.unnorm_key)
        setattr(model.config, "norm_stats", norm_stats)
        setattr(model, "norm_stats", norm_stats)
        metadata = get_original_norm_stats(data_args.unnorm_key)
    else:
        metadata = {
            "format": "satnav_openfly_compact_actions",
            "action_format": COMPACT,
            "action_names": {"0": "stop", "1": "forward", "2": "left", "3": "right"},
            "grid_size": model_args.grid_size,
        }

    if training_args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)
    backend_meta["trainability"] = enforce_full_finetune_contract(model)

    data_collator = OpenFlyDataCollator(
        processor=processor,
        model_max_length=min(int(tokenizer.model_max_length), 2048),
        pad_token_id=tokenizer.pad_token_id,
        action_format=data_args.action_format,
        original_dim_loss_weights=data_args.original_dim_loss_weights,
    )

    backend_meta.update(
        {
            "backend": SCRATCH_BACKEND,
            "action_format": data_args.action_format,
        }
    )
    return TrainBackendArtifacts(
        backend_name=SCRATCH_BACKEND,
        processor=processor,
        model=model,
        data_collator=data_collator,
        metadata=metadata,
        backend_meta=backend_meta,
        comparison_model_path=comparison_model_path,
    )

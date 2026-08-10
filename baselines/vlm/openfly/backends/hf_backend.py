from __future__ import annotations

from transformers import AutoModelForVision2Seq, AutoProcessor

from baselines.vlm.openfly.actions import (
    COMPACT,
    ORIGINAL,
    get_original_action_templates,
    get_original_norm_stats,
    validate_tokenizer_model_contract,
)
from baselines.vlm.openfly.backends.base import (
    CONTINUE_BACKEND,
    TrainBackendArtifacts,
    build_default_backend_meta,
    enforce_full_finetune_contract,
    maybe_enable_gradient_checkpointing,
    reject_non_strict_loading,
    resolve_dtype,
)
from baselines.vlm.openfly.dataset import OpenFlyDataCollator
from baselines.vlm.openfly.openfly_core import register_openfly_auto_classes


def build_hf_train_backend(
    model_args, data_args, training_args
) -> TrainBackendArtifacts:
    register_openfly_auto_classes()

    processor_source = (
        model_args.processor_name_or_path or model_args.model_name_or_path
    )
    backend_meta = build_default_backend_meta(
        backend_name=CONTINUE_BACKEND,
        model_name_or_path=model_args.model_name_or_path,
        processor_source=processor_source,
        action_format=data_args.action_format,
        grid_size=model_args.grid_size,
    )
    source_facts = backend_meta["source_model_identity"]["facts"]
    if source_facts["action_format"] != data_args.action_format:
        raise ValueError(
            "continue backend action format contradicts source checkpoint metadata"
        )
    if int(source_facts["grid_size"]) != int(model_args.grid_size):
        raise ValueError(
            "continue backend grid size contradicts source checkpoint metadata"
        )
    processor = AutoProcessor.from_pretrained(
        processor_source, cache_dir=model_args.cache_dir, local_files_only=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("OpenFly tokenizer has no usable padding token")

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "low_cpu_mem_usage": True,
        "torch_dtype": resolve_dtype(model_args.torch_dtype),
        "local_files_only": True,
    }
    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model, loading_info = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        output_loading_info=True,
        **model_kwargs,
    )
    reject_non_strict_loading(loading_info, "OpenFly continue checkpoint")
    model.config.use_cache = False
    setattr(model, "grid_size", model_args.grid_size)
    setattr(model.config, "grid_size", model_args.grid_size)
    setattr(model.config, "action_format", data_args.action_format)
    setattr(model.config, "satnav_unnorm_key", data_args.unnorm_key)
    setattr(model.config, "satnav_action_templates", get_original_action_templates())
    validate_tokenizer_model_contract(model, tokenizer, data_args.action_format)

    if data_args.action_format == ORIGINAL:
        norm_stats = get_original_norm_stats(data_args.unnorm_key)
        setattr(model.config, "norm_stats", norm_stats)
        setattr(model, "norm_stats", norm_stats)

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

    if data_args.action_format == ORIGINAL:
        metadata = get_original_norm_stats(data_args.unnorm_key)
    else:
        metadata = {
            "format": "satnav_openfly_compact_actions",
            "action_format": COMPACT,
            "action_names": {
                "0": "stop",
                "1": "forward",
                "2": "left",
                "3": "right",
            },
            "grid_size": model_args.grid_size,
        }

    return TrainBackendArtifacts(
        backend_name=CONTINUE_BACKEND,
        processor=processor,
        model=model,
        data_collator=data_collator,
        metadata=metadata,
        backend_meta=backend_meta,
        comparison_model_path=model_args.model_name_or_path,
    )

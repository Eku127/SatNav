from __future__ import annotations

import pytest
import torch
from PIL import Image

from baselines.vlm.openfly.backends.base import enforce_full_finetune_contract
from baselines.vlm.openfly.openfly_core.configuration_prismatic import OpenFlyConfig
from baselines.vlm.openfly.openfly_core.modeling_prismatic import (
    PrismaticForConditionalGeneration,
    PrismaticVisionBackbone,
)
from baselines.vlm.openfly.openfly_core.processing_prismatic import (
    letterbox_pad_transform,
)
from baselines.vlm.openfly.native_core.checkpoint_conversion import (
    _build_native_hf_cache_key,
    _resolve_native_hf_cache_root,
)


def test_openfly_config_is_narrow_and_default_is_supported() -> None:
    config = OpenFlyConfig()
    assert config.vision_backbone_id == "dinosiglip-vit-so-224px"
    assert config.use_fused_vision_backbone is True
    with pytest.raises(ValueError, match="supports only"):
        OpenFlyConfig(vision_backbone_id="siglip-vit-so400m")
    with pytest.raises(ValueError, match="grid_size=16"):
        OpenFlyConfig(grid_size=8)


def test_generation_input_key_and_fail_closed_embeddings() -> None:
    model = object.__new__(PrismaticForConditionalGeneration)
    result = PrismaticForConditionalGeneration.prepare_inputs_for_generation(
        model,
        input_ids=torch.tensor([[1, 2]]),
        pixel_values=torch.zeros(3, 6, 224, 224),
    )
    assert "input_ids" in result and "input_embeds" not in result
    with pytest.raises(ValueError, match="inputs_embeds"):
        PrismaticForConditionalGeneration.prepare_inputs_for_generation(
            model,
            input_ids=torch.tensor([[1]]),
            inputs_embeds=torch.zeros(1, 1, 2),
        )


def test_vision_shape_and_odd_letterbox_are_checked() -> None:
    vision = object.__new__(PrismaticVisionBackbone)
    vision.use_fused_vision_backbone = True
    with pytest.raises(ValueError, match="Expected OpenFly"):
        PrismaticVisionBackbone.forward(vision, torch.zeros(1, 3, 3, 224, 224))
    assert letterbox_pad_transform(Image.new("RGB", (3, 2)), (0, 0, 0)).size == (3, 3)


def test_full_finetune_contract_rejects_frozen_component() -> None:
    class TinyOpenFly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_backbone = torch.nn.Linear(1, 1)
            self.language_model = torch.nn.Linear(1, 1)
            self.projector = torch.nn.Linear(1, 1)

    model = TinyOpenFly()
    report = enforce_full_finetune_contract(model)
    assert set(report["components"]) == {"vision", "language", "projector"}
    model.vision_backbone.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="vision_backbone.weight"):
        enforce_full_finetune_contract(model)


def test_scratch_comparison_cache_is_required_and_action_specific(
    tmp_path, monkeypatch
) -> None:
    required = tmp_path / "preserved-source"
    assert _resolve_native_hf_cache_root(str(required)) == required.resolve()
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_native_hf_cache_root("")
    monkeypatch.setattr(
        "baselines.vlm.openfly.native_core.checkpoint_conversion.native_source_identity",
        lambda checkpoint, processor: {"digest": f"{checkpoint.name}:{processor.name}"},
    )
    common = {
        "checkpoint_path": str(tmp_path / "source.pt"),
        "processor_source": str(tmp_path / "processor"),
        "grid_size": 16,
        "unnorm_key": "satnav_original",
        "use_flash_attention_2": True,
        "torch_dtype": torch.bfloat16,
    }
    compact, _ = _build_native_hf_cache_key(**common, action_format="compact")
    original, _ = _build_native_hf_cache_key(**common, action_format="original")
    assert compact != original

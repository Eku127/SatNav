from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from baselines.vlm.openfly.dataset import IGNORE_INDEX, OpenFlyDataCollator
from baselines.vlm.openfly.openfly_core.processing_prismatic import (
    PrismaticImageProcessor,
)
from baselines.vlm.openfly.openfly_core.registration import (
    register_openfly_auto_classes,
)
from baselines.vlm.openfly.prompting import build_openfly_prompt_with_answer


def _processor_kwargs(strategy: str = "resize-naive") -> dict:
    return {
        "use_fused_vision_backbone": True,
        "image_resize_strategy": strategy,
        "input_sizes": [(3, 8, 8), (3, 8, 8)],
        "interpolations": ["bicubic", "bicubic"],
        "means": [(0.485, 0.456, 0.406), (0.5, 0.5, 0.5)],
        "stds": [(0.229, 0.224, 0.225), (0.5, 0.5, 0.5)],
    }


def test_legacy_non_letterbox_state_is_migrated_without_recomputing_cache() -> None:
    serialized_resize = [
        {
            "size": [8, 8],
            "interpolation": 3,
            "max_size": None,
            "antialias": False,
        }
        for _ in range(2)
    ]
    serialized_crop = [{"output_size": [8, 8]} for _ in range(2)]
    serialized_normalize = [
        {
            "inplace": False,
            "mean": [0.484375, 0.455078125, 0.40625],
            "std": [0.228515625, 0.2236328125, 0.224609375],
        },
        {"inplace": False, "mean": [0.5] * 3, "std": [0.5] * 3},
    ]
    processor = PrismaticImageProcessor(
        **_processor_kwargs(),
        tvf_do_letterbox=False,
        tvf_letterbox_fill=None,
        tvf_resize_params=serialized_resize,
        tvf_crop_params=serialized_crop,
        tvf_normalize_params=serialized_normalize,
    )

    assert processor.tvf_do_letterbox == [False, False]
    assert processor.tvf_letterbox_fill == [None, None]
    assert processor.tvf_resize_params == serialized_resize
    assert processor.tvf_crop_params == serialized_crop
    assert processor.tvf_normalize_params == serialized_normalize


def test_current_state_round_trips_and_handles_odd_image_dimensions(
    tmp_path: Path,
) -> None:
    processor = PrismaticImageProcessor(**_processor_kwargs("letterbox"))
    transformed = processor(images=[Image.new("RGB", (5, 3))], return_tensors="pt")
    assert transformed["pixel_values"].shape == (1, 6, 8, 8)
    assert torch.isfinite(transformed["pixel_values"]).all()

    processor.save_pretrained(tmp_path)
    reloaded = PrismaticImageProcessor.from_pretrained(
        tmp_path, local_files_only=True
    )
    assert reloaded.tvf_do_letterbox == [True, True]
    assert reloaded.tvf_letterbox_fill == [
        list(fill) for fill in processor.tvf_letterbox_fill
    ]
    assert reloaded.tvf_normalize_params == processor.tvf_normalize_params
    reloaded_values = reloaded(
        images=[Image.new("RGB", (5, 3))], return_tensors="pt"
    )["pixel_values"]
    assert reloaded_values.shape == (1, 6, 8, 8)
    torch.testing.assert_close(reloaded_values, transformed["pixel_values"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"tvf_do_letterbox": "false", "tvf_letterbox_fill": [None, None]},
            "tvf_do_letterbox",
        ),
        (
            {"tvf_do_letterbox": True, "tvf_letterbox_fill": None},
            "Unsupported legacy",
        ),
        (
            {"tvf_do_letterbox": False, "tvf_letterbox_fill": [None, None]},
            "Unsupported legacy",
        ),
        (
            {"tvf_do_letterbox": [False], "tvf_letterbox_fill": [None]},
            "tvf_do_letterbox",
        ),
        (
            {"tvf_do_letterbox": [False, 0], "tvf_letterbox_fill": [None, None]},
            "tvf_do_letterbox",
        ),
        (
            {"tvf_do_letterbox": [True, True], "tvf_letterbox_fill": [[0] * 3] * 2},
            "contradicts",
        ),
        (
            {
                "tvf_do_letterbox": [False, False],
                "tvf_letterbox_fill": [None, [0, 0, 0]],
            },
            "null fill",
        ),
        ({"tvf_resize_params": [{}, {}]}, "tvf_resize_params"),
        ({"tvf_crop_params": [{}, {}]}, "tvf_crop_params"),
        ({"tvf_normalize_params": [{}, {}]}, "tvf_normalize_params"),
        (
            {
                "tvf_normalize_params": [
                    {"mean": [0.5] * 3, "std": [0.0] * 3, "inplace": False},
                    {"mean": [0.5] * 3, "std": [0.5] * 3, "inplace": False},
                ]
            },
            "tvf_normalize_params",
        ),
        (
            {
                "tvf_normalize_params": [
                    {"mean": [1e308] * 3, "std": [0.5] * 3, "inplace": False},
                    {"mean": [0.5] * 3, "std": [0.5] * 3, "inplace": False},
                ]
            },
            "tvf_normalize_params",
        ),
        (
            {
                "tvf_normalize_params": [
                    {"mean": [0.5] * 3, "std": [1e-46] * 3, "inplace": False},
                    {"mean": [0.5] * 3, "std": [0.5] * 3, "inplace": False},
                ]
            },
            "tvf_normalize_params",
        ),
    ],
)
def test_invalid_serialized_transform_state_fails_during_load(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PrismaticImageProcessor(**_processor_kwargs(), **overrides)


def test_letterbox_state_validates_rgb_fill() -> None:
    with pytest.raises(ValueError, match="integer RGB"):
        PrismaticImageProcessor(
            **_processor_kwargs("letterbox"),
            tvf_do_letterbox=[True, True],
            tvf_letterbox_fill=[[0, 0, 256], [0, 0, 0]],
        )


_REAL_PROCESSOR = os.environ.get("OPENFLY_REAL_PROCESSOR_PATH", "").strip()


@pytest.mark.skipif(not _REAL_PROCESSOR, reason="real processor path not configured")
def test_real_processor_runs_through_the_training_collator() -> None:
    from transformers import AutoProcessor

    register_openfly_auto_classes()
    processor = AutoProcessor.from_pretrained(
        _REAL_PROCESSOR,
        local_files_only=True,
    )
    assert processor.image_processor.tvf_do_letterbox == [False, False]
    assert processor.image_processor.tvf_letterbox_fill == [None, None]
    assert processor.image_processor.tvf_normalize_params[0]["mean"] == [
        0.484375,
        0.455078125,
        0.40625,
    ]

    collator = OpenFlyDataCollator(
        processor=processor,
        model_max_length=2048,
        pad_token_id=int(processor.tokenizer.pad_token_id),
        action_format="compact",
    )
    batch = collator(
        [
            {
                "full_text": build_openfly_prompt_with_answer(
                    "go to the marked destination", "forward"
                ),
                "images": [
                    Image.new("RGB", (31, 23), (index * 40, 20, 200))
                    for index in range(3)
                ],
            }
        ]
    )
    assert batch["pixel_values"].shape == (1, 3, 6, 224, 224)
    assert torch.isfinite(batch["pixel_values"]).all()
    assert torch.count_nonzero(batch["labels"] != IGNORE_INDEX) > 0

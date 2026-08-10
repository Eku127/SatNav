from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from baselines.vlm.openfly.actions import (
    convert_original_action_vector_to_action,
    get_normalized_original_action_vector,
    parse_action_text,
    resolve_action_format,
    validate_tokenizer_model_contract,
)
from baselines.vlm.openfly.prompting import (
    ASSISTANT_BOUNDARY,
    build_openfly_prompt,
    build_openfly_prompt_with_answer,
)


def test_action_format_auto_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires checkpoint metadata"):
        resolve_action_format("auto")
    assert resolve_action_format("auto", "compact") == "compact"
    with pytest.raises(ValueError):
        resolve_action_format("auto", "sentence")


def test_action_parse_and_original_round_trip() -> None:
    assert parse_action_text("Take FORWARD now") == 1
    assert parse_action_text("unparseable") is None
    normalized = get_normalized_original_action_vector(2)
    assert normalized.shape == (8,)
    action, rounded, distance = convert_original_action_vector_to_action(
        np.array([0, 0, 14.4, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    assert action == 2
    assert rounded.tolist()[:4] == [0, 0, 14, 0]
    assert distance > 0
    with pytest.raises(ValueError):
        convert_original_action_vector_to_action(np.array([np.nan] * 8))


def test_prompt_history_and_last_assistant_boundary() -> None:
    prompt = build_openfly_prompt(" Reach\nTHE bridge ", [1, 2, 3], 2)
    assert "reach the bridge" in prompt
    assert "last 2 of 3" in prompt
    assert prompt.endswith(f"{ASSISTANT_BOUNDARY}")
    full = build_openfly_prompt_with_answer("Reach bridge", "forward", [1], 16)
    assert full.rfind(ASSISTANT_BOUNDARY) > 0
    assert full.endswith("forward</s>")


def test_original_vocab_contract_rejects_unpadded_embeddings() -> None:
    tokenizer = SimpleNamespace(vocab_size=32000)
    config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=32000),
        pad_to_multiple_of=64,
        n_action_bins=256,
    )
    model = SimpleNamespace(config=config)
    with pytest.raises(ValueError, match="model_vocab"):
        validate_tokenizer_model_contract(model, tokenizer, "original")
    config.text_config.vocab_size = 32064
    validate_tokenizer_model_contract(model, tokenizer, "original")

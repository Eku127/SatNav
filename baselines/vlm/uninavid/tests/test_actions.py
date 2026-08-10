import pytest

from baselines.vlm.uninavid.actions import (
    action_chunk_or_stop,
    build_prompt,
    format_action_target,
    parse_action_chunk,
)


def test_exact_prompt_target_and_whole_word_action_parser():
    prompt = build_prompt("go north", "[Navigation]", "<image>")
    assert "given [Navigation] <image>" in prompt
    assert "task is: 'go north'" in prompt
    assert format_action_target((1, 2, 3, 0)) == (
        "1. forward 2. left 3. right 4. stop"
    )
    assert parse_action_chunk("forward leftward LEFT, right; stop forward") == [
        1,
        2,
        3,
        0,
    ]
    assert action_chunk_or_stop("unknown") == ([0], True)


def test_target_rejects_wrong_width_or_action():
    with pytest.raises(ValueError, match="four"):
        format_action_target((1, 2, 3))
    with pytest.raises(ValueError, match="unsupported"):
        format_action_target((1, 2, 3, 9))

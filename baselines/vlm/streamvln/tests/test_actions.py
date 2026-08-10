import pytest

from baselines.vlm.streamvln.actions import (
    format_action_symbols,
    parse_action_symbols,
)


def test_symbol_actions_round_trip_and_ignore_non_actions():
    assert format_action_symbols([1, 2, 3, 0]) == "↑←→STOP"
    assert parse_action_symbols("Next: ↑ then noise ←→, finally STOP.") == [
        1,
        2,
        3,
        0,
    ]


def test_empty_generation_has_no_action_for_adapter_fallback():
    assert parse_action_symbols("I cannot decide") == []


def test_unsupported_action_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        format_action_symbols([4])

"""StreamVLN's symbolic SatNav action vocabulary."""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping


ACTION_SYMBOLS: Mapping[int, str] = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
}
SYMBOL_ACTIONS = {symbol: action for action, symbol in ACTION_SYMBOLS.items()}
_ACTION_PATTERN = re.compile(
    "|".join(re.escape(symbol) for symbol in SYMBOL_ACTIONS)
)


def parse_action_symbols(output: str) -> List[int]:
    """Extract primitive SatNav actions from a StreamVLN text generation.

    Non-action text is ignored and matches retain generation order.  ``STOP``
    is a word token while movement/turn actions use the exact Unicode arrows
    used by the pinned upstream model.
    """

    if not isinstance(output, str):
        raise TypeError(f"model output must be str, got {type(output).__name__}")
    return [SYMBOL_ACTIONS[match] for match in _ACTION_PATTERN.findall(output)]


def format_action_symbols(actions: Iterable[int]) -> str:
    """Render primitive SatNav action indices using StreamVLN symbols."""

    symbols = []
    for action in actions:
        try:
            symbols.append(ACTION_SYMBOLS[int(action)])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"unsupported SatNav action: {action!r}") from error
    return "".join(symbols)

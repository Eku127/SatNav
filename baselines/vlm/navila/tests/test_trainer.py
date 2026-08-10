from types import SimpleNamespace

import pytest

from baselines.vlm.navila.dataset import TRAINING_SENTINEL_TOKEN
from baselines.vlm.navila.trainer import _bind_existing_training_sentinel


class _Tokenizer:
    def __init__(self, *, contains_sentinel=True, skipped=True):
        self._tokens = {"ordinary": 0}
        if contains_sentinel:
            self._tokens[TRAINING_SENTINEL_TOKEN] = 1
        self._skipped = skipped

    def __len__(self):
        return len(self._tokens)

    def get_vocab(self):
        return dict(self._tokens)

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        token_id = self._tokens.get(text, 0)
        return SimpleNamespace(input_ids=[token_id])

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        assert token_ids == [1]
        return "" if self._skipped else TRAINING_SENTINEL_TOKEN

    def add_tokens(self, tokens, *, special_tokens):
        assert tokens == [TRAINING_SENTINEL_TOKEN]
        assert special_tokens is True
        if TRAINING_SENTINEL_TOKEN in self._tokens:
            return 0
        self._tokens[TRAINING_SENTINEL_TOKEN] = len(self._tokens)
        return 1


def test_training_sentinel_reuses_existing_special_token_without_vocab_growth():
    tokenizer = _Tokenizer()
    original_size = len(tokenizer)
    _bind_existing_training_sentinel(tokenizer)
    assert len(tokenizer) == original_size
    assert tokenizer.sentinel_token == TRAINING_SENTINEL_TOKEN
    assert tokenizer.sentinel_token_id == 1


@pytest.mark.parametrize(
    "tokenizer,match",
    (
        (_Tokenizer(contains_sentinel=False), "missing"),
        (_Tokenizer(skipped=False), "skipped special token"),
    ),
)
def test_training_sentinel_fails_closed_for_incompatible_tokenizer(tokenizer, match):
    with pytest.raises(ValueError, match=match):
        _bind_existing_training_sentinel(tokenizer)

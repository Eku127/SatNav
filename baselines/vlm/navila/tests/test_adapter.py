from types import SimpleNamespace

import numpy as np
import pytest

from baselines.vlm.navila.adapter import NaVILAPolicyAdapter


class _Cuda:
    available = True
    seeds = []

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def manual_seed_all(cls, seed):
        cls.seeds.append(seed)


class _Torch:
    cuda = _Cuda
    seeds = []

    @classmethod
    def manual_seed(cls, seed):
        cls.seeds.append(seed)


class _Model:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


def _adapter():
    return NaVILAPolicyAdapter(
        model=_Model(),
        tokenizer=object(),
        image_processor=object(),
        torch_module=_Torch,
        process_images=object(),
        tokenizer_image_token=object(),
        keywords_stopping_criteria=object(),
        image_token_index=-200,
        conversation_templates={},
        separator_style=object(),
        device="cpu",
        dtype="float16",
    )


def _context(seed):
    return SimpleNamespace(
        seed=seed,
        episode=SimpleNamespace(instruction=SimpleNamespace(text="go north")),
    )


def _observation(value):
    return {"rgb": np.full((2, 2, 3), value, dtype=np.uint8)}


def test_lifecycle_queue_history_seed_and_parse_fallback(monkeypatch):
    _Torch.seeds.clear()
    _Cuda.seeds.clear()
    policy = _adapter()
    generated = iter(("turn left 30 degrees", "unparseable response"))
    monkeypatch.setattr(policy, "_predict", lambda frames: next(generated))

    policy.reset(_context(41))
    first = policy.act(_observation(1))
    second = policy.act(_observation(2))
    assert (first.action, second.action) == (2, 2)
    assert first.info["source"] == "generated"
    assert second.info["source"] == "queued"
    assert len(policy._history) == 2
    assert policy._queue == []
    assert _Torch.seeds == [41]
    assert _Cuda.seeds == [41]

    policy.reset(_context(42))
    assert policy._history == []
    assert policy._queue == []
    fallback = policy.act(_observation(3))
    assert fallback.action == 1
    assert fallback.info["parse_fallback"] is True
    assert len(policy._history) == 1

    policy.close()
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset(_context(43))
    with pytest.raises(RuntimeError, match="reset"):
        policy.act(_observation(4))

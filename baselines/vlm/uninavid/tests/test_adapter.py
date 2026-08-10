from types import SimpleNamespace

import numpy as np
import pytest

from baselines.vlm.uninavid.adapter import UniNaVidPolicyAdapter


class _Cuda:
    seeds = []

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def manual_seed_all(cls, seed):
        cls.seeds.append(seed)


class _Torch:
    cuda = _Cuda
    seeds = []

    @classmethod
    def manual_seed(cls, seed):
        cls.seeds.append(seed)


class _TokenResult:
    input_ids = np.asarray([[0, 1]])


class _TokenIds:
    def __getitem__(self, value):
        return self

    def to(self, device):
        del device
        return self


class _Tokenizer:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(input_ids=_TokenIds())


class _Core:
    def __init__(self):
        self.new_frames = -1
        self.feat_cache = None
        self.long_feat_cache = None
        self.reset_calls = 0

    def initialize_online_inference_nav_feat_cache(self):
        self.reset_calls += 1
        self.feat_cache = None
        self.long_feat_cache = None


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(run_type=None)
        self.core = _Core()
        self.eval_calls = 0

    def get_model(self):
        return self.core

    def eval(self):
        self.eval_calls += 1


def _adapter():
    return UniNaVidPolicyAdapter(
        model=_Model(),
        tokenizer=_Tokenizer(),
        image_processor=object(),
        torch_module=_Torch,
        constants=SimpleNamespace(
            VIDEO_START_SPECIAL_TOKEN="a",
            VIDEO_END_SPECIAL_TOKEN="b",
            IMAGE_START_TOKEN="c",
            IMAGE_END_TOKEN="d",
            NAVIGATION_SPECIAL_TOKEN="e",
            IAMGE_SEPARATOR="f",
        ),
        tokenizer_image_token=object(),
        keywords_stopping_criteria=object(),
        conversation_templates={},
        separator_style=object(),
        device="cpu",
    )


def _context(seed):
    return SimpleNamespace(
        seed=seed,
        episode=SimpleNamespace(instruction=SimpleNamespace(text="go north")),
        agent_state=SimpleNamespace(position=(1, 2, 3), rotation=4),
    )


def test_episode_reset_action_queue_and_incremental_frames(monkeypatch):
    _Torch.seeds.clear()
    _Cuda.seeds.clear()
    policy = _adapter()
    calls = []

    def predict(frames):
        calls.append(len(frames))
        return "forward forward forward forward", {"frames": len(frames)}

    monkeypatch.setattr(policy, "_predict", predict)
    policy.reset(_context(41))
    actions = []
    for value in range(5):
        actions.append(policy.act({"rgb": np.full((2, 2, 3), value)}).action)
    assert actions == [1, 1, 1, 1, 1]
    assert calls == [1, 4]
    assert policy.model.core.reset_calls == 1
    assert _Torch.seeds == [41]
    assert _Cuda.seeds == [41]

    policy.close()
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset(_context(42))
    with pytest.raises(RuntimeError, match="reset"):
        policy.act({"rgb": np.zeros((2, 2, 3))})

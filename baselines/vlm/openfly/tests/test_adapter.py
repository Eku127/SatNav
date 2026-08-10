from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.vlm.openfly.adapter import OpenFlyPolicyAdapter
from satnav.evaluation import EpisodeContext


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def batch_decode(self, values, skip_special_tokens=True):
        del values, skip_special_tokens
        return ["forward"]


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.frame_batches = []

    def __call__(self, *, text, images, return_tensors):
        assert "Past actions" in text and return_tensors == "pt"
        self.frame_batches.append(images)
        return {
            "input_ids": torch.tensor([[1, 3]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.zeros(3, 6, 2, 2),
        }


class _Model:
    def eval(self):
        return self

    def generate(self, **inputs):
        return torch.cat([inputs["input_ids"], torch.tensor([[7]])], dim=1)


def _context() -> EpisodeContext:
    return EpisodeContext(
        episode=SimpleNamespace(instruction=SimpleNamespace(instruction_text="Go home")),
        episode_key="key",
        split="val_seen",
        episode_index=0,
        rank=0,
        world_size=1,
        max_steps=5,
        seed=4,
        environment=SimpleNamespace(),
    )


def test_compact_adapter_tracks_exact_three_frame_history() -> None:
    processor = _Processor()
    adapter = OpenFlyPolicyAdapter(
        model=_Model(),
        processor=processor,
        torch_module=torch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        action_format="compact",
    )
    adapter.reset(_context())
    observation = {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
    assert adapter.act(observation).action == 1
    assert adapter.act(observation).action == 1
    assert len(processor.frame_batches[0]) == 3
    assert adapter._actions == [1, 1]
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.reset(_context())


def test_unparseable_output_is_error_not_stop() -> None:
    processor = _Processor()
    processor.tokenizer.batch_decode = lambda *args, **kwargs: ["unknown"]
    adapter = OpenFlyPolicyAdapter(
        model=_Model(), processor=processor, torch_module=torch,
        device=torch.device("cpu"), dtype=torch.float32, action_format="compact"
    )
    adapter.reset(_context())
    with pytest.raises(ValueError, match="no canonical action"):
        adapter.act({"rgb": np.zeros((2, 2, 3), dtype=np.uint8)})

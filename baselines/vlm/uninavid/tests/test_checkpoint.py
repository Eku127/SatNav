import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.vlm.uninavid.artifacts import (
    external_asset_identities,
    model_identity,
)
from baselines.vlm.uninavid.checkpoint import checkpoint_load_report


def _fake_checkpoint(root: Path) -> Path:
    root.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "tokenizer.model").write_bytes(b"tokenizer")
    (root / "pytorch_model-00001-of-00001.bin").write_bytes(b"weights")
    (root / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "pytorch_model-00001-of-00001.bin"
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_model_identity_covers_inference_files_but_not_runtime_logs(tmp_path):
    model = _fake_checkpoint(tmp_path / "model")
    before = model_identity(model)
    (model / "train.log").write_text("runtime\n", encoding="utf-8")
    (model / "checkpoint-1").mkdir()
    (model / "checkpoint-1/state").write_bytes(b"runtime")
    assert model_identity(model)["digest"] == before["digest"]
    (model / "tokenizer.model").write_bytes(b"changed")
    assert model_identity(model)["digest"] != before["digest"]


def test_external_assets_reject_unverified_files(tmp_path):
    eva = tmp_path / "eva_vit_g.pth"
    eva.write_bytes(b"not official")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="EVA checkpoint digest mismatch"):
        external_asset_identities(eva, processor)


def test_checkpoint_load_report_requires_806_registered_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    shard = checkpoint / "pytorch_model-00001-of-00001.bin"
    shard.write_bytes(b"placeholder")
    names = [f"model.layers.{index}.weight" for index in range(291)]
    names.extend(f"model.mm_projector.{index}" for index in range(4))
    names.extend(f"model.vision_tower.{index}" for index in range(511))
    (checkpoint / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in names}}),
        encoding="utf-8",
    )

    class LlamaFlashAttention2:
        pass

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=LlamaFlashAttention2())]
        ),
        state_dict=lambda: {name: torch.zeros(1) for name in names},
    )
    report = checkpoint_load_report(
        model, checkpoint, {}, flash_attention=True
    )
    assert report["status"] == "passed"
    assert report["indexed_tensor_count"] == 806
    assert report["component_tensor_counts"] == {
        "language": 291,
        "projector": 4,
        "vision": 511,
    }
    assert report["meta_indexed_tensor_count"] == 0

    meta_state = {name: torch.zeros(1) for name in names}
    meta_state[names[-1]] = torch.empty(1, device="meta")
    model.state_dict = lambda: meta_state
    report = checkpoint_load_report(
        model, checkpoint, {}, flash_attention=True
    )
    assert report["status"] == "failed"
    assert report["errors"] == ["meta_indexed_keys"]
    assert report["meta_indexed_tensor_count"] == 1
    assert report["meta_indexed_keys"] == [names[-1]]

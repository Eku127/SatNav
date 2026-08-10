from pathlib import Path

from baselines.vlm.navila.artifacts import navila_model_identity
from baselines.vlm.navila.checkpoint import _compare_component


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


def test_component_comparison_uses_loader_semantics():
    result = _compare_component(
        {"loaded": _Tensor((1,)), "extra": _Tensor((2,))},
        {"loaded": (1,), "missing": (3,)},
    )
    assert result["missing_keys"] == ["missing"]
    assert result["unexpected_keys"] == ["extra"]
    assert result["mismatched_keys"] == []


def test_model_identity_covers_every_regular_root_file(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("model card", encoding="utf-8")
    for component in ("llm", "mm_projector", "vision_tower"):
        root = tmp_path / component
        root.mkdir()
        (root / "model.safetensors").write_bytes(component.encode("utf-8"))

    identity = navila_model_identity(tmp_path)

    assert identity["root"]["scope"] == "complete-root-file-snapshot"
    assert {entry["path"] for entry in identity["root"]["files"]} == {
        "README.md",
        "config.json",
        "generation_config.json",
    }

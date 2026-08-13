"""The generic evaluator must remain independent of ML/trainer stacks."""

import ast
from pathlib import Path

import satnav.evaluation
from satnav.core.episode import InstructionData, VLNEpisode
from satnav.evaluation._json import to_jsonable


def test_evaluation_package_does_not_import_torch_or_trainers():
    package_dir = Path(satnav.evaluation.__file__).parent
    forbidden_roots = {"torch", "torchvision"}
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.split(".")[0] in forbidden_roots for name in imports)
        assert not any(name.startswith("satnav.training") for name in imports)


def test_episode_serialization_never_leaks_runtime_scene_path():
    episode = VLNEpisode(
        episode_id="1",
        scene_id="logical-scene",
        start_position=[0.0, 0.0, 0.0],
        start_rotation=0.0,
        goals=[],
        reference_path=[],
        instruction=InstructionData("go"),
        trajectory_id="trajectory",
        split="val_seen",
        scene_path="/private/host/scenes/logical-scene.tif",
    )

    serialized = to_jsonable(episode)
    assert serialized["scene_id"] == "logical-scene"
    assert "scene_path" not in serialized
    assert "/private/host" not in str(serialized)

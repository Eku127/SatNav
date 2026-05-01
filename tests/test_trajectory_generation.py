import json
from pathlib import Path

from applications.trajectory_generation.generate import _load_episode_indices


def test_load_episode_indices_reads_list_of_ids(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps([3, {"episode_index": 5}, {"id": 8}]), encoding="utf-8")

    assert _load_episode_indices(path) == [3, 5, 8]


def test_load_episode_indices_reads_object_and_text(tmp_path: Path) -> None:
    object_path = tmp_path / "episodes_object.json"
    object_path.write_text(json.dumps({"episode_indices": [1, 2]}), encoding="utf-8")

    text_path = tmp_path / "episodes.txt"
    text_path.write_text("4, 6\n9\n", encoding="utf-8")

    assert _load_episode_indices(object_path) == [1, 2]
    assert _load_episode_indices(text_path) == [4, 6, 9]

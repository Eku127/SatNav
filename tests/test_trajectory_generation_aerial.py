import json
from pathlib import Path

from applications.trajectory_generation.aerial_quality import (
    compute_aerial_quality_score,
    passes_aerial_quality_thresholds,
)
from applications.trajectory_generation.generate import _load_episode_indices
from applications.trajectory_generation.produce_aerialsim_recommended import _build_targets


def test_aerial_quality_score_and_thresholds() -> None:
    row = {
        "avg_std": 50.0,
        "avg_height_range": 8.0,
        "avg_mean": 100.0,
        "avg_black_frac": 0.01,
    }
    row["score"] = compute_aerial_quality_score(**row)

    assert abs(row["score"] - 52.8) < 1e-9
    assert passes_aerial_quality_thresholds(
        row,
        min_avg_std=45.0,
        max_avg_black_frac=0.02,
        min_avg_height_range=6.0,
        min_score=52.0,
    )
    assert not passes_aerial_quality_thresholds(
        row,
        min_avg_std=55.0,
        max_avg_black_frac=0.02,
        min_avg_height_range=6.0,
    )


def test_load_episode_indices_reads_recommended_top_list(tmp_path: Path) -> None:
    path = tmp_path / "recommended_top.json"
    path.write_text(
        json.dumps(
            [
                {"trajectory_group_key": "a", "episode_index": 7},
                {"trajectory_group_key": "b", "representative_episode_index": 11},
            ]
        ),
        encoding="utf-8",
    )

    assert _load_episode_indices(path) == [7, 11]


def test_build_targets_expands_recommended_groups(tmp_path: Path) -> None:
    recommended_path = tmp_path / "recommended_top.json"
    groups_path = tmp_path / "trajectory_groups_full.json"
    recommended_path.write_text(
        json.dumps(
            [
                {
                    "trajectory_group_key": "scene/traj-a",
                    "scene_id": "city-a",
                    "trajectory_id": "traj-a",
                    "trajectory_type": "Road",
                    "representative_episode_index": 5,
                    "path_point_count": 10,
                    "avg_std": 50.0,
                    "avg_height_range": 8.0,
                    "avg_mean": 100.0,
                    "avg_black_frac": 0.0,
                    "score": 54.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    groups_path.write_text(
        json.dumps(
            [
                {
                    "trajectory_group_key": "scene/traj-a",
                    "scene_id": "nested/city-a",
                    "episode_count": 3,
                    "path_point_count": 10,
                    "episodes": [
                        {"episode_index": 7},
                        {"episode_index": 5},
                        {"episode_index": 6},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    targets, missing = _build_targets(
        recommended_path=recommended_path,
        groups_path=groups_path,
        max_trajectories=-1,
        min_score=None,
        min_avg_std=45.0,
        max_avg_black_frac=0.002,
        min_avg_height_range=6.0,
    )

    assert missing == []
    assert len(targets) == 1
    assert targets[0]["episode_indices"] == [5, 6, 7]
    assert targets[0]["scene_id"] == "city-a"
    assert targets[0]["representative_episode_index"] == 5

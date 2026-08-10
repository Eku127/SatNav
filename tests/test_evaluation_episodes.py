"""Fast tests for stable episode identity and deterministic sharding."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from satnav.evaluation import (
    EpisodeSelectionError,
    build_episode_plan,
    episode_seed,
    load_benchmark_manifest,
    stable_episode_key,
)


@dataclass
class FakeEpisode:
    scene_id: str
    episode_id: str


def test_stable_key_allows_bare_ids_to_repeat_across_scenes():
    first = FakeEpisode("scene-b", "7")
    second = FakeEpisode("scene-a", "7")

    assert stable_episode_key("val_seen", first) == "val_seen::scene-b::7"
    plan = build_episode_plan([first, second], split="val_seen")
    assert plan.selected_keys == (
        "val_seen::scene-a::7",
        "val_seen::scene-b::7",
    )


def test_sort_offset_limit_then_stride_has_exact_union():
    episodes = [
        FakeEpisode(f"scene-{index % 3}", str(index))
        for index in reversed(range(11))
    ]
    global_plan = build_episode_plan(
        episodes, split="val_unseen", offset=2, limit=7
    )
    shards = [
        build_episode_plan(
            episodes,
            split="val_unseen",
            offset=2,
            limit=7,
            rank=rank,
            world_size=3,
        )
        for rank in range(3)
    ]

    assert tuple(
        key
        for index in range(len(global_plan.selected_keys))
        for key in (shards[index % 3].shard_keys[index // 3],)
    ) == global_plan.selected_keys
    assert {key for shard in shards for key in shard.shard_keys} == set(
        global_plan.selected_keys
    )


def test_limit_zero_means_empty_not_all():
    plan = build_episode_plan(
        [FakeEpisode("scene", "0")], split="test", limit=0
    )
    assert plan.selected_keys == ()
    assert plan.shard_keys == ()


def test_duplicate_composite_key_is_rejected():
    duplicate = FakeEpisode("scene", "episode")
    with pytest.raises(EpisodeSelectionError, match="duplicate"):
        build_episode_plan([duplicate, duplicate], split="test")


def test_episode_seed_depends_on_key_not_rank_assignment():
    key = "val_seen::scene::42"
    assert episode_seed(123, key) == episode_seed(123, key)
    assert episode_seed(123, key) != episode_seed(124, key)
    assert episode_seed(123, key) != episode_seed(123, key + "x")


def test_tracked_v01_official_and_smoke_manifests_are_path_free():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "benchmark"
    expected = {
        "satnav_v0_1_val_seen.json": ("official", 4574, 500),
        "satnav_v0_1_val_unseen.json": ("official", 8756, 500),
        "satnav_v0_1_val_seen_smoke.json": ("smoke", 4574, 5),
        "satnav_v0_1_val_unseen_smoke.json": ("smoke", 8756, 5),
    }
    for filename, (kind, count, max_steps) in expected.items():
        manifest = load_benchmark_manifest(config_dir / filename)
        assert manifest.kind == kind
        assert manifest.episode_count == count
        assert manifest.max_episode_steps == max_steps
        assert manifest.success_threshold == {
            "Boundary": 10.0,
            "LandmarkSet": 30.0,
            "Road": 10.0,
        }
        assert "oracle_success" in manifest.required_metrics
        assert "/mnt/" not in str(manifest)


def test_generation_and_evaluation_landmark_radii_are_separate():
    repository_root = Path(__file__).resolve().parents[1]
    generation = OmegaConf.load(repository_root / "configs" / "satnav_task.yaml")
    evaluation = OmegaConf.load(
        repository_root / "configs" / "satnav_eval_task.yaml"
    )

    assert generation.TASK.SUCCESS_DISTANCE.LandmarkSet == 3.0
    assert evaluation.TASK.SUCCESS_DISTANCE.LandmarkSet == 30.0


@pytest.mark.parametrize("baseline", ("streamvln", "navila", "uninavid", "openfly"))
def test_vlm_task_configs_match_canonical_evaluation_contract(baseline):
    repository_root = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark_manifest(
        repository_root / "configs" / "benchmark" / "satnav_v0_1_val_seen.json"
    )
    task_config = OmegaConf.load(
        repository_root
        / "baselines"
        / "vlm"
        / baseline
        / "configs"
        / "satnav_task.yaml"
    )
    assert tuple(task_config.TASK.POSSIBLE_ACTIONS) == benchmark.action_space
    assert float(task_config.SIMULATOR.FORWARD_STEP_SIZE) == (
        benchmark.forward_step_size
    )
    assert float(task_config.SIMULATOR.TURN_ANGLE) == benchmark.turn_angle
    success_distances = OmegaConf.to_container(
        task_config.TASK.SUCCESS_DISTANCE, resolve=True
    )
    assert success_distances["DEFAULT"] == 10.0
    assert {
        name: success_distances[name] for name in benchmark.success_threshold
    } == benchmark.success_threshold

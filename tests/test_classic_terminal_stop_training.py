#!/usr/bin/env python3
"""Regression tests for terminal STOP supervision in classic IL data."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from satnav.dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
from satnav.dataset.recollect_dataset import RecollectionDataset
from satnav.task.actions import Action
from satnav.training.utils import collate_fn


class _Vocabulary:
    @staticmethod
    def tokens_to_indices(tokens):
        return [2] * len(tokens)


def _offline_dataset(images_root: Path) -> OfflineTrajectoryDataset:
    dataset = OfflineTrajectoryDataset.__new__(OfflineTrajectoryDataset)
    dataset.images_root = images_root
    dataset.target_rgb_size = 2
    dataset.max_instruction_len = 4
    dataset.vocab = _Vocabulary()
    dataset.inflec_weights = torch.tensor([1.0, 1.0])
    return dataset


def _write_rgb_frames(frame_dir: Path, values) -> None:
    frame_dir.mkdir(parents=True)
    for index, value in enumerate(values, start=1):
        rgb = np.full((2, 2, 3), value, dtype=np.uint8)
        Image.fromarray(rgb).save(
            frame_dir / f"{index:03d}.jpg",
            quality=100,
            subsampling=0,
        )


def test_offline_canonical_actions_keep_exactly_one_terminal_stop(tmp_path):
    dataset = _offline_dataset(tmp_path / "images")

    prev_actions, teacher_actions = dataset._action_sequences(
        {"actions": [-1, 1, 2, 0, 0, 3]}
    )

    assert teacher_actions == [1, 2, 0]
    assert prev_actions == [-1, 1, 2]
    assert teacher_actions.count(0) == 1
    assert prev_actions[-1] == teacher_actions[-2]


def test_offline_legacy_boundaries_do_not_fabricate_stop(tmp_path):
    dataset = _offline_dataset(tmp_path / "images")
    cases = (
        ([-1, 1, 2], ([-1, 1], [1, 2])),
        ([1, 2], ([-1, 1], [1, 2])),
        ([1, 2, 0], ([-1, 1, 2], [1, 2, 0])),
    )

    for actions, expected in cases:
        assert dataset._action_sequences({"actions": actions}) == expected


def test_offline_batch_and_loss_use_pre_stop_frame_and_stop_target(tmp_path):
    images_root = tmp_path / "images"
    video = "images/example-episode"
    _write_rgb_frames(
        images_root / "example-episode" / "rgb",
        [20, 80, 140, 240],
    )
    annotation = {
        "id": 0,
        "video": video,
        "instructions": ["move and stop"],
        "actions": [-1, 1, 2, 0],
    }
    dataset = _offline_dataset(images_root)
    dataset.annotations = [annotation]

    observations, prev_actions, teacher_actions, weights = dataset[0]

    assert teacher_actions.tolist() == [1, 2, 0]
    assert prev_actions.tolist() == [0, 1, 2]
    assert observations["rgb"].shape[0] == 3
    assert observations["rgb"][:, 0, 0, 0].tolist() == [20, 80, 140]

    _, _, _, batch_targets, batch_weights = collate_fn(
        [(observations, prev_actions, teacher_actions, weights)]
    )
    assert batch_targets[:, 0].tolist() == [1, 2, 0]
    assert batch_weights[-1, 0].item() > 0.0

    logits = torch.zeros(3, 1, 4, requires_grad=True)
    per_step_loss = F.cross_entropy(
        logits.permute(0, 2, 1),
        batch_targets,
        reduction="none",
    )
    loss = (per_step_loss * batch_weights).sum() / batch_weights.sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[-1, 0, Action.get_action_index(Action.STOP)] < 0


class _Simulator:
    def reset(self, scene_id):
        self.scene_id = scene_id

    def set_agent_state(self, position, rotation):
        self.position = position
        self.rotation = rotation


class _Follower:
    @staticmethod
    def follow_path(reference_path, simulator, execute):
        assert execute is True
        return [
            Action.MOVE_FORWARD,
            Action.TURN_LEFT,
            Action.STOP,
            Action.TURN_RIGHT,
        ]


def test_recollection_trajectory_appends_stop_with_previous_action():
    episode = SimpleNamespace(
        episode_id="episode-0",
        scene_id="scene-0",
        reference_path=[[0, 0, 0], [1, 0, 0]],
        start_position=[0, 0, 0],
        start_rotation=[0, 0, 0, 1],
    )
    dataset = RecollectionDataset.__new__(RecollectionDataset)
    dataset.env = SimpleNamespace(episodes=[episode], simulator=_Simulator())
    dataset.path_follower = _Follower()
    dataset._episode_indices = [0]
    dataset.config = {"IL": {"RECOLLECT_TRAINER": {"max_traj_len": 10}}}

    trajectories = dataset._extract_trajectories()

    assert trajectories["episode-0"] == [(0, 1), (1, 2), (2, 0)]


class _CollectionEnv:
    def __init__(self, episode):
        self.episodes = [episode]
        self.executed_actions = []

    @staticmethod
    def _observation(value):
        return {
            "rgb": np.full((2, 2, 3), value, dtype=np.uint8),
            "instruction": torch.tensor([1, 0]),
        }

    def reset_to_episode(self, episode):
        return self._observation(20)

    def step(self, action):
        self.executed_actions.append(action)
        return self._observation(140), False, {}


def test_recollection_collects_pre_stop_observation_without_post_stop_step():
    episode = SimpleNamespace(episode_id="episode-0")
    dataset = RecollectionDataset.__new__(RecollectionDataset)
    dataset.env = _CollectionEnv(episode)
    dataset.trajectories = {"episode-0": [(0, 1), (1, 0)]}
    dataset.target_rgb_size = 2
    dataset._tokenize_observation = lambda observation: observation

    collected = dataset._collect_episode(0)

    assert [sample[2] for sample in collected] == [1, 0]
    assert collected[-1][1:] == (1, 0)
    assert collected[-1][0]["rgb"][0, 0, 0].item() == 140
    assert dataset.env.executed_actions == [1]

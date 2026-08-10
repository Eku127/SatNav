#!/usr/bin/env python3
"""Tests that trajectory generators isolate follower state per episode."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from applications.trajectory_generation import generate_parallel
from applications.trajectory_generation.runner import SatNavTrajectoryRunner
from satnav.task.actions import Action

from tests.test_path_follower import FakeSimulator


class CountingFollower:
    def __init__(self):
        self.goal_radius = 3.0
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def get_next_action(self, goal_position, simulator):
        if self.reset_count == 0:
            raise AssertionError("follower used before episode reset")
        return Action.STOP


class FakeTrajectoryEnv:
    def __init__(self):
        self.simulator = FakeSimulator()
        self.max_episode_steps = 5
        self.reset_count = 0
        self.current_episode = None

    @property
    def agent_state(self):
        return self.simulator.get_agent_state()

    @staticmethod
    def _observation():
        return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    def reset_to_episode(self, episode):
        self.reset_count += 1
        self.current_episode = episode
        self.simulator.state.position = list(episode.start_position)
        return self._observation()

    def step(self, action):
        return self._observation(), True, {}


def make_episode():
    position = [0.0, 0.0, 0.0]
    return SimpleNamespace(
        episode_id="episode-0",
        trajectory_id="trajectory-0",
        trajectory_type=None,
        scene_id="test-scene",
        start_position=position,
        reference_path=[position, position],
        goals=[SimpleNamespace(position=position)],
        instruction=SimpleNamespace(instruction_text="stop here"),
    )


def make_config(output_path):
    scenes_dir = os.path.join(output_path, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    with open(os.path.join(scenes_dir, "test-scene.tif"), "wb") as handle:
        handle.write(b"test-scene-content")
    return SimpleNamespace(
        ENVIRONMENT=SimpleNamespace(MAX_EPISODE_STEPS=5),
        SIMULATOR=SimpleNamespace(
            TYPE="satsim",
            FORWARD_STEP_SIZE=10,
            TURN_ANGLE=15,
            RGB_SENSOR=SimpleNamespace(WIDTH=2, HEIGHT=2, HFOV=90),
        ),
        TASK=SimpleNamespace(SUCCESS_DISTANCE=3.0),
        DATASET=SimpleNamespace(SCENES_DIR=scenes_dir),
    )


class SerialTrajectoryLifecycleTests(unittest.TestCase):
    def test_runner_resets_follower_and_preserves_action_frame_alignment(self):
        with tempfile.TemporaryDirectory() as output_path:
            runner = SatNavTrajectoryRunner.__new__(SatNavTrajectoryRunner)
            runner.output_path = output_path
            runner.dataset_name = "satnav"
            runner._completed_episode_ids = set()
            runner.config = make_config(output_path)
            runner._scene_identity_cache = {}
            runner.env = FakeTrajectoryEnv()
            runner.path_follower = CountingFollower()

            annotation = runner._run_episode(0, make_episode())

            self.assertIsNotNone(annotation)
            self.assertEqual(runner.path_follower.reset_count, 1)
            self.assertEqual(annotation["actions"], [-1, 0])
            self.assertEqual(annotation["steps"], 1)
            rgb_dir = os.path.join(output_path, annotation["video"], "rgb")
            self.assertEqual(sorted(os.listdir(rgb_dir)), ["001.jpg", "002.jpg"])


class ParallelTrajectoryLifecycleTests(unittest.TestCase):
    def run_generator(self, output_path, prepare_legacy_marker=False):
        episode = make_episode()
        follower = CountingFollower()
        environment = FakeTrajectoryEnv()

        if prepare_legacy_marker:
            episode_dir, done_marker, _, _ = generate_parallel._episode_paths(
                output_path,
                0,
                episode.scene_id,
                "satnav",
            )
            os.makedirs(episode_dir, exist_ok=True)
            with open(done_marker, "w", encoding="utf-8"):
                pass

        replacements = {
            "_worker_env": environment,
            "_worker_path_follower": follower,
            "_worker_dataset": SimpleNamespace(episodes=[episode]),
            "_worker_output_path": output_path,
            "_worker_dataset_name": "satnav",
            "_worker_config": make_config(output_path),
            "_worker_scene_identity_cache": {},
        }
        with mock.patch.multiple(generate_parallel, **replacements):
            annotation = generate_parallel.process_single_episode(0)

        return annotation, follower, environment

    def test_parallel_generator_resets_worker_follower_per_episode(self):
        with tempfile.TemporaryDirectory() as output_path:
            annotation, follower, environment = self.run_generator(output_path)

            self.assertFalse(annotation.get("_failed", False))
            self.assertEqual(follower.reset_count, 1)
            self.assertEqual(environment.reset_count, 1)
            self.assertEqual(annotation["actions"], [-1, 0])

    def test_parallel_legacy_marker_forces_one_clean_full_rerender(self):
        with tempfile.TemporaryDirectory() as output_path:
            annotation, follower, environment = self.run_generator(
                output_path,
                prepare_legacy_marker=True,
            )

            self.assertFalse(annotation.get("_failed", False))
            self.assertEqual(follower.reset_count, 1)
            self.assertEqual(environment.reset_count, 1)
            rgb_dir = os.path.join(output_path, annotation["video"], "rgb")
            self.assertEqual(sorted(os.listdir(rgb_dir)), ["001.jpg", "002.jpg"])


if __name__ == "__main__":
    unittest.main()

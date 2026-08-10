"""Tests for the supported :class:`satnav.core.env.Env` surface."""

import unittest
from types import SimpleNamespace
from unittest import mock

from satnav.core.env import Env
from satnav.core.episode import InstructionData, NavigationGoal, VLNEpisode
from satnav.core.simulator import AgentState


def make_episode(episode_id="1"):
    return VLNEpisode(
        episode_id=episode_id,
        scene_id="logical-scene",
        start_position=[1.0, 2.0, 3.0],
        start_rotation=15.0,
        goals=[NavigationGoal([2.0, 3.0, 3.0])],
        reference_path=[[1.0, 2.0, 3.0], [2.0, 3.0, 3.0]],
        instruction=InstructionData("go"),
        trajectory_id="trajectory-1",
        split="val_seen",
        scene_path="/local/scenes/logical-scene",
    )


class FakeSimulator:
    def __init__(self):
        self.state = AgentState([0.0, 0.0, 0.0], 0.0)
        self.registered_paths = {}
        self.reset_scene_ids = []
        self.close_calls = 0

    def register_scene_path(self, scene_id, scene_path):
        self.registered_paths[scene_id] = scene_path

    def reset(self, scene_id):
        self.reset_scene_ids.append(scene_id)
        return {}

    def set_agent_state(self, position, rotation):
        self.state = AgentState(position, rotation)

    def get_agent_state(self):
        return self.state

    def step(self, action):
        return {"rgb": action}

    def close(self):
        self.close_calls += 1


class FakeTask:
    def __init__(self, config, simulator):
        self.simulator = simulator
        self.is_stop_called = False
        self.metrics = {"path_length": 0.0}

    def reset(self, episode):
        self.is_stop_called = False
        self.metrics = {"path_length": 0.0}
        self.simulator.reset(episode.scene_id)
        self.simulator.set_agent_state(
            episode.start_position, episode.start_rotation
        )
        return {"rgb": "reset"}

    def step(self, action):
        action_name = action.get("action") if isinstance(action, dict) else action
        self.is_stop_called = action_name in ("STOP", 0)
        self.metrics["path_length"] += 1.0
        return self.simulator.step(action)

    def get_metrics(self):
        return dict(self.metrics)


class EnvPublicApiTests(unittest.TestCase):
    def build_env(self, episodes=None, max_steps=2):
        simulator = FakeSimulator()
        dataset = SimpleNamespace(
            episodes=list(episodes or [make_episode()]),
            scenes_dir="/local/scenes",
        )
        dataset.get_episode_iterator = lambda: iter(dataset.episodes)
        config = {
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": max_steps},
            "SIMULATOR": {
                "TYPE": "fake",
                "RGB_SENSOR": {"WIDTH": 448, "HEIGHT": 448},
            },
            "TASK": {},
        }
        with mock.patch("satnav.core.env.create_simulator", return_value=simulator), mock.patch(
            "satnav.core.env.VLNTask", FakeTask
        ):
            environment = Env(config, dataset=dataset)
        return environment, dataset, simulator

    def test_public_state_and_step_info_lifecycle(self):
        environment, dataset, simulator = self.build_env()

        self.assertIs(environment.episodes, dataset.episodes)
        self.assertIsNone(environment.current_episode)
        self.assertIs(environment.simulator, simulator)
        self.assertIsNone(environment.last_step_info)
        self.assertFalse(environment.episode_over)
        self.assertEqual(environment.max_episode_steps, 2)
        self.assertEqual(
            environment.observation_space["rgb"]["shape"], (448, 448, 3)
        )

        observation = environment.reset()
        self.assertEqual(observation, {"rgb": "reset"})
        self.assertIs(environment.current_episode, dataset.episodes[0])
        self.assertEqual(environment.agent_state.position, [1.0, 2.0, 3.0])
        self.assertEqual(simulator.reset_scene_ids, ["logical-scene"])
        self.assertEqual(
            simulator.registered_paths,
            {"logical-scene": "/local/scenes/logical-scene"},
        )
        self.assertIsNone(environment.last_step_info)

        _, done, info = environment.step("MOVE_FORWARD")
        self.assertFalse(done)
        self.assertIs(environment.last_step_info, info)
        self.assertEqual(info["episode_key"], "val_seen::logical-scene::1")
        self.assertEqual(info["scene_id"], "logical-scene")
        self.assertFalse(info["stop_called"])
        self.assertIsNone(info["termination_reason"])

        _, done, info = environment.step("TURN_RIGHT")
        self.assertTrue(done)
        self.assertTrue(environment.episode_over)
        self.assertTrue(info["max_steps_reached"])
        self.assertEqual(info["termination_reason"], "max_steps")

        environment.reset_to_episode(dataset.episodes[0])
        self.assertFalse(environment.episode_over)
        self.assertIsNone(environment.last_step_info)
        _, done, info = environment.step("STOP")
        self.assertTrue(done)
        self.assertTrue(info["stop_called"])
        self.assertEqual(info["termination_reason"], "stop")

    def test_close_is_idempotent(self):
        environment, _, simulator = self.build_env()
        environment.close()
        environment.close()
        self.assertEqual(simulator.close_calls, 1)

    def test_reset_requires_a_dataset(self):
        simulator = FakeSimulator()
        with mock.patch("satnav.core.env.create_simulator", return_value=simulator), mock.patch(
            "satnav.core.env.VLNTask", FakeTask
        ):
            environment = Env(
                {
                    "ENVIRONMENT": {},
                    "SIMULATOR": {"TYPE": "fake"},
                    "TASK": {},
                }
            )
        self.assertEqual(environment.episodes, [])
        with self.assertRaisesRegex(RuntimeError, "dataset is not available"):
            environment.reset()


if __name__ == "__main__":
    unittest.main()

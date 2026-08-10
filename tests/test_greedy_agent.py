#!/usr/bin/env python3
"""Regression tests for the waypoint lifecycle in GreedyAgent."""

import unittest
from types import SimpleNamespace

import torch

from satnav.models.baselines.greedy_agent import GreedyAgent
from satnav.task.actions import Action

from tests.test_path_follower import FakeSimulator, point_at


def make_episode(episode_id, reference_path, goal=None):
    goal_position = goal or reference_path[-1]
    return SimpleNamespace(
        episode_id=episode_id,
        scene_id="test-scene",
        reference_path=reference_path,
        goals=[SimpleNamespace(position=goal_position)],
    )


class FakeEnv:
    def __init__(self, episode, simulator):
        self.current_episode = episode
        self.simulator = simulator

    @property
    def agent_state(self):
        return self.simulator.get_agent_state()


class GreedyAgentWaypointTests(unittest.TestCase):
    def setUp(self):
        self.agent = GreedyAgent(
            goal_radius=3.0,
            turn_angle=20.0,
            dim_actions=4,
        )
        self.rnn_states = torch.zeros(1, 1, 1)

    def act(self):
        action, returned_states = self.agent.act(
            observations={},
            rnn_states=self.rnn_states,
            prev_actions=None,
            masks=None,
        )
        self.assertIs(returned_states, self.rnn_states)
        return int(action.item())

    def test_dense_waypoints_do_not_emit_premature_stop(self):
        simulator = FakeSimulator()
        episode = make_episode(
            "dense",
            [
                [0.0, 0.0, 0.0],
                point_at(1.0, 0.0),
                point_at(2.0, 10.0),
                point_at(100.0, 0.0),
            ],
        )
        self.agent.set_env(FakeEnv(episode, simulator))

        action = self.act()

        self.assertEqual(action, Action.get_action_index(Action.MOVE_FORWARD))
        self.assertEqual(self.agent._current_waypoint_idx, 3)

    def test_stop_is_returned_when_every_waypoint_is_reached(self):
        simulator = FakeSimulator()
        episode = make_episode(
            "complete",
            [
                [0.0, 0.0, 0.0],
                point_at(1.0, 0.0),
                point_at(2.0, 30.0),
            ],
        )
        self.agent.set_env(FakeEnv(episode, simulator))

        self.assertEqual(self.act(), Action.get_action_index(Action.STOP))

    def test_new_episode_resets_path_follower_hysteresis(self):
        simulator = FakeSimulator()
        first_episode = make_episode(
            "first",
            [[0.0, 0.0, 0.0], point_at(100.0, 0.0)],
        )
        env = FakeEnv(first_episode, simulator)
        self.agent.set_env(env)
        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.MOVE_FORWARD),
        )

        env.current_episode = make_episode(
            "second",
            [[0.0, 0.0, 0.0], point_at(100.0, 12.0)],
        )

        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.TURN_RIGHT),
        )

    def test_distinct_episodes_with_same_id_are_reset(self):
        simulator = FakeSimulator()
        first_episode = make_episode(
            "shared-id",
            [[0.0, 0.0, 0.0], point_at(100.0, 0.0)],
        )
        env = FakeEnv(first_episode, simulator)
        self.agent.set_env(env)
        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.MOVE_FORWARD),
        )

        env.current_episode = make_episode(
            "shared-id",
            [[0.0, 0.0, 0.0], point_at(100.0, 12.0)],
        )

        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.TURN_RIGHT),
        )

    def test_sparse_waypoint_action_is_unchanged(self):
        simulator = FakeSimulator()
        episode = make_episode(
            "sparse",
            [[0.0, 0.0, 0.0], point_at(100.0, 90.0)],
        )
        self.agent.set_env(FakeEnv(episode, simulator))

        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.TURN_RIGHT),
        )

    def test_goal_fallback_without_reference_path_is_unchanged(self):
        simulator = FakeSimulator()
        goal = point_at(100.0, 0.0)
        episode = SimpleNamespace(
            episode_id="fallback",
            scene_id="test-scene",
            reference_path=[],
            goals=[SimpleNamespace(position=goal)],
        )
        self.agent.set_env(FakeEnv(episode, simulator))

        self.assertEqual(
            self.act(),
            Action.get_action_index(Action.MOVE_FORWARD),
        )


if __name__ == "__main__":
    unittest.main()

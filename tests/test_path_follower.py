#!/usr/bin/env python3
"""Regression tests for SatNav path-following behavior."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

from satnav.navigation.path_follower import (
    ReferencePathFollower,
    SatNavPathFollower,
    calculate_bearing,
    normalize_angle_diff,
)
from satnav.task.actions import Action


EARTH_RADIUS_METERS = 6371000.0


def point_at(distance_meters, bearing_degrees):
    """Return a small equatorial displacement from [0, 0, 0]."""
    angular_distance = distance_meters / EARTH_RADIUS_METERS
    bearing = math.radians(bearing_degrees)
    return [
        math.degrees(math.sin(bearing) * angular_distance),
        math.degrees(math.cos(bearing) * angular_distance),
        0.0,
    ]


class FakeSimulator:
    """Minimal simulator with deterministic geographic motion."""

    def __init__(
        self,
        position=None,
        heading=0.0,
        forward_step_meters=1.0,
        turn_angle=15.0,
    ):
        self.state = SimpleNamespace(
            position=list(position or [0.0, 0.0, 0.0]),
            rotation=float(heading),
        )
        self.forward_step_meters = float(forward_step_meters)
        self.turn_angle = float(turn_angle)
        self.actions = []

    def get_agent_state(self):
        return self.state

    def step(self, action):
        self.actions.append(action)
        if action == Action.TURN_LEFT:
            self.state.rotation = (self.state.rotation - self.turn_angle) % 360
        elif action == Action.TURN_RIGHT:
            self.state.rotation = (self.state.rotation + self.turn_angle) % 360
        elif action == Action.MOVE_FORWARD:
            heading = math.radians(self.state.rotation)
            angular_distance = self.forward_step_meters / EARTH_RADIUS_METERS
            latitude = self.state.position[1]
            cos_latitude = math.cos(math.radians(latitude))
            self.state.position[0] += math.degrees(
                math.sin(heading) * angular_distance / cos_latitude
            )
            self.state.position[1] += math.degrees(
                math.cos(heading) * angular_distance
            )


class BearingUtilityTests(unittest.TestCase):
    def test_cardinal_bearings_remain_unchanged(self):
        cases = [
            ((0.0, 0.0, 0.0, 1.0), 0.0),
            ((0.0, 0.0, 1.0, 0.0), 90.0),
            ((0.0, 0.0, 0.0, -1.0), 180.0),
            ((0.0, 0.0, -1.0, 0.0), 270.0),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertAlmostEqual(calculate_bearing(*arguments), expected)

    def test_angle_normalization_remains_unchanged(self):
        cases = [
            (0.0, 0.0),
            (181.0, -179.0),
            (-181.0, 179.0),
            (540.0, 180.0),
            (-540.0, -180.0),
        ]
        for angle, expected in cases:
            with self.subTest(angle=angle):
                self.assertEqual(normalize_angle_diff(angle), expected)


class SatNavPathFollowerTests(unittest.TestCase):
    def test_existing_direction_decision_matrix_is_preserved(self):
        cases = [
            (0.0, 0.0, Action.MOVE_FORWARD),
            (90.0, 0.0, Action.TURN_RIGHT),
            (270.0, 0.0, Action.TURN_LEFT),
            (90.0, 90.0, Action.MOVE_FORWARD),
            (0.0, 90.0, Action.TURN_LEFT),
        ]
        for goal_bearing, heading, expected in cases:
            with self.subTest(goal_bearing=goal_bearing, heading=heading):
                follower = SatNavPathFollower(
                    goal_radius=3.0,
                    turn_angle=15.0,
                )
                simulator = FakeSimulator(heading=heading)
                self.assertEqual(
                    follower.get_next_action(
                        point_at(100.0, goal_bearing),
                        simulator,
                    ),
                    expected,
                )

    def test_goal_radius_boundary_returns_stop(self):
        follower = SatNavPathFollower(goal_radius=3.0)
        simulator = FakeSimulator()
        with mock.patch(
            "satnav.navigation.path_follower.geodesic_distance",
            return_value=3.0,
        ):
            action = follower.get_next_action(point_at(100.0, 0.0), simulator)
        self.assertEqual(action, Action.STOP)

    def test_integer_action_output_is_preserved(self):
        follower = SatNavPathFollower(
            goal_radius=3.0,
            return_action_string=False,
        )
        simulator = FakeSimulator()
        self.assertEqual(
            follower.get_next_action(point_at(100.0, 90.0), simulator),
            Action.get_action_index(Action.TURN_RIGHT),
        )

    def test_hysteresis_is_preserved_within_one_episode(self):
        follower = SatNavPathFollower(goal_radius=3.0, turn_angle=20.0)
        simulator = FakeSimulator()

        self.assertEqual(
            follower.get_next_action(point_at(100.0, 0.0), simulator),
            Action.MOVE_FORWARD,
        )
        self.assertEqual(
            follower.get_next_action(point_at(100.0, 12.0), simulator),
            Action.MOVE_FORWARD,
        )

    def test_reset_prevents_hysteresis_from_leaking_to_next_episode(self):
        follower = SatNavPathFollower(goal_radius=3.0, turn_angle=20.0)
        simulator = FakeSimulator()
        follower.get_next_action(point_at(100.0, 0.0), simulator)

        follower.reset()

        self.assertEqual(
            follower.get_next_action(point_at(100.0, 12.0), simulator),
            Action.TURN_RIGHT,
        )

    def test_action_sequence_uses_a_fresh_history_and_string_actions(self):
        follower = SatNavPathFollower(
            goal_radius=3.0,
            turn_angle=20.0,
            return_action_string=False,
        )
        simulator = FakeSimulator()
        follower.get_next_action(point_at(100.0, 0.0), simulator)

        actions = follower.get_action_sequence_to_goal(
            point_at(100.0, 12.0),
            simulator,
            max_steps=1,
            execute_actions=False,
        )

        self.assertEqual(actions, [Action.TURN_RIGHT])


class ReferencePathFollowerTests(unittest.TestCase):
    def test_requires_reset_before_navigation(self):
        follower = ReferencePathFollower()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            follower.get_next_action(FakeSimulator())

    def test_sparse_path_keeps_existing_next_action(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        follower.reset([[0.0, 0.0, 0.0], point_at(100.0, 0.0)])

        action = follower.get_next_action(FakeSimulator())

        self.assertEqual(action, Action.MOVE_FORWARD)
        self.assertEqual(follower.get_current_waypoint_index(), 1)

    def test_dense_path_skips_all_consecutive_reached_waypoints(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        path = [
            [0.0, 0.0, 0.0],
            point_at(1.0, 0.0),
            point_at(2.0, 0.0),
            point_at(100.0, 0.0),
        ]
        follower.reset(path)

        action = follower.get_next_action(FakeSimulator())

        self.assertEqual(action, Action.MOVE_FORWARD)
        self.assertEqual(follower.get_current_waypoint_index(), 3)

    def test_dense_path_stops_only_when_all_remaining_points_are_reached(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        path = [
            [0.0, 0.0, 0.0],
            point_at(1.0, 0.0),
            point_at(2.0, 15.0),
            point_at(2.9, 30.0),
        ]
        follower.reset(path)

        action = follower.get_next_action(FakeSimulator())

        self.assertEqual(action, Action.STOP)
        self.assertEqual(follower.get_current_waypoint_index(), len(path))

    def test_waypoint_at_goal_radius_is_treated_as_reached(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        path = [
            [0.0, 0.0, 0.0],
            point_at(100.0, 0.0),
            point_at(100.0, 90.0),
        ]
        follower.reset(path)
        with mock.patch(
            "satnav.navigation.path_follower.geodesic_distance",
            side_effect=[3.0, 100.0, 100.0],
        ):
            action = follower.get_next_action(FakeSimulator())

        self.assertEqual(action, Action.TURN_RIGHT)
        self.assertEqual(follower.get_current_waypoint_index(), 2)

    def test_reset_clears_nested_follower_history(self):
        follower = ReferencePathFollower(goal_radius=3.0, turn_angle=20.0)
        simulator = FakeSimulator()
        follower.reset([[0.0, 0.0, 0.0], point_at(100.0, 0.0)])
        self.assertEqual(follower.get_next_action(simulator), Action.MOVE_FORWARD)

        follower.reset([[0.0, 0.0, 0.0], point_at(100.0, 12.0)])

        self.assertEqual(follower.get_next_action(simulator), Action.TURN_RIGHT)

    def test_empty_and_start_only_paths_stop_safely(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        for path in ([], [[0.0, 0.0, 0.0]]):
            with self.subTest(path=path):
                follower.reset(path)
                self.assertEqual(
                    follower.get_next_action(FakeSimulator()),
                    Action.STOP,
                )

    def test_completed_path_does_not_query_simulator_again(self):
        follower = ReferencePathFollower(goal_radius=3.0)
        follower.reset([])
        simulator = mock.Mock()
        simulator.get_agent_state.side_effect = AssertionError(
            "completed path should not read simulator state"
        )

        self.assertEqual(follower.get_next_action(simulator), Action.STOP)
        simulator.get_agent_state.assert_not_called()

    def test_follow_path_executes_motion_and_finishes_with_stop(self):
        follower = ReferencePathFollower(goal_radius=0.1, turn_angle=15.0)
        simulator = FakeSimulator(forward_step_meters=2.0)
        path = [[0.0, 0.0, 0.0], point_at(2.0, 0.0)]

        actions = follower.follow_path(
            path,
            simulator,
            execute=True,
            max_steps=5,
        )

        self.assertEqual(actions, [Action.MOVE_FORWARD, Action.STOP])
        self.assertEqual(simulator.actions, [Action.MOVE_FORWARD])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Action-driven consistency tests for AgentPoseSensor.

This test validates that AgentPoseSensor output matches pose computed from
executed actions and current config settings.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from satnav.core.utils import wrap_heading_deg
from satnav.sims.satsim_wrapper import SatSimWrapper
from satnav.task.sensors import AgentPoseSensor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "satnav_task.yaml"
TEST_MAP_PATH = Path(__file__).resolve().parent / "test_data" / "map.tif"


def _expected_pose_from_actions(step_size, turn_angle, actions):
    """Compute expected ego pose from action sequence."""
    delta_forward = 0.0
    delta_right = 0.0
    delta_heading = 0.0
    expected = []

    for action in actions:
        if action == "TURN_LEFT":
            delta_heading = wrap_heading_deg(delta_heading - turn_angle)
        elif action == "TURN_RIGHT":
            delta_heading = wrap_heading_deg(delta_heading + turn_angle)
        elif action == "MOVE_FORWARD":
            heading_rad = math.radians(delta_heading)
            delta_forward += step_size * math.cos(heading_rad)
            delta_right += step_size * math.sin(heading_rad)

        heading_rad = math.radians(delta_heading)
        expected.append(
            np.array(
                [
                    delta_forward,
                    delta_right,
                    math.sin(heading_rad),
                    math.cos(heading_rad),
                ],
                dtype=np.float32,
            )
        )
    return expected


@pytest.mark.skipif(not TEST_MAP_PATH.exists(), reason="tests/test_data/map.tif not found")
def test_agent_pose_matches_action_integral_with_current_config():
    """Use config settings and verify action-computed pose equals sensor output."""
    config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    step_size = float(config.SIMULATOR.FORWARD_STEP_SIZE)
    turn_angle = float(config.SIMULATOR.TURN_ANGLE)

    sim = SatSimWrapper(config)
    sim.reset(str(TEST_MAP_PATH))

    start_position = [114.064413, 22.543496, 100.0]
    start_heading = 45.0
    sim.set_agent_state(start_position, start_heading)

    sensor = AgentPoseSensor(simulator=sim)
    episode = SimpleNamespace(
        start_position=start_position,
        start_rotation=start_heading,
    )
    sensor.reset(episode)

    actions = [
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "MOVE_FORWARD",
    ]
    expected_poses = _expected_pose_from_actions(step_size, turn_angle, actions)

    for idx, (action, expected_pose) in enumerate(zip(actions, expected_poses), start=1):
        sim.step(action)
        actual_pose = sensor.get_observation(simulator=sim)

        # Validate heading component both via sensor output and simulator state.
        state = sim.get_agent_state()
        expected_delta_heading = math.degrees(math.atan2(expected_pose[2], expected_pose[3]))
        actual_delta_heading = wrap_heading_deg(state.rotation - start_heading)
        assert abs(actual_delta_heading - expected_delta_heading) < 1e-4, (
            f"Heading mismatch at step {idx}: "
            f"action={action}, expected={expected_delta_heading:.6f}, got={actual_delta_heading:.6f}"
        )

        # Allow small numerical drift from geo projection conversion.
        assert np.allclose(actual_pose, expected_pose, atol=1.2), (
            f"Pose mismatch at step {idx}: action={action}\n"
            f"expected={expected_pose}\nactual={actual_pose}"
        )

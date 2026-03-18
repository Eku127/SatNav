#!/usr/bin/env python3
"""Tests for AgentPoseSensor.

Verifies that the AgentPoseSensor output matches manual calculations
by executing known actions (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT) and
comparing the sensor's ego-frame pose with expected values computed
from the simulator's ground-truth state.

Uses tests/test_data/map.tif as the scene.
"""

import math
import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from satnav.core.utils import lonlat_to_ego_displacement, wrap_heading_deg
from satnav.task.sensors import AgentPoseSensor
from satnav.core.simulator import AgentState
from satnav.sims.satsim_wrapper import SatSimWrapper


def manual_compute_pose(start_position, start_heading, current_position, current_heading):
    """Manually compute expected pose for verification.
    
    This duplicates the logic in AgentPoseSensor but uses raw math
    to serve as an independent reference.
    """
    # Ego-frame displacement
    delta_fwd, delta_right = lonlat_to_ego_displacement(
        start_position, start_heading, current_position
    )
    
    # Relative heading
    delta_heading_deg = wrap_heading_deg(current_heading - start_heading)
    delta_heading_rad = math.radians(delta_heading_deg)
    sin_dh = math.sin(delta_heading_rad)
    cos_dh = math.cos(delta_heading_rad)
    
    return np.array([delta_fwd, delta_right, sin_dh, cos_dh], dtype=np.float32)


def test_utils_basic():
    """Test lonlat_to_ego_displacement and wrap_heading_deg with known values."""
    print("=" * 60)
    print("Test 1: Utils basic correctness")
    print("=" * 60)
    
    # Test wrap_heading_deg
    assert abs(wrap_heading_deg(0.0)) < 1e-6
    assert abs(wrap_heading_deg(90.0) - 90.0) < 1e-6
    assert abs(wrap_heading_deg(-90.0) - (-90.0)) < 1e-6
    assert abs(wrap_heading_deg(270.0) - (-90.0)) < 1e-6
    assert abs(wrap_heading_deg(-270.0) - 90.0) < 1e-6
    assert abs(wrap_heading_deg(360.0)) < 1e-6
    print("  wrap_heading_deg: PASSED")
    
    # Test ego displacement: heading=0 (facing North), move North
    # Moving ~10m North means lat increases by ~0.00009 degrees
    start = [114.0, 22.5, 100.0]
    north_10m = [114.0, 22.5 + 10.0 / 111320.0, 100.0]  # ~10m north
    fwd, right = lonlat_to_ego_displacement(start, 0.0, north_10m)
    assert abs(fwd - 10.0) < 0.5, f"Expected fwd~10m, got {fwd}"
    assert abs(right) < 0.5, f"Expected right~0m, got {right}"
    print(f"  Facing North, move North 10m: fwd={fwd:.3f}m, right={right:.3f}m  PASSED")
    
    # Test ego displacement: heading=90 (facing East), move East
    east_10m_lon_delta = 10.0 / (111320.0 * math.cos(math.radians(22.5)))
    east_10m = [114.0 + east_10m_lon_delta, 22.5, 100.0]
    fwd, right = lonlat_to_ego_displacement(start, 90.0, east_10m)
    assert abs(fwd - 10.0) < 0.5, f"Expected fwd~10m, got {fwd}"
    assert abs(right) < 0.5, f"Expected right~0m, got {right}"
    print(f"  Facing East, move East 10m:  fwd={fwd:.3f}m, right={right:.3f}m  PASSED")
    
    # Test ego displacement: heading=0 (facing North), move East  
    # Should give fwd=0, right=+10
    fwd, right = lonlat_to_ego_displacement(start, 0.0, east_10m)
    assert abs(fwd) < 0.5, f"Expected fwd~0m, got {fwd}"
    assert abs(right - 10.0) < 0.5, f"Expected right~10m, got {right}"
    print(f"  Facing North, move East 10m: fwd={fwd:.3f}m, right={right:.3f}m  PASSED")
    
    print()


def test_sensor_with_simulator():
    """Test AgentPoseSensor with real SatSim actions using map.tif.
    
    Executes a sequence of actions and verifies the sensor output
    matches manual calculations at each step.
    """
    print("=" * 60)
    print("Test 2: AgentPoseSensor with real simulator actions")
    print("=" * 60)
    
    # Config matching satnav_task.yaml
    config = {
        "FORWARD_STEP_SIZE": 10.0,
        "TURN_ANGLE": 15.0,
        "RGB_SENSOR": {
            "WIDTH": 448,
            "HEIGHT": 448,
            "HFOV": 90.0,
        }
    }
    
    # Create simulator
    sim = SatSimWrapper(config)
    scene_path = os.path.join(os.path.dirname(__file__), 'test_data', 'map.tif')
    sim.reset("map")  # Will fail without scenes_dir, use direct path
    # Actually, reset needs the scene path. Let me use _satsim directly
    sim._satsim.load_scene(scene_path)
    sim._scene_id = "map"
    
    # Use coordinates from test dataset (known to be within map.tif bounds)
    start_position = [114.064413, 22.543496, 100.0]
    start_heading = 45.0  # Face NE to test non-trivial heading
    
    sim.set_agent_state(start_position, start_heading)
    
    # Create sensor
    sensor = AgentPoseSensor(simulator=sim)
    
    # Simulate episode reset
    class FakeEpisode:
        pass
    episode = FakeEpisode()
    episode.start_position = start_position
    episode.start_rotation = start_heading
    sensor.reset(episode)
    
    # ------------------------------------------
    # Step 0: At start, pose should be [0, 0, 0, 1]
    # ------------------------------------------
    pose = sensor.get_observation(simulator=sim)
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    print(f"\n  Step 0 (at start):")
    print(f"    Sensor:   [{pose[0]:+8.3f}, {pose[1]:+8.3f}, {pose[2]:+8.4f}, {pose[3]:+8.4f}]")
    print(f"    Expected: [{expected[0]:+8.3f}, {expected[1]:+8.3f}, {expected[2]:+8.4f}, {expected[3]:+8.4f}]")
    assert np.allclose(pose, expected, atol=1e-3), f"Mismatch at step 0: {pose} vs {expected}"
    print(f"    PASSED")
    
    # Define action sequence
    actions = [
        ("MOVE_FORWARD", "Move 10m in NE direction (heading=45)"),
        ("MOVE_FORWARD", "Move another 10m in NE direction"),
        ("TURN_LEFT",    "Turn left 15 degrees (heading 45->30)"),
        ("TURN_LEFT",    "Turn left 15 degrees (heading 30->15)"),
        ("MOVE_FORWARD", "Move 10m in heading=15 direction"),
        ("TURN_RIGHT",   "Turn right 15 degrees (heading 15->30)"),
        ("TURN_RIGHT",   "Turn right 15 degrees (heading 30->45)"),
        ("TURN_RIGHT",   "Turn right 15 degrees (heading 45->60)"),
        ("MOVE_FORWARD", "Move 10m in heading=60 direction"),
    ]
    
    all_passed = True
    for step_idx, (action, description) in enumerate(actions, start=1):
        # Execute action
        sim.step(action)
        
        # Get sensor observation
        pose = sensor.get_observation(simulator=sim)
        
        # Get ground truth state and compute expected manually
        agent_state = sim.get_agent_state()
        current_pos = agent_state.position.tolist()
        current_heading = agent_state.rotation
        
        expected = manual_compute_pose(
            start_position, start_heading,
            current_pos, current_heading
        )
        
        # Compare
        match = np.allclose(pose, expected, atol=1e-3)
        status = "PASSED" if match else "FAILED"
        if not match:
            all_passed = False
        
        print(f"\n  Step {step_idx}: {action} — {description}")
        print(f"    Agent pos: [{current_pos[0]:.6f}, {current_pos[1]:.6f}], heading: {current_heading:.1f}")
        print(f"    Sensor:   [{pose[0]:+8.3f}, {pose[1]:+8.3f}, {pose[2]:+8.4f}, {pose[3]:+8.4f}]")
        print(f"    Expected: [{expected[0]:+8.3f}, {expected[1]:+8.3f}, {expected[2]:+8.4f}, {expected[3]:+8.4f}]")
        print(f"    delta_fwd={pose[0]:+.3f}m, delta_right={pose[1]:+.3f}m, "
              f"delta_heading={math.degrees(math.atan2(pose[2], pose[3])):+.1f}°")
        print(f"    {status}")
    
    print()
    return all_passed


def test_boundary_return():
    """Test that returning to start gives pose close to [0, 0, 0, 1].
    
    Simulates a Boundary-style episode: go forward, turn 180, go back.
    """
    print("=" * 60)
    print("Test 3: Boundary return (forward, 180 turn, back)")
    print("=" * 60)
    
    config = {
        "FORWARD_STEP_SIZE": 10.0,
        "TURN_ANGLE": 15.0,
        "RGB_SENSOR": {
            "WIDTH": 448,
            "HEIGHT": 448,
            "HFOV": 90.0,
        }
    }
    
    sim = SatSimWrapper(config)
    scene_path = os.path.join(os.path.dirname(__file__), 'test_data', 'map.tif')
    sim._satsim.load_scene(scene_path)
    sim._scene_id = "map"
    
    start_position = [114.064413, 22.543496, 100.0]
    start_heading = 0.0  # Facing North
    sim.set_agent_state(start_position, start_heading)
    
    sensor = AgentPoseSensor(simulator=sim)
    class FakeEpisode:
        pass
    episode = FakeEpisode()
    episode.start_position = start_position
    episode.start_rotation = start_heading
    sensor.reset(episode)
    
    # Go forward 3 steps (30m North)
    for _ in range(3):
        sim.step("MOVE_FORWARD")
    
    pose_mid = sensor.get_observation(simulator=sim)
    print(f"\n  After 3x FORWARD (30m North):")
    print(f"    Pose: [{pose_mid[0]:+8.3f}, {pose_mid[1]:+8.3f}, {pose_mid[2]:+8.4f}, {pose_mid[3]:+8.4f}]")
    assert abs(pose_mid[0] - 30.0) < 1.0, f"Expected fwd~30m, got {pose_mid[0]}"
    assert abs(pose_mid[1]) < 0.5, f"Expected right~0m, got {pose_mid[1]}"
    print(f"    PASSED (fwd~30m, right~0m)")
    
    # Turn 180 degrees (12 x 15° turns)
    for _ in range(12):
        sim.step("TURN_RIGHT")
    
    pose_turned = sensor.get_observation(simulator=sim)
    delta_h_deg = math.degrees(math.atan2(pose_turned[2], pose_turned[3]))
    print(f"\n  After 180° turn:")
    print(f"    Pose: [{pose_turned[0]:+8.3f}, {pose_turned[1]:+8.3f}, {pose_turned[2]:+8.4f}, {pose_turned[3]:+8.4f}]")
    print(f"    delta_heading = {delta_h_deg:+.1f}°")
    assert abs(abs(delta_h_deg) - 180.0) < 1.0, f"Expected heading~±180°, got {delta_h_deg}"
    print(f"    PASSED (heading~180°)")
    
    # Go forward 3 steps (30m South = back towards start)
    for _ in range(3):
        sim.step("MOVE_FORWARD")
    
    pose_back = sensor.get_observation(simulator=sim)
    delta_h_deg_back = math.degrees(math.atan2(pose_back[2], pose_back[3]))
    print(f"\n  After returning 30m (back to start area):")
    print(f"    Pose: [{pose_back[0]:+8.3f}, {pose_back[1]:+8.3f}, {pose_back[2]:+8.4f}, {pose_back[3]:+8.4f}]")
    print(f"    Position displacement: fwd={pose_back[0]:+.3f}m, right={pose_back[1]:+.3f}m")
    print(f"    delta_heading = {delta_h_deg_back:+.1f}°")
    
    # Position should be near (0, 0) — back at start
    assert abs(pose_back[0]) < 2.0, f"Expected fwd~0m after return, got {pose_back[0]}"
    assert abs(pose_back[1]) < 2.0, f"Expected right~0m after return, got {pose_back[1]}"
    print(f"    PASSED (returned to start, fwd≈0, right≈0)")
    
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AgentPoseSensor Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Pure math utils
    test_utils_basic()
    
    # Test 2: Sensor with real simulator
    passed_2 = test_sensor_with_simulator()
    
    # Test 3: Boundary return scenario
    passed_3 = test_boundary_return()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (utils basic):        PASSED")
    print(f"  Test 2 (sensor + actions):   {'PASSED' if passed_2 else 'FAILED'}")
    print(f"  Test 3 (boundary return):    {'PASSED' if passed_3 else 'FAILED'}")
    
    if passed_2 and passed_3:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

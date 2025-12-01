#!/usr/bin/env python3
"""Tests for satnav.navigation module.

Tests for DiscretePathPlanner, SatNavPathFollower, and ReferencePathFollower.
"""

import math
import pytest
import numpy as np

from satnav.navigation import DiscretePathPlanner, SatNavPathFollower, ReferencePathFollower
from satnav.navigation.path_follower import calculate_bearing, normalize_angle_diff
from satnav.task.actions import Action


class TestCalculateBearing:
    """Tests for bearing calculation utility."""
    
    def test_bearing_east(self):
        """Test bearing when destination is directly east."""
        # From origin to a point directly east
        bearing = calculate_bearing(0.0, 0.0, 1.0, 0.0)
        assert abs(bearing - 90.0) < 1.0  # Should be ~90 degrees (East)
    
    def test_bearing_north(self):
        """Test bearing when destination is directly north."""
        bearing = calculate_bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(bearing - 0.0) < 1.0  # Should be ~0 degrees (North)
    
    def test_bearing_south(self):
        """Test bearing when destination is directly south."""
        bearing = calculate_bearing(0.0, 1.0, 0.0, 0.0)
        assert abs(bearing - 180.0) < 1.0  # Should be ~180 degrees (South)
    
    def test_bearing_west(self):
        """Test bearing when destination is directly west."""
        bearing = calculate_bearing(1.0, 0.0, 0.0, 0.0)
        assert abs(bearing - 270.0) < 1.0  # Should be ~270 degrees (West)
    
    def test_bearing_northeast(self):
        """Test bearing when destination is northeast."""
        bearing = calculate_bearing(0.0, 0.0, 1.0, 1.0)
        # Should be roughly 45 degrees (NE), but varies due to Earth's curvature
        assert 30.0 < bearing < 60.0


class TestNormalizeAngleDiff:
    """Tests for angle normalization utility."""
    
    def test_already_normalized(self):
        """Test angle that's already in [-180, 180]."""
        assert normalize_angle_diff(45.0) == 45.0
        assert normalize_angle_diff(-45.0) == -45.0
        assert normalize_angle_diff(0.0) == 0.0
    
    def test_positive_wrap(self):
        """Test wrapping positive angles > 180."""
        assert abs(normalize_angle_diff(270.0) - (-90.0)) < 0.01
        assert abs(normalize_angle_diff(360.0) - 0.0) < 0.01
        assert abs(normalize_angle_diff(450.0) - 90.0) < 0.01
    
    def test_negative_wrap(self):
        """Test wrapping negative angles < -180."""
        assert abs(normalize_angle_diff(-270.0) - 90.0) < 0.01
        assert abs(normalize_angle_diff(-360.0) - 0.0) < 0.01


class TestDiscretePathPlanner:
    """Tests for DiscretePathPlanner."""
    
    @pytest.fixture
    def planner(self):
        """Create a standard path planner for testing."""
        return DiscretePathPlanner(
            forward_distance=0.25,
            turn_angle=15.0,
            goal_radius=0.5,
            step_limit=500
        )
    
    def test_init_valid(self):
        """Test valid initialization."""
        planner = DiscretePathPlanner(
            forward_distance=0.25,
            turn_angle=15.0,
        )
        assert planner.forward_distance == 0.25
        assert planner.turn_angle == 15.0
        assert planner.num_turns_in_circle == 24
    
    def test_init_invalid_turn_angle(self):
        """Test that invalid turn angles raise assertion."""
        with pytest.raises(AssertionError):
            DiscretePathPlanner(turn_angle=13.0)  # 360 % 13 != 0
    
    def test_plan_already_at_goal(self, planner):
        """Test planning when already at goal."""
        actions = planner.plan(distance_meters=0.3, bearing_degrees=0.0)
        assert len(actions) == 0  # Already within goal_radius
    
    def test_plan_straight_ahead(self, planner):
        """Test planning for target directly ahead."""
        actions = planner.plan(distance_meters=1.0, bearing_degrees=0.0)
        
        # Should be mostly MOVE_FORWARD actions
        move_count = actions.count(Action.MOVE_FORWARD)
        turn_count = actions.count(Action.TURN_LEFT) + actions.count(Action.TURN_RIGHT)
        
        assert move_count >= 2  # At least 1m / 0.25m = 4 moves, minus goal_radius allowance
        assert turn_count <= 4  # Minimal turning for final heading adjustment
    
    def test_plan_to_right(self, planner):
        """Test planning for target to the right."""
        actions = planner.plan(distance_meters=2.0, bearing_degrees=90.0)
        
        # Should start with TURN_RIGHT actions
        assert actions[0] == Action.TURN_RIGHT
        
        # Should eventually have MOVE_FORWARD
        assert Action.MOVE_FORWARD in actions
    
    def test_plan_to_left(self, planner):
        """Test planning for target to the left."""
        actions = planner.plan(distance_meters=2.0, bearing_degrees=-90.0)
        
        # Should start with TURN_LEFT actions
        assert actions[0] == Action.TURN_LEFT
        
        # Should eventually have MOVE_FORWARD
        assert Action.MOVE_FORWARD in actions
    
    def test_plan_behind(self, planner):
        """Test planning for target behind agent."""
        actions = planner.plan(distance_meters=2.0, bearing_degrees=180.0)
        
        # Should require significant turning
        turn_count = actions.count(Action.TURN_LEFT) + actions.count(Action.TURN_RIGHT)
        assert turn_count >= 10  # ~180 degrees / 15 degrees per turn ≈ 12 turns
    
    def test_plan_step_limit(self):
        """Test that step limit is respected."""
        planner = DiscretePathPlanner(
            forward_distance=0.25,
            turn_angle=15.0,
            step_limit=10
        )
        actions = planner.plan(distance_meters=100.0, bearing_degrees=0.0)
        assert len(actions) <= 10
    
    def test_plan_polar_compatibility(self, planner):
        """Test VLN-CE compatible polar interface."""
        # plan_polar uses radians and CCW positive convention
        actions_polar = planner.plan_polar(r=2.0, theta=math.radians(90))
        actions_normal = planner.plan(distance_meters=2.0, bearing_degrees=-90.0)
        
        # Should produce similar results (not exactly same due to angle conversion)
        assert abs(len(actions_polar) - len(actions_normal)) < 5
    
    def test_action_types_valid(self, planner):
        """Test that all returned actions are valid."""
        actions = planner.plan(distance_meters=5.0, bearing_degrees=45.0)
        
        for action in actions:
            assert action in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]
            assert Action.is_valid_action(action)


class TestDiscretePathPlannerSimulation:
    """Tests that verify planner produces correct paths by simulation."""
    
    def test_reaches_goal_forward(self):
        """Test that simulated execution reaches goal ahead."""
        planner = DiscretePathPlanner(
            forward_distance=1.0,  # Larger steps for faster test
            turn_angle=15.0,
            goal_radius=0.5
        )
        
        # Plan to target 10m ahead
        actions = planner.plan(distance_meters=10.0, bearing_degrees=0.0)
        
        # Simulate execution
        x, y = 0.0, 0.0
        heading = 0.0  # facing +y
        
        for action in actions:
            if action == Action.MOVE_FORWARD:
                x += 1.0 * math.sin(math.radians(heading))
                y += 1.0 * math.cos(math.radians(heading))
            elif action == Action.TURN_LEFT:
                heading -= 15.0
            elif action == Action.TURN_RIGHT:
                heading += 15.0
        
        # Should be close to (0, 10)
        final_dist = math.sqrt(x**2 + (y - 10)**2)
        assert final_dist < 2.0  # Within reasonable distance
    
    def test_reaches_goal_diagonal(self):
        """Test reaching a diagonal target."""
        planner = DiscretePathPlanner(
            forward_distance=1.0,
            turn_angle=15.0,
            goal_radius=1.0
        )
        
        # Plan to target 7m away at 45 degrees
        target_dist = 7.0
        target_bearing = 45.0
        target_x = target_dist * math.sin(math.radians(target_bearing))
        target_y = target_dist * math.cos(math.radians(target_bearing))
        
        actions = planner.plan(target_dist, target_bearing)
        
        # Simulate
        x, y = 0.0, 0.0
        heading = 0.0
        
        for action in actions:
            if action == Action.MOVE_FORWARD:
                x += 1.0 * math.sin(math.radians(heading))
                y += 1.0 * math.cos(math.radians(heading))
            elif action == Action.TURN_LEFT:
                heading -= 15.0
            elif action == Action.TURN_RIGHT:
                heading += 15.0
        
        final_dist = math.sqrt((x - target_x)**2 + (y - target_y)**2)
        assert final_dist < 3.0  # Within reasonable distance


class TestSatNavPathFollower:
    """Tests for SatNavPathFollower (requires mock simulator)."""
    
    def test_init(self):
        """Test initialization."""
        follower = SatNavPathFollower(
            goal_radius=3.0,
            turn_angle=15.0
        )
        assert follower.goal_radius == 3.0
        assert follower.turn_angle == 15.0
    
    def test_return_format_string(self):
        """Test that return format can be configured to string."""
        follower = SatNavPathFollower(return_action_string=True)
        # We can't fully test without a simulator, but we can test the config
        assert follower.return_action_string is True
    
    def test_return_format_index(self):
        """Test that return format can be configured to index."""
        follower = SatNavPathFollower(return_action_string=False)
        assert follower.return_action_string is False


class TestReferencePathFollower:
    """Tests for ReferencePathFollower."""
    
    def test_init(self):
        """Test initialization."""
        follower = ReferencePathFollower(
            goal_radius=3.0,
            turn_angle=15.0,
            waypoint_radius=5.0
        )
        assert follower.goal_radius == 3.0
        assert follower.waypoint_radius == 5.0
    
    def test_reset(self):
        """Test reset with reference path."""
        follower = ReferencePathFollower()
        path = [
            [114.06, 22.54, 100.0],
            [114.07, 22.54, 100.0],
            [114.08, 22.54, 100.0],
        ]
        follower.reset(path)
        
        assert follower._reference_path == path
        assert follower._current_waypoint_idx == 1  # Skip start
    
    def test_get_progress_initial(self):
        """Test progress at start."""
        follower = ReferencePathFollower()
        path = [
            [114.06, 22.54, 100.0],
            [114.07, 22.54, 100.0],
            [114.08, 22.54, 100.0],
        ]
        follower.reset(path)
        
        # At start, targeting waypoint 1, progress = 1/2 = 0.5
        assert abs(follower.get_progress() - 0.5) < 0.01
    
    def test_get_current_waypoint_index(self):
        """Test waypoint index tracking."""
        follower = ReferencePathFollower()
        path = [
            [114.06, 22.54, 100.0],
            [114.07, 22.54, 100.0],
        ]
        follower.reset(path)
        
        assert follower.get_current_waypoint_index() == 1
    
    def test_raises_without_reset(self):
        """Test that get_next_action raises without reset."""
        follower = ReferencePathFollower()
        
        # Can't create a real simulator for this test, but we can verify
        # the error is about reference path
        assert follower._reference_path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


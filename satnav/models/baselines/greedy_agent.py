#!/usr/bin/env python3
"""Greedy Agent for SatNav VLN.

This module implements a greedy baseline agent that always navigates directly
toward the goal position using the shortest path strategy.

The agent reuses the SatNavPathFollower implementation to avoid code duplication
and ensure consistency with existing navigation logic.

Reference:
    - VLN-CE: vlnce_baselines/nonlearning_agents.py (HandcraftedAgent)
    - SatNav: satnav/navigation/path_follower.py (SatNavPathFollower)
    - Example: examples/satnav_path_follower_example.py
"""

from typing import List, Tuple

import torch
from omegaconf import DictConfig

from satnav.core.utils import geodesic_distance
from satnav.models.base import ILPolicy
from satnav.models.baselines.random_agent import DummyNet
from satnav.navigation import SatNavPathFollower


class GreedyAgent(ILPolicy):
    """Greedy baseline agent that navigates directly to the goal.
    
    This agent uses a greedy strategy to navigate: it always selects the action
    that moves the agent toward the goal position. The navigation logic is
    implemented by reusing SatNavPathFollower.
    
    Algorithm:
    1. Extract waypoints from episode's reference_path
    2. Navigate to current waypoint using greedy strategy:
       - Calculate distance and bearing to current waypoint
       - If distance < goal_radius: Move to next waypoint
       - If heading is close to target bearing: MOVE_FORWARD
       - Otherwise: TURN_LEFT or TURN_RIGHT toward waypoint
    3. When all waypoints reached: STOP
    
    This provides an oracle-like upper bound for VLN performance, since it
    knows the reference path waypoints and takes optimal actions.
    
    Note: This implementation follows the same logic as 
    examples/satnav_path_follower_example.py, navigating through waypoints
    sequentially rather than going directly to the final goal.
    
    Attributes:
        path_follower: SatNavPathFollower instance for navigation logic
        _env: Environment reference (set by evaluator via set_env())
        _waypoints: Current episode's waypoints
        _current_waypoint_idx: Index of current target waypoint
        _current_episode: Current episode object used for lifecycle detection
        
    Example:
        >>> config.MODEL.GREEDY_AGENT.goal_radius = 10.0
        >>> config.MODEL.GREEDY_AGENT.turn_angle = 15.0
        >>> agent = GreedyAgent.from_config(config, obs_space, act_space)
        >>> 
        >>> # Evaluator will call set_env to provide environment access
        >>> agent.set_env(env)
        >>> 
        >>> # Agent navigates through waypoints
        >>> action, rnn_states = agent.act(obs, rnn_states, prev_actions, masks)
    """
    
    def __init__(
        self,
        goal_radius: float,
        turn_angle: float,
        dim_actions: int
    ):
        """Initialize greedy agent.
        
        Args:
            goal_radius: Distance threshold for considering goal reached (meters).
                When agent is within this distance, STOP is returned.
            turn_angle: Angle rotated per TURN_LEFT/TURN_RIGHT action (degrees).
                Used to determine when agent is facing the goal.
            dim_actions: Number of discrete actions (should be 4)
        """
        # Initialize with DummyNet
        super().__init__(DummyNet(), dim_actions)
        
        # Create SatNavPathFollower for navigation logic
        self.path_follower = SatNavPathFollower(
            goal_radius=goal_radius,
            turn_angle=turn_angle,
            return_action_string=False  # Return action index (int)
        )
        
        # Environment reference (will be set by evaluator)
        self._env = None
        
        # Waypoint tracking (initialized per episode)
        self._waypoints: List = []
        self._current_waypoint_idx: int = 0
        self._current_episode = None
    
    def set_env(self, env):
        """Set environment reference for accessing agent state and episode info.
        
        This method is called by the evaluator to provide the agent with access
        to the environment, which is needed to get the current position and
        extract waypoints from the reference path.
        
        Args:
            env: Environment instance
        """
        self._env = env
        self._waypoints = []
        self._current_waypoint_idx = 0
        self._current_episode = None
    
    def _extract_waypoints(self, episode) -> List:
        """Extract waypoints from episode's reference_path.
        
        Args:
            episode: Current episode with reference_path
        
        Returns:
            List of waypoint positions [lon, lat, alt]
        """
        waypoints = []
        
        if hasattr(episode, 'reference_path') and episode.reference_path:
            for wp in episode.reference_path:
                # Handle both list format [lon, lat, alt] and dict format
                if isinstance(wp, list):
                    waypoints.append(wp)
                elif isinstance(wp, dict) and 'position' in wp:
                    waypoints.append(wp['position'])
        
        # If no waypoints found, use the final goal as single waypoint
        if not waypoints and episode.goals:
            waypoints = [episode.goals[0].position]
        
        return waypoints
    
    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the greedy action that moves toward the goal.
        
        Args:
            observations: Not used (agent uses environment state directly)
            rnn_states: Hidden states (passed through unchanged)
            prev_actions: Not used
            masks: Not used
            deterministic: Not used (greedy agent is always deterministic)
        
        Returns:
            Tuple of (action, rnn_states) where action is a tensor of shape
            (batch_size, 1) containing action indices
        
        Raises:
            RuntimeError: If environment has not been set via set_env()
        """
        if self._env is None:
            raise RuntimeError(
                "Environment not set. Call set_env(env) before using the agent. "
                "This is typically done automatically by the evaluator."
            )
        
        batch_size = rnn_states.size(1)
        device = rnn_states.device
        
        # Track the episode object rather than episode_id: multi-instruction
        # datasets may legitimately contain distinct episodes with the same ID.
        episode = self._env.current_episode
        if episode is not self._current_episode:
            # New episode - extract waypoints
            self._waypoints = self._extract_waypoints(episode)
            self._current_waypoint_idx = 0
            self._current_episode = episode
            self.path_follower.reset()
        
        # Check if we've reached all waypoints
        if self._current_waypoint_idx >= len(self._waypoints):
            # All waypoints reached - stop
            action = torch.full(
                (batch_size, 1),
                0,  # STOP action
                device=device,
                dtype=torch.long
            )
            return action, rnn_states
        
        # Get current position
        agent_state = self._env.agent_state
        current_position = agent_state.position

        # Dense reference paths can contain multiple consecutive waypoints
        # inside one goal-radius neighborhood (including the start point).
        # Advance across all of them before selecting an action; advancing only
        # once can make the nested follower return STOP for the next waypoint
        # and terminate evaluation early.
        while self._current_waypoint_idx < len(self._waypoints):
            current_waypoint = self._waypoints[self._current_waypoint_idx]
            distance_to_waypoint = geodesic_distance(
                current_position,
                current_waypoint,
            )
            if distance_to_waypoint > self.path_follower.goal_radius:
                break
            self._current_waypoint_idx += 1

        # All remaining waypoints are already reached.
        if self._current_waypoint_idx >= len(self._waypoints):
            action = torch.full(
                (batch_size, 1),
                0,  # STOP action
                device=device,
                dtype=torch.long
            )
            return action, rnn_states

        current_waypoint = self._waypoints[self._current_waypoint_idx]
        
        # Use SatNavPathFollower to get next action toward current waypoint
        # path_follower.get_next_action returns action index (0-3)
        action_idx = self.path_follower.get_next_action(
            current_waypoint,
            self._env.simulator,
        )
        
        # Convert to torch tensor with correct shape: (batch_size, 1)
        action = torch.full(
            (batch_size, 1),
            action_idx,
            device=device,
            dtype=torch.long
        )
        
        return action, rnn_states
    
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space,
        action_space,
    ):
        """Create GreedyAgent from configuration.
        
        Args:
            config: Configuration object with MODEL.GREEDY_AGENT section
            observation_space: Observation space (not used)
            action_space: Action space to get number of actions
        
        Returns:
            GreedyAgent instance
        
        Configuration options:
            MODEL.GREEDY_AGENT.goal_radius: Distance threshold in meters (default: 3.0)
            MODEL.GREEDY_AGENT.turn_angle: Turn angle in degrees (default: 15.0)
        """
        # Handle both dict and object action_space
        if isinstance(action_space, dict):
            dim_actions = len(action_space['actions'])
        else:
            dim_actions = action_space.n
        
        # Get configuration parameters
        if hasattr(config.MODEL, 'GREEDY_AGENT'):
            agent_config = config.MODEL.GREEDY_AGENT
            goal_radius = getattr(agent_config, 'goal_radius', 3.0)
            turn_angle = getattr(agent_config, 'turn_angle', 15.0)
        else:
            # Use defaults if no config provided
            goal_radius = 3.0
            turn_angle = 15.0
        
        print(f"GreedyAgent: goal_radius={goal_radius}m, turn_angle={turn_angle}°")
        
        return cls(
            goal_radius=goal_radius,
            turn_angle=turn_angle,
            dim_actions=dim_actions
        )

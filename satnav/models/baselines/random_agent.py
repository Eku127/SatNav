#!/usr/bin/env python3
"""Random Agent for SatNav VLN.

This module implements a random baseline agent that samples actions according
to a specified probability distribution (e.g., from training dataset statistics).

Reference:
    - VLN-CE: vlnce_baselines/nonlearning_agents.py (RandomAgent)
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from satnav.models.base import ILPolicy, Net
from satnav.utils.action_stats import compute_action_distribution


class DummyNet(Net):
    """Minimal Net implementation for non-learning agents.
    
    This network doesn't actually process observations - it just returns
    a fixed-size zero vector to satisfy the ILPolicy interface.
    """
    
    def __init__(self, output_size: int = 512):
        """Initialize dummy network.
        
        Args:
            output_size: Size of output features (arbitrary, not used)
        """
        super().__init__()
        self._output_size = output_size
    
    def forward(self, observations, rnn_states, prev_actions, masks):
        """Forward pass that returns zeros.
        
        Args:
            observations: Not used
            rnn_states: Passed through unchanged
            prev_actions: Not used
            masks: Not used
        
        Returns:
            Tuple of (zero features, unchanged rnn_states)
        """
        batch_size = rnn_states.size(1)
        device = rnn_states.device
        
        # Return zero features
        features = torch.zeros(batch_size, self._output_size, device=device)
        
        return features, rnn_states
    
    @property
    def output_size(self) -> int:
        """Size of network output features."""
        return self._output_size
    
    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent layers (0 for non-recurrent)."""
        return 1  # Need at least 1 for RNN state compatibility
    
    @property
    def is_blind(self) -> bool:
        """Whether the network is blind (doesn't use visual input)."""
        return True


class RandomAgent(ILPolicy):
    """Random baseline agent that samples actions from a distribution.
    
    This agent ignores all observations and simply samples actions according
    to a fixed probability distribution. The distribution can be:
    1. Computed from training dataset statistics (recommended)
    2. Manually specified in the configuration
    
    This provides a minimal baseline for VLN tasks - any learning-based agent
    should significantly outperform a random agent.
    
    Attributes:
        action_probs: Probability distribution over actions
        
    Example:
        >>> # From training data statistics
        >>> config.MODEL.RANDOM_AGENT.stats_dataset = 'data/train.json'
        >>> agent = RandomAgent.from_config(config, obs_space, act_space)
        
        >>> # Or with manual probabilities
        >>> config.MODEL.RANDOM_AGENT.action_probs = [0.02, 0.68, 0.15, 0.15]
        >>> agent = RandomAgent.from_config(config, obs_space, act_space)
    """
    
    def __init__(self, action_probs: list, dim_actions: int, seed: int = None):
        """Initialize random agent.
        
        Args:
            action_probs: List of probabilities for each action.
                Must sum to 1.0. Order: [STOP, FORWARD, LEFT, RIGHT]
            dim_actions: Number of discrete actions (should be 4)
            seed: Random seed for reproducibility (optional)
        """
        # Initialize with DummyNet
        super().__init__(DummyNet(), dim_actions)
        
        # Store action probabilities as tensor
        self.action_probs = torch.tensor(action_probs, dtype=torch.float32)
        
        # Set random seed if provided
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        
        # Validate probabilities
        if len(action_probs) != dim_actions:
            raise ValueError(
                f"Length of action_probs ({len(action_probs)}) must match "
                f"dim_actions ({dim_actions})"
            )
        
        if not torch.isclose(self.action_probs.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError(
                f"action_probs must sum to 1.0, got {self.action_probs.sum():.6f}"
            )
    
    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action by sampling from the distribution.
        
        Args:
            observations: Not used (agent is random)
            rnn_states: Hidden states (passed through unchanged)
            prev_actions: Not used
            masks: Not used
            deterministic: IGNORED for RandomAgent - always samples randomly.
                         A random agent should always be random by definition.
        
        Returns:
            Tuple of (action, rnn_states) where action is a tensor of shape
            (batch_size, 1) containing action indices
        """
        batch_size = rnn_states.size(1)
        device = rnn_states.device
        
        # Move probabilities to correct device
        probs = self.action_probs.to(device)
        
        # Always sample from the distribution (ignore deterministic flag)
        # A random agent should be random by definition
        probs_batch = probs.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_actions)
        action = torch.multinomial(probs_batch, num_samples=1)  # Shape: (batch_size, 1)
        
        return action, rnn_states
    
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space,
        action_space,
    ):
        """Create RandomAgent from configuration.
        
        Args:
            config: Configuration object with MODEL.RANDOM_AGENT section
            observation_space: Observation space (not used)
            action_space: Action space to get number of actions
        
        Returns:
            RandomAgent instance
        
        Configuration options:
            MODEL.RANDOM_AGENT.action_probs: List of probabilities (optional)
            MODEL.RANDOM_AGENT.stats_dataset: Path to dataset for statistics (optional)
            MODEL.RANDOM_AGENT.seed: Random seed for reproducibility (optional)
        
        Note:
            If both action_probs and stats_dataset are provided, action_probs
            takes precedence. If neither is provided, uses VLN-CE defaults.
            Setting a seed ensures reproducible behavior across runs.
        """
        # Handle both dict and object action_space
        if isinstance(action_space, dict):
            dim_actions = len(action_space['actions'])
        else:
            dim_actions = action_space.n
        
        # Get random seed if provided
        seed = None
        if hasattr(config.MODEL, 'RANDOM_AGENT') and hasattr(config.MODEL.RANDOM_AGENT, 'seed'):
            seed = config.MODEL.RANDOM_AGENT.seed
            if seed is not None:
                print(f"RandomAgent: Using random seed: {seed}")
        
        # Try to get action probabilities from config
        if hasattr(config.MODEL, 'RANDOM_AGENT'):
            agent_config = config.MODEL.RANDOM_AGENT
            
            # Option 1: Directly specified probabilities
            if hasattr(agent_config, 'action_probs') and agent_config.action_probs:
                action_probs = agent_config.action_probs
                print(f"RandomAgent: Using configured action probabilities: {action_probs}")
            
            # Option 2: Compute from dataset statistics
            elif hasattr(agent_config, 'stats_dataset') and agent_config.stats_dataset:
                dataset_path = Path(agent_config.stats_dataset)
                
                # Make path absolute if needed
                if not dataset_path.is_absolute():
                    # Assume relative to project root
                    dataset_path = Path.cwd() / dataset_path
                
                print(f"RandomAgent: Computing action distribution from {dataset_path}")
                dist = compute_action_distribution(dataset_path)
                action_probs = dist['probs']
                print(f"  Action distribution: {action_probs}")
                print(f"  Total actions: {dist['total']}")
            
            # Option 3: Use VLN-CE defaults (R2R training set statistics)
            else:
                print("RandomAgent: Using VLN-CE default probabilities (R2R training set)")
                action_probs = [0.02, 0.68, 0.15, 0.15]  # STOP, FORWARD, LEFT, RIGHT
        else:
            # No RANDOM_AGENT config - use defaults
            print("RandomAgent: Using VLN-CE default probabilities (R2R training set)")
            action_probs = [0.02, 0.68, 0.15, 0.15]
        
        return cls(action_probs, dim_actions, seed)


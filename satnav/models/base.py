"""Base classes for VLN models in SatNav.

This module defines the base policy interface for imitation learning models,
following the architecture from VLN-CE and Habitat-Lab.

Reference:
    - VLN-CE: vlnce_baselines/models/policy.py
    - Habitat-Lab: habitat_baselines/rl/ppo/policy.py
"""

import abc
from typing import Any, Tuple

import torch
import torch.nn as nn


class Net(nn.Module, metaclass=abc.ABCMeta):
    """Base network class for VLN models.
    
    This is the base class for neural networks that will be used as the 
    backbone for policies. Follows the interface from Habitat-Lab's 
    habitat_baselines.rl.ppo.policy.Net.
    """
    
    @abc.abstractmethod
    def forward(self, observations, rnn_states, prev_actions, masks):
        """Forward pass of the network.
        
        Args:
            observations: Dict of observations from the environment
            rnn_states: Hidden states of the recurrent network
            prev_actions: Previous actions taken by the agent
            masks: Binary masks indicating episode boundaries
            
        Returns:
            Tuple of (features, rnn_states) where features are the output
            embeddings and rnn_states are the updated hidden states
        """
        pass
    
    @property
    @abc.abstractmethod
    def output_size(self):
        """Size of the network's output features."""
        pass
    
    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        """Number of recurrent layers in the network."""
        pass
    
    @property
    @abc.abstractmethod
    def is_blind(self):
        """Whether the network is blind (no visual input)."""
        pass


class ILPolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base policy class for imitation learning in VLN.
    
    This class defines the interface for imitation learning policies,
    which produce action distributions given observations.
    
    Reference:
        - VLN-CE: vlnce_baselines/models/policy.py (ILPolicy class)
    """
    
    def __init__(self, net: Net, dim_actions: int):
        """Initialize the IL policy.
        
        Args:
            net: The neural network backbone
            dim_actions: Number of discrete actions
        """
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        
        # Action distribution head
        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
    
    def forward(self, *x):
        """Forward pass (must be implemented by subclasses)."""
        raise NotImplementedError
    
    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action given observations.
        
        Args:
            observations: Dict of observations from the environment
            rnn_states: Hidden states of the recurrent network
            prev_actions: Previous actions taken by the agent
            masks: Binary masks indicating episode boundaries
            deterministic: If True, select the mode of the distribution;
                         otherwise sample from it
        
        Returns:
            Tuple of (action, rnn_states) where action is the selected
            action and rnn_states are the updated hidden states
        """
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        
        return action, rnn_states
    
    def get_value(self, *args: Any, **kwargs: Any):
        """Get value estimate (not used in IL, only in RL)."""
        raise NotImplementedError
    
    def evaluate_actions(self, *args: Any, **kwargs: Any):
        """Evaluate actions (not used in IL, only in RL)."""
        raise NotImplementedError
    
    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ):
        """Build action distribution given observations.
        
        Args:
            observations: Dict of observations from the environment
            rnn_states: Hidden states of the recurrent network
            prev_actions: Previous actions taken by the agent
            masks: Binary masks indicating episode boundaries
        
        Returns:
            Action distribution (Categorical)
        """
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        return self.action_distribution(features)
    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        """Create a policy instance from a configuration.
        
        Args:
            config: Configuration object
            observation_space: Observation space definition
            action_space: Action space definition
        
        Returns:
            Policy instance
        """
        pass


class CategoricalNet(nn.Module):
    """Categorical action distribution network.
    
    This network produces logits for a categorical distribution over
    discrete actions.
    
    Reference:
        - Habitat-Lab: habitat_baselines/utils/common.py (CategoricalNet)
    """
    
    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize the categorical network.
        
        Args:
            num_inputs: Size of input features
            num_outputs: Number of discrete actions
        """
        super().__init__()
        
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> "FixedCategorical":
        """Forward pass to produce action distribution.
        
        Args:
            x: Input features
        
        Returns:
            Categorical distribution over actions
        """
        x = self.linear(x)
        return FixedCategorical(logits=x)


class FixedCategorical(torch.distributions.Categorical):
    """Fixed categorical distribution with additional utility methods.
    
    Reference:
        - Habitat-Lab: habitat_baselines/rl/ppo/policy.py
    """
    
    def sample(self, sample_shape=torch.Size()):
        """Sample an action from the distribution."""
        return super().sample(sample_shape).unsqueeze(-1)
    
    def log_probs(self, actions):
        """Compute log probabilities of actions."""
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
    
    def mode(self):
        """Get the most likely action (mode of the distribution)."""
        return self.probs.argmax(dim=-1, keepdim=True)


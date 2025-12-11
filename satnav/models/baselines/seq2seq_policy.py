"""Seq2Seq policy for VLN in SatNav.

This module implements a sequence-to-sequence baseline model for
Vision-and-Language Navigation. The model encodes instructions and
visual observations, then uses an RNN to produce action distributions.

Key difference from VLN-CE:
    - NO depth encoder (SatNav uses satellite overhead imagery)
    - Only RGB visual encoder is used

Reference:
    - VLN-CE: vlnce_baselines/models/seq2seq_policy.py
"""

import torch
import torch.nn as nn
from typing import Tuple

from satnav.models.base import ILPolicy, Net
from satnav.models.encoders.instruction_encoder import InstructionEncoder
from satnav.models.encoders.visual_encoder import TorchVisionResNet50
from satnav.models.encoders.rnn_state_encoder import build_rnn_state_encoder


def _get_num_actions(action_space):
    """Extract number of actions from action_space.
    
    Supports both gym.Space objects (with .n attribute) and
    SatNav dict format (with 'actions' list).
    
    Args:
        action_space: Action space object or dict
    
    Returns:
        Number of discrete actions
    
    Raises:
        ValueError: If action_space format is not recognized
    """
    # Try gym.Space format first (has .n attribute)
    if hasattr(action_space, 'n'):
        return action_space.n
    
    # Try SatNav dict format (has 'actions' list)
    if isinstance(action_space, dict) and 'actions' in action_space:
        return len(action_space['actions'])
    
    # Unknown format
    raise ValueError(
        f"Unsupported action_space format. Expected gym.Space with .n "
        f"or dict with 'actions' key, got {type(action_space)}"
    )


class Seq2SeqPolicy(ILPolicy):
    """Sequence-to-sequence policy for VLN.
    
    This policy uses a Seq2SeqNet as its backbone network.
    
    Args:
        observation_space: Observation space definition
        action_space: Action space definition. Supports:
            - gym.Space with .n attribute (standard format)
            - dict with 'actions' list (SatNav format)
        model_config: Configuration for the model
    
    Reference:
        VLN-CE: vlnce_baselines/models/seq2seq_policy.py (lines 20-49)
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
    ):
        """Initialize the Seq2Seq policy.
        
        Args:
            observation_space: Observation space
            action_space: Discrete action space (gym.Space or dict with 'actions')
            model_config: Model configuration
        """
        num_actions = _get_num_actions(action_space)
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=num_actions,
            ),
            num_actions,
        )
    
    @classmethod
    def from_config(
        cls, config, observation_space, action_space
    ):
        """Create a policy from configuration.
        
        Args:
            config: Full configuration object
            observation_space: Observation space
            action_space: Action space
        
        Returns:
            Seq2SeqPolicy instance
        """
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class Seq2SeqNet(Net):
    """Seq2Seq network for VLN.
    
    Architecture:
        1. Instruction Encoder (LSTM) -> instruction_embedding
        2. RGB Visual Encoder (ResNet50) -> rgb_embedding
        3. [Optional] Previous Action Embedding -> prev_action_embedding
        4. Concatenate embeddings
        5. RNN State Encoder (GRU) -> hidden_state
        6. Action head (in policy) -> action logits
    
    Key difference from VLN-CE:
        - NO depth encoder (SatNav uses satellite overhead imagery without depth)
        - Only RGB encoder is used for visual input
    
    Args:
        observation_space: Observation space definition
        model_config: Model configuration with:
            - INSTRUCTION_ENCODER: Config for instruction encoder
            - RGB_ENCODER: Config for RGB visual encoder
            - SEQ2SEQ: Config for Seq2Seq model (hidden_size, rnn_type, use_prev_action)
            - normalize_rgb: Whether to normalize RGB inputs
    
    Reference:
        VLN-CE: vlnce_baselines/models/seq2seq_policy.py
                (Seq2SeqNet class, lines 52-180)
    """
    
    def __init__(
        self, observation_space, model_config, num_actions: int
    ):
        """Initialize the Seq2Seq network.
        
        Args:
            observation_space: Observation space
            model_config: Model configuration
            num_actions: Number of discrete actions
        """
        super().__init__()
        self.model_config = model_config
        
        # Initialize instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )
        
        # Initialize RGB visual encoder
        # Note: VLN-CE supports both ResNet18 and ResNet50
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ], f"Unsupported RGB encoder: {model_config.RGB_ENCODER.cnn_type}"
        
        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            from satnav.models.encoders.visual_encoder import TorchVisionResNet50
            rgb_encoder_class = TorchVisionResNet50
        else:
            from satnav.models.encoders.visual_encoder import TorchVisionResNet18
            rgb_encoder_class = TorchVisionResNet18
        
        self.rgb_encoder = rgb_encoder_class(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=False,
        )
        
        # Initialize previous action embedding (optional)
        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        
        # Calculate RNN input size
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.RGB_ENCODER.output_size
        )
        
        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim
        
        # Initialize RNN state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.SEQ2SEQ.hidden_size,
            rnn_type=model_config.SEQ2SEQ.rnn_type,
            num_layers=1,
        )
        
        self.train()
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create initial hidden states for the Seq2Seq model.
        
        Args:
            batch_size: Number of parallel sequences
            device: Device to create tensors on
            
        Returns:
            Initial hidden states with shape (num_recurrent_layers, batch_size, hidden_size)
        """
        return torch.zeros(
            self.state_encoder.num_recurrent_layers,
            batch_size,
            self.model_config.SEQ2SEQ.hidden_size,
            device=device
        )
    
    @property
    def output_size(self):
        """Output size of the network (SEQ2SEQ.hidden_size)."""
        return self.model_config.SEQ2SEQ.hidden_size
    
    @property
    def is_blind(self):
        """Whether the network is blind (no visual input)."""
        return self.rgb_encoder.is_blind
    
    def forward(
        self, observations, rnn_states, prev_actions, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            observations: Dictionary containing:
                - "instruction": Tokenized instruction [batch, seq_len]
                - "rgb": RGB image [batch, height, width, 3]
            rnn_states: Hidden states [num_layers, batch, hidden_size]
            prev_actions: Previous actions [batch, 1]
            masks: Episode boundary masks [batch, 1]
        
        Returns:
            Tuple of (features, rnn_states):
                - features: Output features [batch, hidden_size]
                - rnn_states: Updated hidden states [num_layers, batch, hidden_size]
        
        Reference:
            VLN-CE: vlnce_baselines/models/seq2seq_policy.py
                    (Seq2SeqNet.forward, lines 142-180)
        """
        # Encode instruction
        instruction_embedding = self.instruction_encoder(observations)
        
        # Encode RGB visual observation
        rgb_embedding = self.rgb_encoder(observations)
        
        # Apply ablations if specified (for ablation studies)
        if hasattr(self.model_config, "ablate_instruction") and self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if hasattr(self.model_config, "ablate_rgb") and self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0
        
        # Concatenate all embeddings
        x = torch.cat([instruction_embedding, rgb_embedding], dim=1)
        
        # Add previous action embedding if enabled
        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)
        
        # Pass through RNN state encoder
        x, rnn_states_out = self.state_encoder(x, rnn_states, masks)
        
        return x, rnn_states_out


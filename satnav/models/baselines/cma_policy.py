"""Cross-Modal Attention (CMA) policy for VLN.

This module implements the CMA model from "Improving Vision-and-Language
Navigation with Image-Text Pairs from the Web" (Zhu et al., 2020).

Architecture:
    1. Instruction Encoder (LSTM bidirectional) -> instruction tokens
    2. RGB Visual Encoder (ResNet50 spatial) -> RGB spatial features
    3. Previous Action Embedding
    4. First RNN State Encoder (GRU)
    5. Cross-Modal Attention:
        - Text-State Attention: state queries instruction
        - Text-RGB Attention: text embedding queries RGB spatial features
    6. Second RNN State Encoder (GRU)
    7. Action Distribution Head (in ILPolicy)

Key differences from VLN-CE CMA:
    - NO depth encoder (SatNav uses satellite imagery without depth)
    - NO progress monitor (simplified implementation)
    - Only RGB-instruction cross-modal attention

Reference:
    - Paper: https://arxiv.org/abs/2004.02857
    - VLN-CE: vlnce_baselines/models/cma_policy.py
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from satnav.models.base import ILPolicy, Net
from satnav.models.encoders.instruction_encoder import InstructionEncoder
from satnav.models.encoders.visual_encoder import TorchVisionResNet50
from satnav.models.encoders.rnn_state_encoder import build_rnn_state_encoder


class CMAPolicy(ILPolicy):
    """Cross-Modal Attention policy for imitation learning.
    
    This policy uses CMANet as its backbone network.
    
    Args:
        observation_space: Observation space definition
        action_space: Action space definition. Supports:
            - gym.Space with .n attribute (standard format)
            - dict with 'actions' list (SatNav format)
        model_config: Configuration for the model
    
    Reference:
        VLN-CE: vlnce_baselines/models/cma_policy.py (CMAPolicy class, lines 24-49)
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
    ):
        """Initialize the CMA policy.
        
        Args:
            observation_space: Observation space
            action_space: Discrete action space (gym.Space or dict with 'actions')
            model_config: Model configuration
        """
        # Get number of actions
        if hasattr(action_space, 'n'):
            num_actions = action_space.n
        elif isinstance(action_space, dict) and 'actions' in action_space:
            num_actions = len(action_space['actions'])
        else:
            raise ValueError(
                f"action_space must have .n attribute (gym.Space) "
                f"or dict with 'actions' key, got {type(action_space)}"
            )
        
        super().__init__(
            CMANet(
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
            CMAPolicy instance
        """
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class CMANet(Net):
    """Cross-Modal Attention network for VLN.
    
    An implementation of the cross-modal attention (CMA) network from
    "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web"
    https://arxiv.org/abs/2004.02857
    
    Simplified version for SatNav (satellite imagery):
    - NO depth encoder (only RGB)
    - NO progress monitor
    - Core CMA mechanism preserved
    
    Architecture:
        1. Instruction Encoder -> instruction_embedding [B, hidden, seq_len]
        2. RGB Encoder (spatial) -> rgb_embedding [B, C, 4, 4]
        3. Previous Action Embedding -> prev_action [B, 32]
        4. First State Encoder:
           Input: [rgb_pooled, prev_action] -> state [B, hidden]
        5. Cross-Modal Attention:
           a) Text-State Attention: state queries instruction -> text_embedding
           b) Text-RGB Attention: text queries RGB spatial -> rgb_attended
        6. Second State Encoder:
           Input: [state, text_embedding, rgb_attended, prev_action] -> output
    
    Args:
        observation_space: Observation space definition
        model_config: Model configuration
        num_actions: Number of discrete actions
    
    Reference:
        VLN-CE: vlnce_baselines/models/cma_policy.py (CMANet class, lines 52-309)
    """
    
    def __init__(
        self,
        observation_space,
        model_config,
        num_actions: int,
    ):
        """Initialize CMA network.
        
        Args:
            observation_space: Observation space definition
            model_config: Model configuration
            num_actions: Number of discrete actions
        """
        super().__init__()
        self.model_config = model_config
        
        # Ensure instruction encoder returns all timesteps
        # (CMA needs full sequence for attention, not just final state)
        if hasattr(model_config, 'INSTRUCTION_ENCODER'):
            # Make a copy to avoid modifying original config
            inst_config = model_config.INSTRUCTION_ENCODER
            if hasattr(inst_config, 'final_state_only'):
                # If config is mutable, set it
                try:
                    inst_config.final_state_only = False
                except:
                    # If config is frozen, we'll need to work around it
                    pass
        
        # Initialize the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )
        
        # Initialize the RGB visual encoder with spatial output
        self.rgb_encoder = TorchVisionResNet50(
            output_size=model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.get('normalize_rgb', False),
            trainable=model_config.RGB_ENCODER.get('trainable', False),
            spatial_output=True,  # CMA requires spatial features
        )
        
        # Previous action embedding
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        
        hidden_size = model_config.CMA.hidden_size
        self._hidden_size = hidden_size
        
        # RGB linear layer (pool spatial features to vector)
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],  # C dimension
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        
        # First RNN state encoder
        rnn_input_size = (
            model_config.RGB_ENCODER.output_size
            + self.prev_action_embedding.embedding_dim
        )
        
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.CMA.hidden_size,
            rnn_type=model_config.CMA.rnn_type,
            num_layers=1,
        )
        
        # Output size calculation (for action distribution)
        self._output_size = model_config.CMA.hidden_size
        
        # Cross-modal attention components
        # 1. RGB key-value projection
        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],  # input channels
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,  # output channels
            1,  # kernel size
        )
        
        # 2. Text-state attention (state queries instruction)
        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size,
            hidden_size // 2,
            1,
        )
        
        # 3. Text-RGB attention (text queries RGB spatial features)
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )
        
        # Attention scale factor
        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )
        
        # Second state compression and encoder
        self._second_state_input_size = (
            self._hidden_size
            + self.instruction_encoder.output_size
            + model_config.RGB_ENCODER.output_size
            + self.prev_action_embedding.embedding_dim
        )
        
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._second_state_input_size,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )
        
        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.CMA.rnn_type,
            num_layers=1,
        )
        
        # Final output size is from second state encoder
        self._output_size = model_config.CMA.hidden_size
        
        self.train()
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create initial hidden states for the CMA model.
        
        CMA has two RNN encoders, so we create states for both.
        
        Args:
            batch_size: Number of parallel sequences
            device: Device to create tensors on
            
        Returns:
            Initial hidden states with shape (num_total_layers, batch_size, hidden_size)
            where num_total_layers = 2 (one layer for each encoder)
        """
        num_total_layers = (
            self.state_encoder.num_recurrent_layers
            + self.second_state_encoder.num_recurrent_layers
        )
        return torch.zeros(
            num_total_layers,
            batch_size,
            self._hidden_size,
            device=device
        )
    
    @property
    def output_size(self) -> int:
        """Size of the network's output features."""
        return self._output_size
    
    @property
    def is_blind(self) -> bool:
        """Whether the network is blind (no visual input)."""
        return self.rgb_encoder.is_blind
    
    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [batch, hidden_dim]
            k: Key tensor [batch, hidden_dim, seq_len]
            v: Value tensor [batch, hidden_dim, seq_len]
            mask: Optional mask [batch, seq_len] (True = masked out)
        
        Returns:
            Attended values [batch, hidden_dim]
        
        Reference:
            VLN-CE: vlnce_baselines/models/cma_policy.py (_attn method, lines 207-217)
        """
        # Compute attention logits: Q * K^T
        logits = torch.einsum("nc, nci -> ni", q, k)
        
        # Apply mask (set masked positions to large negative value)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        
        # Compute attention weights with scaling
        attn = F.softmax(logits * self._scale, dim=1)
        
        # Apply attention to values
        return torch.einsum("ni, nci -> nc", attn, v)
    
    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through CMA network.
        
        Args:
            observations: Dict of observations
                - "instruction": [T*N] or [T*N, max_len] instruction tokens
                - "rgb": [T*N, H, W, 3] RGB images
            rnn_states: [num_layers, N, hidden_size] RNN hidden states
            prev_actions: [T*N, 1] previous actions
            masks: [T*N, 1] episode boundary masks
        
        Returns:
            Tuple of (features, rnn_states_out)
                features: [T*N, output_size] output features
                rnn_states_out: [num_layers, N, hidden_size] updated states
        
        Reference:
            VLN-CE: vlnce_baselines/models/cma_policy.py (forward method, lines 219-309)
        """
        # 1. Encode instruction (returns all timesteps for attention)
        instruction_embedding = self.instruction_encoder(observations)
        
        # 2. Encode RGB (returns spatial features)
        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)  # [B, C, H*W]
        
        # 3. Encode previous action
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        
        # Optional ablation studies
        if self.model_config.get('ablate_instruction', False):
            instruction_embedding = instruction_embedding * 0
        if self.model_config.get('ablate_rgb', False):
            rgb_embedding = rgb_embedding * 0
        
        # 4. First state encoder
        # Pool RGB spatial features to vector
        rgb_in = self.rgb_linear(rgb_embedding)
        
        # Concatenate RGB and prev_action
        state_in = torch.cat([rgb_in, prev_actions], dim=1)
        
        # Clone RNN states for output
        rnn_states_out = rnn_states.detach().clone()
        
        # Forward through first state encoder
        # Split RNN states for the two encoders
        first_encoder_layers = self.state_encoder.num_recurrent_layers
        (
            state,
            rnn_states_out[0 : first_encoder_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[0 : first_encoder_layers],
            masks,
        )
        
        # 5. Cross-Modal Attention
        # a) Text-State Attention: state queries instruction
        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )
        
        # b) Text-RGB Attention: text queries RGB spatial features
        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        
        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        
        # 6. Second state encoder
        # Concatenate all features
        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                prev_actions,
            ],
            dim=1,
        )
        
        # Compress and forward through second state encoder
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[first_encoder_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[first_encoder_layers :],
            masks,
        )
        
        return x, rnn_states_out


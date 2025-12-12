"""RNN state encoder for VLN models.

This module implements recurrent state encoders (GRU and LSTM) for
processing sequential observations in VLN tasks.

Directly ported from Habitat-Lab with minimal modifications.

Reference:
    - Habitat-Lab: habitat_baselines/rl/models/rnn_state_encoder.py
"""

import torch
import torch.nn as nn
from typing import Tuple


class RNNStateEncoder(nn.Module):
    """Base RNN encoder for use with RL and IL.
    
    The main functionality this provides over just using PyTorch's RNN
    interface directly is that it takes an additional masks input that
    resets the hidden state between two adjacent timesteps to handle
    episodes ending in the middle of a rollout.
    
    Reference:
        Habitat-Lab: habitat_baselines/rl/models/rnn_state_encoder.py
                     (RNNStateEncoder class, lines 261-348)
    """
    
    def layer_init(self):
        """Initialize RNN weights with orthogonal initialization."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
    
    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pack hidden states (identity for GRU, concat for LSTM)."""
        return hidden_states
    
    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Unpack hidden states (identity for GRU, split for LSTM)."""
        return hidden_states
    
    def single_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for a non-sequence input (single timestep).
        
        Following Habitat-Lab's approach.
        
        Args:
            x: Input features [batch_size, input_size]
            hidden_states: Hidden states [num_layers, batch_size, hidden_size]
            masks: Episode boundary masks [batch_size, 1]
        
        Returns:
            Tuple of (output, hidden_states)
                output: [batch_size, hidden_size]
                hidden_states: [num_layers, batch_size, hidden_size]
        """
        # Reset hidden state where mask is 0 (episode boundary)
        # hidden_states: [num_layers, batch, hidden], masks: [batch, 1]
        # Reshape masks to [1, batch, 1] for broadcasting
        hidden_states = torch.where(
            masks.view(1, -1, 1).bool(),
            hidden_states,
            hidden_states.new_zeros(())
        )
        
        # Add sequence dimension and forward through RNN
        # hidden_states is already [num_layers, batch, hidden]
        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)
        
        # Remove sequence dimension
        x = x.squeeze(0)
        return x, hidden_states
    
    def forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the RNN.
        
        Automatically handles both single-step and sequence inputs.
        Following Habitat-Lab's simpler approach without unnecessary permutes.
        
        Args:
            x: Input features [batch_size, input_size] or [T*batch_size, input_size]
            hidden_states: Hidden states [num_layers, batch_size, hidden_size]
            masks: Episode boundary masks [batch_size, 1] or [T*batch_size, 1]
        
        Returns:
            Tuple of (output, hidden_states) where hidden_states is [num_layers, batch, hidden]
        """
        # Get batch_size from hidden_states (which is [num_layers, batch, hidden])
        batch_size = hidden_states.size(1)

        # Single-step forward (common case for IL/RL)
        if x.dim() == 2 and masks.dim() == 2 and x.size(0) == batch_size:
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            # Sequence forward: reshape to [T, batch, feat]
            assert (
                x.size(0) % batch_size == 0
            ), f"Sequence length {x.size(0)} must be a multiple of batch size {batch_size}"
            T = x.size(0) // batch_size
            x = x.view(T, batch_size, -1)
            masks = masks.view(T, batch_size, 1)

            outputs = []
            for t in range(T):
                # Reset hidden where mask is 0 (episode boundary)
                # masks[t] is [batch, 1], reshape to [1, batch, 1] for broadcasting
                hidden_states = torch.where(
                    masks[t].view(1, batch_size, 1).bool(),
                    hidden_states,
                    hidden_states.new_zeros(())
                )

                # Forward through RNN
                # hidden_states is already [num_layers, batch, hidden]
                out, hidden_states = self.rnn(
                    x[t].unsqueeze(0), self.unpack_hidden(hidden_states)
                )
                hidden_states = self.pack_hidden(hidden_states)
                outputs.append(out)

            # Collapse time dimension back to [T*batch, hidden]
            x = torch.cat(outputs, dim=0).view(T * batch_size, -1)
        
        return x, hidden_states


class LSTMStateEncoder(RNNStateEncoder):
    """LSTM-based state encoder.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        num_layers (int): Number of LSTM layers (default: 1)
    
    Reference:
        Habitat-Lab: habitat_baselines/rl/models/rnn_state_encoder.py
                     (LSTMStateEncoder class, lines 350-379)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        """Initialize LSTM state encoder.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Number of recurrent layers is doubled for LSTM (hidden + cell state)
        self.num_recurrent_layers = num_layers * 2
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.layer_init()
    
    def pack_hidden(
        self, hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Pack LSTM hidden and cell states into single tensor.
        
        Args:
            hidden_states: Tuple of (hidden, cell) states
        
        Returns:
            Concatenated tensor [2*num_layers, batch_size, hidden_size]
        """
        return torch.cat(hidden_states, 0)
    
    def unpack_hidden(
        self, hidden_states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack LSTM hidden and cell states from single tensor.
        
        Args:
            hidden_states: Concatenated tensor [2*num_layers, batch_size, hidden_size]
        
        Returns:
            Tuple of (hidden, cell) states
        """
        lstm_states = torch.chunk(hidden_states, 2, 0)
        return (lstm_states[0], lstm_states[1])


class GRUStateEncoder(RNNStateEncoder):
    """GRU-based state encoder.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        num_layers (int): Number of GRU layers (default: 1)
    
    Reference:
        Habitat-Lab: habitat_baselines/rl/models/rnn_state_encoder.py
                     (GRUStateEncoder class, lines 381-399)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        """Initialize GRU state encoder.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
        """
        super().__init__()
        
        self.num_recurrent_layers = num_layers
        
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.layer_init()


def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "GRU",
    num_layers: int = 1,
):
    """Factory function for creating RNN state encoders.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        rnn_type: Type of RNN cell - "GRU" or "LSTM" (case-insensitive)
        num_layers: Number of RNN layers
    
    Returns:
        RNNStateEncoder instance (GRU or LSTM)
    
    Raises:
        RuntimeError: If rnn_type is not "GRU" or "LSTM"
    
    Reference:
        Habitat-Lab: habitat_baselines/rl/models/rnn_state_encoder.py
                     (build_rnn_state_encoder function, lines 401-422)
        
        VLN-CE default config (vlnce_baselines/config/default.py lines 255-257):
            hidden_size: 512
            rnn_type: GRU
            num_layers: 1 (implicit)
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(input_size, hidden_size, num_layers)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(input_size, hidden_size, num_layers)
    else:
        raise RuntimeError(
            f"Did not recognize rnn type '{rnn_type}'. "
            f"Must be either 'GRU' or 'LSTM'."
        )


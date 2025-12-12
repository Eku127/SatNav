"""Encoder modules for VLN models.

This package contains reusable encoder modules:
    - instruction_encoder.py: Encodes natural language instructions
    - visual_encoder.py: Encodes RGB visual observations
    - rnn_state_encoder.py: Recurrent state encoder

Reference:
    - VLN-CE: vlnce_baselines/models/encoders/
"""

from satnav.models.encoders.instruction_encoder import InstructionEncoder
from satnav.models.encoders.visual_encoder import (
    TorchVisionResNet50,
    TorchVisionResNet18,
)
from satnav.models.encoders.rnn_state_encoder import (
    build_rnn_state_encoder,
    GRUStateEncoder,
    LSTMStateEncoder,
    RNNStateEncoder,
)

__all__ = [
    "InstructionEncoder",
    "TorchVisionResNet50",
    "TorchVisionResNet18",
    "build_rnn_state_encoder",
    "GRUStateEncoder",
    "LSTMStateEncoder",
    "RNNStateEncoder",
]


"""Instruction encoder for VLN models.

This module implements an RNN-based encoder for natural language instructions,
supporting both pretrained GloVe embeddings and random initialization.

Reference:
    - VLN-CE: vlnce_baselines/models/encoders/instruction_encoder.py
    - Parameters match VLN-CE defaults (embedding_size=50, hidden_size=128)
"""

import gzip
import json
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class InstructionEncoder(nn.Module):
    """Encodes natural language instructions using an RNN.
    
    This encoder processes tokenized instruction sequences and returns
    either the final hidden state or the full sequence of hidden states.
    
    Supports two modes:
    1. Pretrained embeddings (GloVe): Load from compressed JSON file
    2. Random initialization: Standard Gaussian initialization
    
    The embedding format for pretrained mode:
        - Index 0: PAD token (all zeros)
        - Index 1: UNK token (mean of all word embeddings)
        - Index 2+: Vocabulary words
    
    Args:
        config: Configuration object with the following attributes:
            - embedding_size (int): Dimension of embedding vectors (default: 50)
            - hidden_size (int): Hidden size of the RNN (default: 128)
            - rnn_type (str): Type of RNN cell - "GRU" or "LSTM" (default: "LSTM")
            - bidirectional (bool): Whether to use bidirectional RNN (default: False)
            - final_state_only (bool): If True, return only final state (default: True)
            - sensor_uuid (str): Name of the sensor (default: "instruction")
            - use_pretrained_embeddings (bool): Whether to use GloVe (default: True)
            - embedding_file (str): Path to embeddings.json.gz file
            - fine_tune_embeddings (bool): Whether to fine-tune embeddings (default: False)
            - vocab_size (int): Size of vocabulary (for random init mode)
    
    Reference:
        VLN-CE default config (vlnce_baselines/config/default.py lines 222-237):
            embedding_size: 50
            hidden_size: 128
            rnn_type: LSTM
            bidirectional: False
            final_state_only: True
    """
    
    def __init__(self, config):
        """Initialize the instruction encoder.
        
        Args:
            config: Configuration object with encoder parameters
        """
        super().__init__()
        
        self.config = config
        
        # Select RNN type
        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )
        
        # Initialize embedding layer
        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                # Load pretrained GloVe embeddings
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:
                # Random initialization with standard Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )
    
    @property
    def output_size(self) -> int:
        """Get the output size of the encoder.
        
        Returns:
            hidden_size * 2 if bidirectional, hidden_size otherwise
        """
        return self.config.hidden_size * (1 + int(self.config.bidirectional))
    
    def _load_embeddings(self) -> Tensor:
        """Load pretrained word embeddings from a compressed JSON file.
        
        The embeddings file should be in gzipped JSON format, containing
        a 2D array of shape [vocab_size, embedding_dim].
        
        Embedding format (VLN-CE compatible):
            - Index 0: PAD token - [0.0, 0.0, ..., 0.0]
            - Index 1: UNK token - [mean_0, mean_1, ..., mean_n]
              (mean of all R2R word embeddings)
            - Index 2+: Word embeddings for vocabulary
        
        Why UNK is averaged: https://bit.ly/3u3hkYg
        
        Returns:
            Tensor of shape [num_words, embedding_dim]
        
        Reference:
            VLN-CE: vlnce_baselines/models/encoders/instruction_encoder.py (lines 51-61)
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings
    
    def forward(self, observations: Dict) -> Tensor:
        """Encode instruction observations.
        
        Args:
            observations: Dictionary containing:
                - "instruction": Tensor of shape [batch_size, seq_length]
                  containing tokenized instruction indices
        
        Returns:
            Encoded instruction features:
                - If final_state_only=True: [batch_size, hidden_size]
                - If final_state_only=False: [batch_size, hidden_size, seq_length]
        
        Tensor shapes during computation:
            instruction: [batch_size, seq_length]
            lengths: [batch_size]
            embedded: [batch_size, seq_length, embedding_size]
            output: [batch_size, seq_length, hidden_size * num_directions]
            final_state: [num_directions, batch_size, hidden_size]
        
        Reference:
            VLN-CE: vlnce_baselines/models/encoders/instruction_encoder.py (lines 63-94)
        """
        # Extract instruction indices from observations
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            # Embed tokens: [batch, seq_len] → [batch, seq_len, embedding_size]
            instruction = self.embedding_layer(instruction)
        else:
            # For RxR multilingual instructions (future support)
            instruction = observations["rxr_instruction"]
        
        # Compute lengths by detecting non-PAD positions via all-zero embedding rows
        # PAD token (index 0) maps to all-zeros vector, so sum(dim=2)==0 flags padding
        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).clamp_min(1).cpu()
        
        # Pack sequence for efficient RNN processing
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Forward pass through RNN
        output, final_state = self.encoder_rnn(packed_seq)
        
        # Extract hidden state for LSTM (ignore cell state)
        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]
        
        # Return final state or full sequence
        if self.config.final_state_only:
            # final_state: [num_directions, batch, hidden_size]
            # For unidirectional: squeeze to [batch, hidden_size]
            # For bidirectional: reshape to [batch, hidden_size * 2]
            if self.config.bidirectional:
                return final_state.permute(1, 0, 2).contiguous().view(
                    final_state.size(1), -1
                )
            return final_state.squeeze(0)
        else:
            # Unpack sequence and permute to [batch, hidden, seq_len]
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output.permute(0, 2, 1)

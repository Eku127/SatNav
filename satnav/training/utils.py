#!/usr/bin/env python3
"""Training utility functions for SatNav VLN tasks.

This module provides utility functions for batch processing, padding,
and data collation used during training.

Reference: VLN-CE vlnce_baselines/dagger_trainer.py
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch


class ObservationsDict(dict):
    """Dictionary wrapper for observations that supports pin_memory."""
    
    def pin_memory(self):
        """Pin memory for all tensors in the dictionary."""
        for k in self.keys():
            self[k] = self[k].pin_memory()
        return self


def pad_helper(t: torch.Tensor, max_len: int, fill_val: float = 0) -> torch.Tensor:
    """Pad or truncate a tensor to the specified length.
    
    Args:
        t: Tensor to pad, shape (T, ...)
        max_len: Target length
        fill_val: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len, ...)
    """
    if t.size(0) >= max_len:
        # Truncate if too long
        return t[:max_len]
    
    # Pad if too short
    pad_amount = max_len - t.size(0)
    pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
    return torch.cat([t, pad], dim=0)


def collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], ...]]
) -> Tuple[ObservationsDict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for training data.
    
    Takes a batch of variable-length episodes and pads them to the same length,
    then formats them for RNN input (T*N format).
    
    Args:
        batch: List of tuples, each containing:
            - observations: Dict of sensor observations (each is a tensor of shape (T, ...))
            - prev_actions: Tensor of previous actions, shape (T,)
            - teacher_actions: Tensor of teacher actions, shape (T,)
            
    Returns:
        Tuple of:
        - observations_batch: Dict[str, Tensor] with each sensor having shape (T*N, ...)
        - prev_actions_batch: Tensor of shape (T*N, 1)
        - not_done_masks: Tensor of shape (T*N, 1) - binary mask where 0 indicates episode start
        - teacher_actions_batch: Tensor of shape (T, N)
        
    Note:
        T = trajectory length (max length in batch)
        N = batch size
    """
    # Unpack batch
    transposed = list(zip(*batch))
    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    teacher_actions_batch = list(transposed[2])
    weights_batch: Optional[List[torch.Tensor]] = None
    if len(transposed) > 3:
        weights_batch = list(transposed[3])
    
    B = len(prev_actions_batch)  # Batch size
    
    # Find maximum trajectory length in the batch
    max_traj_len = max(len(actions) for actions in teacher_actions_batch)
    
    # Pad observations
    new_observations_batch = defaultdict(list)
    for bid in range(B):
        for sensor in observations_batch[bid]:
            # Pad each sensor's observations
            # Use fill_val=1.0 for images (white padding)
            # Use fill_val=0 for other sensors
            if sensor == 'rgb':
                fill_val = 255.0  # White padding for images
            else:
                fill_val = 0.0
            
            padded = pad_helper(
                observations_batch[bid][sensor],
                max_traj_len,
                fill_val=fill_val
            )
            new_observations_batch[sensor].append(padded)
        
        # Pad actions
        prev_actions_batch[bid] = pad_helper(
            prev_actions_batch[bid], max_traj_len, fill_val=0
        )
        teacher_actions_batch[bid] = pad_helper(
            teacher_actions_batch[bid], max_traj_len, fill_val=0
        )
        if weights_batch is not None:
            # Padded timesteps must not contribute to the training loss.
            weights_batch[bid] = pad_helper(
                weights_batch[bid], max_traj_len, fill_val=0.0
            )
    
    # Stack observations into (T, N, ...) format
    for sensor in new_observations_batch:
        new_observations_batch[sensor] = torch.stack(
            new_observations_batch[sensor], dim=1  # Stack along batch dimension
        )
        # Reshape to (T*N, ...) for RNN processing
        new_observations_batch[sensor] = new_observations_batch[sensor].view(
            -1, *new_observations_batch[sensor].size()[2:]
        )
    
    # Stack actions into (T, N) format
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)  # (T, N)
    teacher_actions_batch = torch.stack(teacher_actions_batch, dim=1)  # (T, N)
    
    # Create not_done_masks
    # Mask is 1 for all steps except the first step of each episode (which is 0)
    not_done_masks = torch.ones_like(teacher_actions_batch, dtype=torch.uint8)
    not_done_masks[0, :] = 0  # First step is episode start for all episodes in batch
    
    # Convert to ObservationsDict for pin_memory support
    observations_batch_dict = ObservationsDict(new_observations_batch)
    
    output = (
        observations_batch_dict,
        prev_actions_batch.view(-1, 1),  # (T*N, 1)
        not_done_masks.view(-1, 1),      # (T*N, 1)
        teacher_actions_batch,            # (T, N)
    )

    if weights_batch is None:
        return output

    weights_batch_tensor = torch.stack(weights_batch, dim=1)  # (T, N)
    return output + (weights_batch_tensor,)


def tokenize_instruction(
    instruction_text: str,
    vocab: Any,
    max_length: int = 200
) -> torch.Tensor:
    """Tokenize an instruction text into indices.
    
    Args:
        instruction_text: Natural language instruction
        vocab: VocabDict instance
        max_length: Maximum sequence length (pad or truncate)
        
    Returns:
        Tensor of token indices, shape (max_length,)
    """
    # Tokenize text
    from satnav.utils.build_vocab import tokenize
    
    tokens = tokenize(instruction_text)
    indices = vocab.tokens_to_indices(tokens)
    
    # Convert to tensor
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    
    # Pad or truncate to max_length
    if len(indices_tensor) > max_length:
        indices_tensor = indices_tensor[:max_length]
    elif len(indices_tensor) < max_length:
        # Pad with 0 (PAD token)
        padding = torch.zeros(max_length - len(indices_tensor), dtype=torch.long)
        indices_tensor = torch.cat([indices_tensor, padding])
    
    return indices_tensor

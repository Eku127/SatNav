"""Utility functions and classes for models.

This module contains helper functions for model initialization,
checkpoint management, and other utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def init_weights_normal(module: nn.Module, std: float = 0.01):
    """Initialize module weights with normal distribution.
    
    Args:
        module: The module to initialize
        std: Standard deviation for the normal distribution
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0):
    """Initialize module weights with orthogonal initialization.
    
    Args:
        module: The module to initialize
        gain: Gain for the orthogonal initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    extra_state: Optional[Dict] = None,
):
    """Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer (optional)
        epoch: Current training epoch
        path: Path to save the checkpoint
        extra_state: Additional state to save (optional)
    """
    state_dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra_state is not None:
        state_dict.update(extra_state)
    
    torch.save(state_dict, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """Load a model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        model: The model to load state into (optional)
        optimizer: The optimizer to load state into (optional)
        device: Device to load the checkpoint on
    
    Returns:
        Dictionary containing the loaded checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


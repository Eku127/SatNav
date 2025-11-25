#!/usr/bin/env python3
"""Configuration loading system for SatNav."""

import os
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    configs_dir: Optional[Union[str, Path]] = None,
) -> DictConfig:
    """Load YAML configuration file using OmegaConf.
    
    This function loads a YAML configuration file and returns a DictConfig object
    that can be used to access configuration values. The function supports both
    absolute and relative paths.
    
    Args:
        config_path: Path to the YAML configuration file. Can be absolute or
            relative to the current working directory or configs_dir.
        configs_dir: Optional directory to search for config files if config_path
            is relative. If None, searches from current working directory.
            Defaults to None.
    
    Returns:
        DictConfig: OmegaConf configuration object containing the loaded config.
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        OmegaConfException: If the YAML file is invalid or cannot be parsed.
    
    Example:
        >>> config = load_config("configs/vln_task.yaml")
        >>> print(config.ENVIRONMENT.MAX_EPISODE_STEPS)
        500
    """
    config_path = Path(config_path)
    
    # Check if absolute path exists
    if config_path.is_absolute() and config_path.exists():
        return OmegaConf.load(config_path)
    
    # Check if relative path exists from current directory
    if config_path.exists():
        return OmegaConf.load(config_path)
    
    # Try to find in configs_dir if provided
    if configs_dir is not None:
        configs_dir = Path(configs_dir)
        proposed_path = configs_dir / config_path
        if proposed_path.exists():
            return OmegaConf.load(proposed_path)
    
    # Try to find in default configs directory (project root/configs)
    project_root = Path(__file__).parent.parent.parent
    default_configs_dir = project_root / "configs"
    if default_configs_dir.exists():
        proposed_path = default_configs_dir / config_path
        if proposed_path.exists():
            return OmegaConf.load(proposed_path)
    
    # If all attempts fail, raise error
    raise FileNotFoundError(
        f"Configuration file not found: {config_path}\n"
        f"Searched in:\n"
        f"  - Current directory: {Path.cwd()}\n"
        f"  - Configs directory: {configs_dir if configs_dir else 'not specified'}\n"
        f"  - Default configs: {default_configs_dir}"
    )


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: DictConfig object to save.
        output_path: Path where to save the configuration file.
    """
    output_path = Path(output_path)
    OmegaConf.save(config, output_path)


"""Model registry for SatNav.

This module implements a registration system for models, allowing
dynamic model instantiation by name.

Reference:
    - MISSING_MODULES.md Section 7.2
"""

from typing import Dict, Type, Optional


class ModelRegistry:
    """Central registry for all VLN models in SatNav.
    
    This registry manages three categories of models:
    1. Baseline models - SatNav's own baseline implementations
    2. Custom models - User-defined models
    3. External adapters - Adapters for external models (NavID, NavILA, etc.)
    
    Example:
        >>> ModelRegistry.register_baseline("seq2seq", Seq2SeqPolicy)
        >>> model_class = ModelRegistry.get_model("seq2seq")
        >>> model = model_class.from_config(config, obs_space, act_space)
    """
    
    _baseline_models: Dict[str, Type] = {}
    _custom_models: Dict[str, Type] = {}
    _external_adapters: Dict[str, Type] = {}
    
    @classmethod
    def register_baseline(cls, name: str, model_class: Type):
        """Register a baseline model.
        
        Args:
            name: Unique name for the model (e.g., "seq2seq", "cma")
            model_class: The model class to register
        """
        if name in cls._baseline_models:
            raise ValueError(
                f"Baseline model '{name}' is already registered. "
                f"Use a different name or unregister the existing model."
            )
        cls._baseline_models[name] = model_class
    
    @classmethod
    def register_custom(cls, name: str, model_class: Type):
        """Register a custom (user-defined) model.
        
        Args:
            name: Unique name for the model
            model_class: The model class to register
        """
        if name in cls._custom_models:
            raise ValueError(
                f"Custom model '{name}' is already registered. "
                f"Use a different name or unregister the existing model."
            )
        cls._custom_models[name] = model_class
    
    @classmethod
    def register_external(cls, name: str, adapter_class: Type):
        """Register an external model adapter.
        
        Args:
            name: Unique name for the model (e.g., "navid", "navila")
            adapter_class: The adapter class to register
        """
        if name in cls._external_adapters:
            raise ValueError(
                f"External adapter '{name}' is already registered. "
                f"Use a different name or unregister the existing adapter."
            )
        cls._external_adapters[name] = adapter_class
    
    @classmethod
    def get_model(cls, name: str, model_type: str = "auto") -> Optional[Type]:
        """Get a registered model class by name.
        
        Args:
            name: Name of the model to retrieve
            model_type: Type of model to search for. Options:
                - "auto": Search all registries in order (baseline, custom, external)
                - "baseline": Search only baseline models
                - "custom": Search only custom models
                - "external": Search only external adapters
        
        Returns:
            The model class if found, otherwise None
        
        Raises:
            ValueError: If model_type is invalid or model not found
        """
        if model_type == "auto":
            # Search in order: baseline -> custom -> external
            if name in cls._baseline_models:
                return cls._baseline_models[name]
            elif name in cls._custom_models:
                return cls._custom_models[name]
            elif name in cls._external_adapters:
                return cls._external_adapters[name]
            else:
                raise ValueError(
                    f"Model '{name}' not found in any registry. "
                    f"Available models: "
                    f"baseline={list(cls._baseline_models.keys())}, "
                    f"custom={list(cls._custom_models.keys())}, "
                    f"external={list(cls._external_adapters.keys())}"
                )
        elif model_type == "baseline":
            if name not in cls._baseline_models:
                raise ValueError(
                    f"Baseline model '{name}' not found. "
                    f"Available: {list(cls._baseline_models.keys())}"
                )
            return cls._baseline_models[name]
        elif model_type == "custom":
            if name not in cls._custom_models:
                raise ValueError(
                    f"Custom model '{name}' not found. "
                    f"Available: {list(cls._custom_models.keys())}"
                )
            return cls._custom_models[name]
        elif model_type == "external":
            if name not in cls._external_adapters:
                raise ValueError(
                    f"External adapter '{name}' not found. "
                    f"Available: {list(cls._external_adapters.keys())}"
                )
            return cls._external_adapters[name]
        else:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of: 'auto', 'baseline', 'custom', 'external'"
            )
    
    @classmethod
    def list_models(cls) -> Dict[str, list]:
        """List all registered models.
        
        Returns:
            Dictionary with keys "baseline", "custom", "external",
            each mapping to a list of registered model names
        """
        return {
            "baseline": list(cls._baseline_models.keys()),
            "custom": list(cls._custom_models.keys()),
            "external": list(cls._external_adapters.keys()),
        }
    
    @classmethod
    def unregister(cls, name: str, model_type: str = "auto"):
        """Unregister a model.
        
        Args:
            name: Name of the model to unregister
            model_type: Type of model (same options as get_model)
        """
        if model_type == "auto":
            if name in cls._baseline_models:
                del cls._baseline_models[name]
            elif name in cls._custom_models:
                del cls._custom_models[name]
            elif name in cls._external_adapters:
                del cls._external_adapters[name]
            else:
                raise ValueError(f"Model '{name}' not found in any registry")
        elif model_type == "baseline":
            if name not in cls._baseline_models:
                raise ValueError(f"Baseline model '{name}' not found")
            del cls._baseline_models[name]
        elif model_type == "custom":
            if name not in cls._custom_models:
                raise ValueError(f"Custom model '{name}' not found")
            del cls._custom_models[name]
        elif model_type == "external":
            if name not in cls._external_adapters:
                raise ValueError(f"External adapter '{name}' not found")
            del cls._external_adapters[name]
        else:
            raise ValueError(f"Invalid model_type '{model_type}'")


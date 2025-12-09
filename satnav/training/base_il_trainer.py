#!/usr/bin/env python3
"""Base trainer for imitation learning in SatNav VLN tasks.

This module provides the base trainer class for imitation learning,
adapted from VLN-CE's BaseVLNCETrainer.

Reference: VLN-CE vlnce_baselines/common/base_il_trainer.py
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from satnav.models import ModelRegistry


class BaseILTrainer:
    """Base trainer for imitation learning.
    
    This trainer provides core functionality for:
    - Policy initialization from ModelRegistry
    - Optimizer management
    - Checkpoint saving/loading
    - Gradient updates for teacher forcing
    
    Attributes:
        config: Training configuration
        device: torch.device for computation
        policy: The policy network being trained
        optimizer: The optimizer for training
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the base IL trainer.
        
        Args:
            config: Training configuration (OmegaConf DictConfig)
        """
        self.config = config
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda", config.TORCH_GPU_ID)
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Policy and optimizer will be initialized later
        self.policy = None
        self.optimizer = None
        
        # Training state
        self.start_epoch = 0
        self.step_id = 0
    
    def _initialize_policy(
        self,
        config: DictConfig,
        load_from_ckpt: bool,
        observation_space: Dict[str, Any],
        action_space: Dict[str, Any]
    ) -> None:
        """Initialize the policy from ModelRegistry.
        
        Args:
            config: Model configuration
            load_from_ckpt: Whether to load from checkpoint
            observation_space: Observation space definition (dict format)
            action_space: Action space definition (dict format)
        """
        # Get policy class from ModelRegistry
        policy_name = config.MODEL.policy_name
        print(f"Loading policy: {policy_name}")
        
        policy_class = ModelRegistry.get_model(policy_name)
        
        # Create policy instance
        self.policy = policy_class.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space
        )
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.IL.lr
        )
        
        # Load from checkpoint if specified
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint from: {ckpt_path}")
                ckpt_dict = self.load_checkpoint(ckpt_path)
                self.policy.load_state_dict(ckpt_dict['state_dict'])
                
                # Optionally load optimizer state
                if 'optim_state' in ckpt_dict:
                    self.optimizer.load_state_dict(ckpt_dict['optim_state'])
                
                # Restore training state
                if 'epoch' in ckpt_dict:
                    self.start_epoch = ckpt_dict['epoch'] + 1
                if 'step_id' in ckpt_dict:
                    self.step_id = ckpt_dict['step_id']
                
                print(f"Resumed from epoch {self.start_epoch}, step {self.step_id}")
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}, starting from scratch")
        
        # Count parameters
        params = sum(p.numel() for p in self.policy.parameters())
        params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"Agent parameters: {params}. Trainable: {params_t}")
        print("Finished setting up policy.")
    
    def _update_agent(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        not_done_masks: torch.Tensor,
        teacher_actions: torch.Tensor
    ) -> float:
        """Execute one gradient update step.
        
        Args:
            observations: Dict of observations (each with shape T*N, ...)
            prev_actions: Previous actions, shape (T*N, 1)
            not_done_masks: Episode boundary masks, shape (T*N, 1)
            teacher_actions: Ground truth actions, shape (T, N)
            
        Returns:
            Loss value (float)
        """
        T, N = teacher_actions.size()
        
        # Initialize RNN hidden states
        # Format: [num_layers, batch_size, hidden_size] as expected by RNNStateEncoder
        # Note: We need the actual RNN num_layers, not num_recurrent_layers
        # For GRU: num_recurrent_layers = num_layers
        # For LSTM: num_recurrent_layers = num_layers * 2 (because it stores both hidden and cell states)
        # The RNN itself expects [num_layers, batch_size, hidden_size], not [num_recurrent_layers, ...]
        # Get num_layers from the RNN module directly
        state_encoder = self.policy.net.state_encoder
        rnn_num_layers = state_encoder.rnn.num_layers
        
        rnn_states = torch.zeros(
            rnn_num_layers,
            N,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device
        )
        
        # Forward pass through policy
        distribution = self.policy.build_distribution(
            observations, rnn_states, prev_actions, not_done_masks
        )
        
        # Get logits and reshape to (T, N, num_actions)
        logits = distribution.logits.view(T, N, -1)
        
        # Compute cross-entropy loss
        # Permute logits to (T, num_actions, N) for F.cross_entropy
        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1),
            teacher_actions,
            reduction='mean'
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        action_loss.backward()
        
        # Gradient clipping (optional, but recommended for RNNs)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return action_loss.item()
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        step_id: int,
        loss: Optional[float] = None
    ) -> None:
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename (e.g., "best.pth")
            epoch: Current epoch number
            step_id: Current training step
            loss: Current loss value (optional)
        """
        checkpoint_dir = self.config.CHECKPOINT_FOLDER
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.policy.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'config': OmegaConf.to_container(self.config, resolve=True),
            'epoch': epoch,
            'step_id': step_id,
        }
        
        if loss is not None:
            checkpoint['loss'] = loss
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        # Use weights_only=False because checkpoint contains model state_dict and optimizer state
        # which are safe to load (they come from our own training process)
        return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    
    def _make_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
    
    def eval(self) -> None:
        """Main evaluation entry point.
        
        This method is called when run.py is invoked with --run-type eval.
        It loads the checkpoint specified in config.EVAL and runs evaluation.
        """
        print("=" * 80)
        print("Starting Evaluation")
        print("=" * 80)
        
        # Get evaluation config
        eval_config = self.config.get('EVAL', {})
        
        # Determine checkpoint path
        if 'CKPT_PATH' in eval_config:
            ckpt_path = eval_config.CKPT_PATH
        elif hasattr(self.config, 'IL') and hasattr(self.config.IL, 'ckpt_to_load'):
            ckpt_path = self.config.IL.ckpt_to_load
        else:
            print("Error: No checkpoint path specified in config")
            print("  Please set EVAL.CKPT_PATH or IL.ckpt_to_load")
            return
        
        # Validate checkpoint exists
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}")
            return
        
        # Run evaluation
        self._eval_checkpoint(ckpt_path, checkpoint_index=0)
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_index: int = 0
    ) -> Dict[str, float]:
        """Evaluate a single checkpoint (basic implementation stub).
        
        This is a basic stub implementation that validates checkpoint loading.
        
        TODO: Implement complete evaluation loop:
            - Create environment from config
            - Load checkpoint into policy
            - Initialize policy in eval mode
            - Run episodes and collect trajectories
            - Compute metrics (SPL, Success, Oracle Success, etc.)
            - Save results to JSON if EVAL.SAVE_RESULTS is true
            - Return aggregated metrics
        
        Args:
            checkpoint_path: Path to checkpoint file
            checkpoint_index: Index for logging purposes
            
        Returns:
            Dictionary of aggregated metrics (empty in stub implementation)
        """
        print(f"\nEvaluating checkpoint: {checkpoint_path}")
        print("Note: This is a basic evaluation stub")
        print("      Full evaluation implementation is TODO\n")
        
        # Get evaluation config
        from omegaconf import OmegaConf
        
        # Use OmegaConf.select() with default values  
        split = OmegaConf.select(self.config, 'EVAL.SPLIT', default='val_seen')
        episode_count = OmegaConf.select(self.config, 'EVAL.EPISODE_COUNT', default=-1)
        save_results = OmegaConf.select(self.config, 'EVAL.SAVE_RESULTS', default=True)
        
        # Handle None values explicitly
        if split is None:
            split = 'val_seen'
        if episode_count is None:
            episode_count = -1
        if save_results is None:
            save_results = True
        
        print(f"Evaluation configuration:")
        print(f"  Split: {split}")
        print(f"  Episode count: {episode_count if episode_count > 0 else 'all'}")
        print(f"  Save results: {save_results}")
        
        # Basic implementation: Just load and validate checkpoint
        try:
            ckpt = self.load_checkpoint(checkpoint_path)
            epoch = ckpt.get('epoch', 'N/A')
            step_id = ckpt.get('step_id', 'N/A')
            loss = ckpt.get('loss', 'N/A')
            
            print(f"\nCheckpoint loaded successfully:")
            print(f"  Epoch: {epoch}")
            print(f"  Step: {step_id}")
            print(f"  Loss: {loss}")
            
            # Check if state_dict exists
            if 'state_dict' in ckpt:
                num_params = sum(p.numel() for p in ckpt['state_dict'].values())
                print(f"  Parameters: {num_params:,}")
            
        except Exception as e:
            print(f"\nError loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        print("\n" + "=" * 80)
        print("Evaluation stub completed")
        print("TODO: Implement full evaluation with environment rollouts")
        print("=" * 80)
        
        return {}


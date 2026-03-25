#!/usr/bin/env python3
"""Base trainer for imitation learning in SatNav VLN tasks.

This module provides the base trainer class for imitation learning,
adapted from VLN-CE's BaseVLNCETrainer.

Reference: VLN-CE vlnce_baselines/common/base_il_trainer.py
"""

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig, OmegaConf

from satnav.models import ModelRegistry
from satnav.training.distributed import (
    get_rank,
    get_world_size,
    is_distributed_runtime,
    is_main_process,
    reduce_scalar,
)


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
        self.distributed = is_distributed_runtime(config)
        self.rank = get_rank(config)
        self.world_size = get_world_size(config)
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda", config.TORCH_GPU_ID)
        else:
            self.device = torch.device("cpu")
        
        print(
            f"Using device: {self.device}"
            f" | rank={self.rank}"
            f" | world_size={self.world_size}"
            f" | distributed={self.distributed}"
        )
        
        # Policy and optimizer will be initialized later
        self.policy = None
        self.optimizer = None
        
        # Training state
        self.start_epoch = 0
        self.step_id = 0
        
        # Scheduled sampling state
        self._init_scheduled_sampling()
        
        # Loss weighting state
        self._init_loss_weighting()

    def _init_swanlab(self) -> Tuple[Optional[object], bool]:
        """Initialize SwanLab logging. Returns (swanlab_module, use_swanlab)."""
        swanlab_cfg = getattr(self.config, "SWANLAB", {})
        if getattr(swanlab_cfg, "mode", "disabled") == "disabled":
            if is_main_process(self.config):
                print("SwanLab logging disabled")
            return None, False

        if not is_main_process(self.config):
            return None, False

        try:
            import swanlab

            callbacks = []
            if bool(getattr(swanlab_cfg, "use_wxwork_notification", False)):
                webhook_url = getattr(swanlab_cfg, "webhook_url", None)
                if webhook_url:
                    try:
                        from swanlab.plugin.notification import WxWebhookCallback

                        callbacks.append(
                            WxWebhookCallback(
                                webhook_url=webhook_url,
                                secret=getattr(swanlab_cfg, "secret", None),
                            )
                        )
                    except Exception as exc:
                        print(
                            f"Warning: SwanLab WXWork callback failed, "
                            f"continuing without notifications: {exc}"
                        )
                else:
                    print(
                        "Warning: SWANLAB.webhook_url is empty, "
                        "skipping WXWork notification"
                    )

            swanlab.init(
                project=getattr(swanlab_cfg, "project", "SatNav"),
                workspace=getattr(swanlab_cfg, "workspace", None),
                experiment_name=getattr(swanlab_cfg, "experiment_name", "satnav-train"),
                config=OmegaConf.to_container(self.config, resolve=True),
                logdir=getattr(swanlab_cfg, "logdir", "output/swanlab"),
                mode=getattr(swanlab_cfg, "mode", "local"),
                callbacks=callbacks or None,
            )
            return swanlab, True
        except ImportError:
            print("Warning: swanlab not installed, continuing without logging")
            return None, False
        except Exception as exc:
            print(f"Warning: SwanLab initialization failed, continuing without logging: {exc}")
            return None, False

    def _policy_module(self):
        """Return the underlying policy module, unwrapping DDP if needed."""
        if isinstance(self.policy, DistributedDataParallel):
            return self.policy.module
        return self.policy
    
    def _init_scheduled_sampling(self):
        """Initialize scheduled sampling configuration.
        
        Reads scheduled sampling settings from config:
        - IL.RECOLLECT_TRAINER.use_scheduled_sampling: Enable/disable scheduled sampling
        - IL.RECOLLECT_TRAINER.scheduled_sampling_ratio: Fixed ratio (0.0-1.0)
          or "decay" for exponential decay
        - IL.RECOLLECT_TRAINER.scheduled_sampling_p: Decay parameter (if using decay)
        """
        if isinstance(self.config, dict):
            il_config = self.config.get('IL', {})
            recollect_config = il_config.get('RECOLLECT_TRAINER', {})
            self.use_scheduled_sampling = recollect_config.get('use_scheduled_sampling', False)
            self.scheduled_sampling_ratio = recollect_config.get('scheduled_sampling_ratio', 0.0)
            self.scheduled_sampling_p = recollect_config.get('scheduled_sampling_p', 0.5)
        else:
            il_config = getattr(self.config, 'IL', {})
            recollect_config = getattr(il_config, 'RECOLLECT_TRAINER', {})
            self.use_scheduled_sampling = getattr(recollect_config, 'use_scheduled_sampling', False)
            self.scheduled_sampling_ratio = getattr(recollect_config, 'scheduled_sampling_ratio', 0.0)
            self.scheduled_sampling_p = getattr(recollect_config, 'scheduled_sampling_p', 0.5)
        
        # Track training progress for decay
        self._scheduled_sampling_decay_mode = (
            isinstance(self.scheduled_sampling_ratio, str) and 
            self.scheduled_sampling_ratio.lower() == 'decay'
        )
    
    def _init_loss_weighting(self):
        """Initialize loss weighting configuration.
        
        Reads loss weighting settings from config:
        - IL.use_class_weighting: Enable/disable class weighting
        - IL.class_weights: Dictionary of action weights
        """
        if isinstance(self.config, dict):
            il_config = self.config.get('IL', {})
            self.use_class_weighting = il_config.get('use_class_weighting', False)
            class_weights_dict = il_config.get('class_weights', {})
        else:
            il_config = getattr(self.config, 'IL', {})
            self.use_class_weighting = getattr(il_config, 'use_class_weighting', False)
            class_weights_dict = getattr(il_config, 'class_weights', {})
        
        # Convert class weights dict to tensor
        # Action indices: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        if self.use_class_weighting and class_weights_dict:
            # Default weights if not specified
            default_weights = {
                'STOP': 1.0,
                'MOVE_FORWARD': 1.0,
                'TURN_LEFT': 1.0,
                'TURN_RIGHT': 1.0
            }
            default_weights.update(class_weights_dict)
            
            # Create weight tensor in action order: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
            self.class_weights = torch.tensor([
                default_weights.get('STOP', 1.0),
                default_weights.get('MOVE_FORWARD', 1.0),
                default_weights.get('TURN_LEFT', 1.0),
                default_weights.get('TURN_RIGHT', 1.0)
            ], dtype=torch.float32)
        else:
            self.class_weights = None
    
    def _get_scheduled_sampling_ratio(self, epoch: int) -> float:
        """Get current scheduled sampling ratio.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Scheduled sampling ratio (0.0 = always teacher forcing, 1.0 = always autoregressive)
        """
        if not self.use_scheduled_sampling:
            return 0.0
        
        if self._scheduled_sampling_decay_mode:
            # Exponential decay: ratio = p^epoch
            # Starts at 1.0 (all teacher forcing) and decays to 0.0 (all autoregressive)
            # p=0.5 means: epoch 0: 1.0, epoch 1: 0.5, epoch 2: 0.25, epoch 3: 0.125, ...
            ratio = self.scheduled_sampling_p ** epoch
            return max(0.0, min(1.0, ratio))
        else:
            # Fixed ratio
            return float(self.scheduled_sampling_ratio)
    
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

        # Load checkpoint once; reuse dict for both state_dict and optimizer/epoch restoration
        ckpt_dict = None
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint from: {ckpt_path}")
                ckpt_dict = self.load_checkpoint(ckpt_path)
                self.policy.load_state_dict(ckpt_dict['state_dict'])
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}, starting from scratch")

        if self.distributed:
            self.policy = DistributedDataParallel(
                self.policy,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=bool(
                    OmegaConf.select(
                        config, "DISTRIBUTED.find_unused_parameters", default=False
                    )
                ),
            )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.IL.lr
        )

        if ckpt_dict is not None:
            if 'optim_state' in ckpt_dict:
                self.optimizer.load_state_dict(ckpt_dict['optim_state'])
            if 'epoch' in ckpt_dict:
                self.start_epoch = ckpt_dict['epoch'] + 1
            if 'step_id' in ckpt_dict:
                self.step_id = ckpt_dict['step_id']
            print(f"Resumed from epoch {self.start_epoch}, step {self.step_id}")
        
        # Count parameters
        policy_module = self._policy_module()
        params = sum(p.numel() for p in policy_module.parameters())
        params_t = sum(p.numel() for p in policy_module.parameters() if p.requires_grad)
        print(f"Agent parameters: {params}. Trainable: {params_t}")
        print("Finished setting up policy.")
    
    def _update_agent(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        not_done_masks: torch.Tensor,
        teacher_actions: torch.Tensor,
        epoch: int = 0,
        weights: Optional[torch.Tensor] = None
    ) -> float:
        """Execute one gradient update step with optional scheduled sampling.
        
        Args:
            observations: Dict of observations (each with shape T*N, ...)
            prev_actions: Previous actions (ground truth), shape (T*N, 1)
            not_done_masks: Episode boundary masks, shape (T*N, 1)
            teacher_actions: Ground truth actions, shape (T, N)
            epoch: Current epoch number (for scheduled sampling decay)
            
        Returns:
            Loss value (float)
        """
        T, N = teacher_actions.size()
        
        # Initialize model states
        # Let the model create its own initial state based on its architecture
        policy_module = self._policy_module()
        states = policy_module.net.get_initial_state(N, self.device)
        
        # Scheduled Sampling: Replace prev_actions with model predictions probabilistically
        # This helps bridge the gap between training (teacher forcing) and evaluation (autoregressive)
        # 
        # The idea: For each timestep t, prev_action[t] should be the action from timestep t-1.
        # With scheduled sampling, we probabilistically use the model's prediction at t-1
        # instead of the ground truth action at t-1.
        prev_actions_to_use = prev_actions.clone()
        
        if self.use_scheduled_sampling:
            sampling_ratio = self._get_scheduled_sampling_ratio(epoch)
            
            if sampling_ratio > 0.0:
                # First, do a forward pass with teacher forcing to get model predictions
                # Note: This is an approximation - ideally we'd do sequential processing,
                # but this is more efficient and still effective
                with torch.no_grad():
                    distribution_pred = policy_module.build_distribution(
                        observations, states, prev_actions, not_done_masks
                    )
                    # Get predicted actions (greedy)
                    predicted_actions = distribution_pred.mode()  # Shape: (T*N, 1)
                
                # Reshape to (T, N, 1) for easier manipulation
                predicted_actions_reshaped = predicted_actions.view(T, N, 1)
                prev_actions_reshaped = prev_actions.view(T, N, 1)
                sampled_prev_actions = prev_actions_reshaped.clone()
                
                # Apply scheduled sampling: for each timestep t > 0, decide whether to use
                # predicted action from t-1 or teacher action from t-1 as prev_action
                # At timestep 0, prev_action is always STOP (0) - keep as is
                for t in range(1, T):
                    # Decide whether to use predicted action from t-1 or teacher action from t-1
                    use_predicted = random.random() < sampling_ratio
                    if use_predicted:
                        # Use model's prediction at t-1 as prev_action for timestep t
                        sampled_prev_actions[t] = predicted_actions_reshaped[t-1]
                    # else: keep teacher action (already in sampled_prev_actions)
                
                # Reshape back to (T*N, 1)
                prev_actions_to_use = sampled_prev_actions.view(T * N, 1)
        
        # Forward pass through policy with (possibly modified) prev_actions
        distribution = policy_module.build_distribution(
            observations, states, prev_actions_to_use, not_done_masks
        )
        
        # Get logits and reshape to (T, N, num_actions)
        logits = distribution.logits.view(T, N, -1)
        
        # Compute cross-entropy loss
        # Permute logits to (T, num_actions, N) for F.cross_entropy
        # Use class weights if enabled
        if self.use_class_weighting and self.class_weights is not None:
            # Move class weights to device
            class_weights_device = self.class_weights.to(self.device)
            action_loss = F.cross_entropy(
                logits.permute(0, 2, 1),
                teacher_actions,
                weight=class_weights_device,
                reduction='none'  # Use 'none' to apply sample weights
            )
        else:
            action_loss = F.cross_entropy(
                logits.permute(0, 2, 1),
                teacher_actions,
                reduction='none'  # Use 'none' to apply sample weights
            )
        
        # Apply inflection weights (sample-level weighting)
        # weights may be shaped as (T,), (T, 1), or (T, N)
        if weights is not None:
            if weights.dim() == 1:
                weights_expanded = weights.view(T, 1).expand(T, N)
            elif weights.dim() == 2 and weights.size(0) == T:
                if weights.size(1) == 1:
                    weights_expanded = weights.expand(T, N)
                elif weights.size(1) == N:
                    weights_expanded = weights
                else:
                    raise ValueError(
                        f"Unexpected weights shape {tuple(weights.shape)} "
                        f"for teacher actions {(T, N)}"
                    )
            else:
                raise ValueError(
                    f"Unexpected weights shape {tuple(weights.shape)} "
                    f"for teacher actions {(T, N)}"
                )

            weights_expanded = weights_expanded.to(action_loss.device)
            total_weight = weights_expanded.sum().clamp_min(1e-6)
            action_loss = (weights_expanded * action_loss).sum() / total_weight
        else:
            # No sample weighting: simple mean
            action_loss = action_loss.mean()
        
        # Debug: Print weight info occasionally (first batch of first epoch only)
        if self.step_id == 0 and epoch == 0 and is_main_process(self.config):
            if self.use_class_weighting and self.class_weights is not None:
                print(f"\n[Weight Debug] Class weights: {self.class_weights.tolist()}")
            if weights is not None:
                inflection_count = (weights > 1.0).sum().item()
                print(f"[Weight Debug] Inflection points in batch: {inflection_count}/{weights.numel()}")
                print(f"[Weight Debug] Weight range: [{weights.min().item():.2f}, {weights.max().item():.2f}]")
        
        # Backward pass
        self.optimizer.zero_grad()
        action_loss.backward()
        
        # Gradient clipping (optional, but recommended for RNNs)
        # Increased max_norm to allow larger gradient updates for better overfitting
        torch.nn.utils.clip_grad_norm_(policy_module.parameters(), max_norm=5.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return reduce_scalar(self.config, action_loss.item(), device=self.device, average=True)
    
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
            'state_dict': self._policy_module().state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'config': OmegaConf.to_container(self.config, resolve=True),
            'epoch': epoch,
            'step_id': step_id,
        }
        
        if loss is not None:
            checkpoint['loss'] = loss
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        if is_main_process(self.config):
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

    def _distributed_min(self, value: int) -> int:
        """Compute the global minimum across ranks."""
        tensor = torch.tensor(int(value), device=self.device)
        if self.distributed:
            import torch.distributed as dist

            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def _distributed_sum(self, value: float) -> float:
        """Compute the global sum across ranks."""
        return reduce_scalar(self.config, value, device=self.device, average=False)
    
    def _make_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
    
    def eval(self) -> None:
        """Main evaluation entry point.
        
        This method is called when run.py is invoked with --run-type eval.
        It loads the checkpoint specified in config.EVAL and runs evaluation.
        
        This method synchronizes EVAL.SPLIT to DATASET.SPLIT to ensure the
        correct dataset split is loaded during evaluation.
        """
        from omegaconf import OmegaConf
        from satnav.training.evaluator import Evaluator
        
        print("=" * 80)
        print("Starting Evaluation")
        print("=" * 80)
        
        # Get evaluation split
        eval_split = OmegaConf.select(self.config, 'EVAL.SPLIT', default='val_seen')
        if eval_split is None:
            eval_split = 'val_seen'
        
        print(f"Synchronizing EVAL.SPLIT ({eval_split}) to DATASET.SPLIT")
        
        # Synchronize EVAL.SPLIT to DATASET.SPLIT (like VLN-CE)
        # This ensures the correct dataset split is loaded during evaluation
        # Ensure config is mutable
        try:
            # Try to defrost if frozen
            self.config.defrost()
        except Exception:
            # If defrost fails, convert to dict and recreate
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            self.config = OmegaConf.create(config_dict)
        
        # Update DATASET.SPLIT (create DATASET section if it doesn't exist)
        if not hasattr(self.config, 'DATASET'):
            self.config.DATASET = OmegaConf.create({})
        self.config.DATASET.SPLIT = eval_split
        
        # Freeze config again
        try:
            self.config.freeze()
        except Exception:
            pass  # Already frozen or not freezable
        
        print(f"  Updated DATASET.SPLIT to: {self.config.DATASET.SPLIT}")
        
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
        
        # Validate checkpoint exists (skip for non-learning agents)
        if ckpt_path is not None and not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}")
            return
        
        # Initialize policy if needed (lazy initialization)
        if self.policy is None:
            print(f"\nInitializing policy...")
            from satnav.dataset.recollect_dataset import RecollectionDataset
            
            # Create a temporary RecollectionDataset to get observation/action spaces
            temp_dataset = RecollectionDataset(self.config)
            
            observation_space = temp_dataset.observation_space
            action_space = temp_dataset.action_space
            
            self._initialize_policy(
                self.config,
                load_from_ckpt=False,  # We'll load manually in evaluator
                observation_space=observation_space,
                action_space=action_space
            )
            
            print(f"  Policy initialized")
        
        # Create evaluator and run evaluation
        evaluator = Evaluator(self.config, self.device)
        evaluator.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            policy=self.policy,
            checkpoint_index=0
        )

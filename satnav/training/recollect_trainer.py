#!/usr/bin/env python3
"""Recollect trainer for real-time trajectory collection.

This trainer collects trajectories from the environment in real-time
using teacher forcing and trains the policy with imitation learning.

Reference: VLN-CE vlnce_baselines/recollect_trainer.py
"""

import time
from typing import Optional

import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from satnav.training.base_il_trainer import BaseILTrainer
from satnav.dataset.recollect_dataset import RecollectionDataset
from satnav.training.utils import collate_fn
from satnav.training.registry import register_trainer


@register_trainer("recollect_trainer")
class RecollectTrainer(BaseILTrainer):
    """Trainer that collects trajectories in real-time and trains with IL.
    
    This trainer:
    1. Creates a RecollectionDataset that collects trajectories using ReferencePathFollower
    2. Trains the policy using teacher forcing with collected data
    3. Tracks best model and saves checkpoint
    4. Logs metrics to wandb
    
    Attributes:
        config: Training configuration
        device: torch.device for computation
        policy: The policy network being trained
        optimizer: The optimizer for training
        dataset: RecollectionDataset for data collection
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the recollect trainer.
        
        Args:
            config: Training configuration (OmegaConf DictConfig)
        """
        super().__init__(config)
        self.dataset: Optional[RecollectionDataset] = None
    
    def train(self) -> None:
        """Main training loop."""
        print("="*80)
        print("Starting RecollectTrainer")
        print("="*80)
        
        # Create checkpoint directory
        self._make_checkpoint_dir()
        
        # Initialize wandb
        if isinstance(self.config, dict):
            wandb_config = self.config.get('WANDB', {})
            use_wandb = wandb_config.get('mode', 'online') != 'disabled'
        else:
            wandb_config = getattr(self.config, 'WANDB', {})
            use_wandb = getattr(wandb_config, 'mode', 'online') != 'disabled'
        
        if use_wandb:
            try:
                import wandb
                if isinstance(self.config, dict):
                    wandb.init(
                        project=wandb_config.get('project', 'satnav-vln'),
                        name=wandb_config.get('run_name', 'recollect-seq2seq'),
                        entity=wandb_config.get('entity', None),
                        config=self.config,
                        mode=wandb_config.get('mode', 'online')
                    )
                else:
                    wandb.init(
                        project=getattr(wandb_config, 'project', 'satnav-vln'),
                        name=getattr(wandb_config, 'run_name', 'recollect-seq2seq'),
                        entity=getattr(wandb_config, 'entity', None),
                        config=OmegaConf.to_container(self.config, resolve=True),
                        mode=getattr(wandb_config, 'mode', 'online')
                    )
                if isinstance(self.config, dict):
                    project = wandb_config.get('project', 'satnav-vln')
                    run_name = wandb_config.get('run_name', 'recollect-seq2seq')
                else:
                    project = getattr(wandb_config, 'project', 'satnav-vln')
                    run_name = getattr(wandb_config, 'run_name', 'recollect-seq2seq')
                print(f"Initialized wandb: {project}/{run_name}")
            except ImportError:
                print("Warning: wandb not installed, continuing without logging")
                use_wandb = False
        else:
            print("W&B logging disabled")
            use_wandb = False
        
        # Create dataset
        print("\n" + "="*80)
        print("Creating dataset...")
        print("="*80)
        self.dataset = RecollectionDataset(self.config)
        
        # Create DataLoader
        # Disable pin_memory to avoid GPU memory issues with preloaded data
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.IL.batch_size,
            collate_fn=collate_fn,
            pin_memory=False,  # Disabled to avoid OOM with preloaded RGB images
            num_workers=0  # Single process
        )
        
        # Initialize policy
        print("\n" + "="*80)
        print("Initializing policy...")
        print("="*80)
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            self.dataset.observation_space,
            self.dataset.action_space
        )
        
        # Training loop
        print("\n" + "="*80)
        print(f"Training for {self.config.IL.epochs} epochs")
        print("="*80)
        
        best_loss = float('inf')
        
        for epoch in range(self.start_epoch, self.config.IL.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # Progress bar
            pbar = tqdm.tqdm(
                total=len(self.dataset.trajectories),
                desc=f"Epoch {epoch+1}/{self.config.IL.epochs}",
                dynamic_ncols=True
            )
            
            try:
                max_batches = len(self.dataset.trajectories) // self.config.IL.batch_size
                if max_batches == 0:
                    print(f"Warning: No batches to process! trajectories={len(self.dataset.trajectories)}, batch_size={self.config.IL.batch_size}")
                
                for batch_idx, batch in enumerate(dataloader):
                    batch_start_time = time.time()
                    
                    # Move batch to device
                    # Handle both old format (without weights) and new format (with weights)
                    if len(batch) == 4:
                        observations, prev_actions, not_done_masks, teacher_actions = batch
                        weights = None
                    else:
                        observations, prev_actions, not_done_masks, teacher_actions, weights = batch
                    
                    observations = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in observations.items()
                    }
                    prev_actions = prev_actions.to(self.device, non_blocking=True)
                    not_done_masks = not_done_masks.to(self.device, non_blocking=True)
                    teacher_actions = teacher_actions.to(self.device, non_blocking=True)
                    if weights is not None:
                        weights = weights.to(self.device, non_blocking=True)
                    
                    # Gradient update (pass epoch for scheduled sampling decay)
                    loss = self._update_agent(
                        observations,
                        prev_actions,
                        not_done_masks,
                        teacher_actions,
                        epoch=epoch,
                        weights=weights
                    )
                    
                    # Check for invalid loss
                    if not torch.isfinite(torch.tensor(loss)):
                        print(f"Warning: Invalid loss value {loss} at batch {batch_idx}")
                        continue
                    
                    # Accumulate loss
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.update(self.config.IL.batch_size)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'batch_time': f'{time.time() - batch_start_time:.2f}s'
                    })
                    
                    # Log to wandb
                    if use_wandb:
                        wandb.log({
                            'train/loss': loss,
                            'train/epoch': epoch,
                            'train/learning_rate': self.optimizer.param_groups[0]['lr']
                        }, step=self.step_id)
                    
                    self.step_id += 1
                    
                    # Break after processing all episodes once
                    if max_batches > 0 and batch_idx + 1 >= max_batches:
                        break
                        
            except StopIteration:
                pass
            except Exception as e:
                print(f"Error during training: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                pbar.close()
            
            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Number of batches: {num_batches}")
            
            # Check for invalid loss values
            if not (torch.isfinite(torch.tensor(avg_epoch_loss)) and num_batches > 0):
                print(f"  Warning: Invalid loss value ({avg_epoch_loss}) or no batches processed!")
                if num_batches == 0:
                    print("  No batches were processed in this epoch. Check dataset and batch_size configuration.")
                continue
            
            # Log epoch metrics to wandb
            if use_wandb:
                wandb.log({
                    'epoch/loss': avg_epoch_loss,
                    'epoch/time': epoch_time,
                    'epoch/number': epoch
                }, step=self.step_id)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
                self.save_checkpoint('best.pth', epoch, self.step_id, best_loss)
                
                if use_wandb:
                    # Save checkpoint to wandb
                    import os
                    checkpoint_path = os.path.join(
                        self.config.CHECKPOINT_FOLDER,
                        'best.pth'
                    )
                    wandb.save(checkpoint_path)
        
        print("\n" + "="*80)
        print("Training completed!")
        if best_loss == float('inf'):
            print("Warning: Best loss is inf. This usually means:")
            print("  - No batches were processed during training")
            print("  - Loss values were invalid (nan/inf)")
            print("  - Check dataset configuration and batch_size")
        else:
            print(f"Best loss: {best_loss:.4f}")
        print("="*80)
        
        # Finish wandb
        if use_wandb:
            wandb.finish()

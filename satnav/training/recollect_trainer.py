#!/usr/bin/env python3
"""Recollect trainer for real-time trajectory collection.

This trainer collects trajectories from the environment in real-time
using teacher forcing and trains the policy with imitation learning.

This trainer is retained for future recollection / online imitation work. The
current release training path for Seq2Seq and CMA uses OfflineTrainer.

Reference: VLN-CE vlnce_baselines/recollect_trainer.py
"""

import time
from typing import Optional

import torch
import tqdm
from omegaconf import DictConfig

from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.distributed import is_main_process
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
    4. Logs metrics to SwanLab
    
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

        swanlab_module, use_swanlab = self._init_swanlab()
        
        # Create dataset
        if is_main_process(self.config):
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
        if is_main_process(self.config):
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
        if is_main_process(self.config):
            print("\n" + "="*80)
            print(f"Training for {self.config.IL.epochs} epochs")
            print("="*80)
        
        best_loss = float('inf')
        
        try:
            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                num_batches = 0
                
                # Progress bar
                local_max_batches = len(self.dataset.trajectories) // self.config.IL.batch_size
                max_batches = self._distributed_min(local_max_batches)

                if is_main_process(self.config):
                    pbar = tqdm.tqdm(
                        total=max_batches * self.config.IL.batch_size,
                        desc=f"Epoch {epoch+1}/{self.config.IL.epochs}",
                        dynamic_ncols=True
                    )
                else:
                    pbar = None
                
                try:
                    if max_batches == 0:
                        if is_main_process(self.config):
                            print(
                                "Warning: No batches to process! "
                                f"min_batches={max_batches}, "
                                f"local_trajectories={len(self.dataset.trajectories)}, "
                                f"batch_size={self.config.IL.batch_size}"
                            )
                        continue
                    
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
                        if pbar is not None:
                            pbar.update(self.config.IL.batch_size)
                            pbar.set_postfix({
                                'loss': f'{loss:.4f}',
                                'batch_time': f'{time.time() - batch_start_time:.2f}s'
                            })

                        if is_main_process(self.config) and use_swanlab and swanlab_module is not None:
                            swanlab_module.log(
                                {
                                    "train/loss": loss,
                                    "train/epoch": epoch,
                                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                                },
                                step=self.step_id,
                            )
                        
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
                    if pbar is not None:
                        pbar.close()
                
                # Compute average loss for epoch
                global_epoch_loss = self._distributed_sum(epoch_loss)
                global_num_batches = self._distributed_sum(num_batches)
                avg_epoch_loss = global_epoch_loss / max(global_num_batches, 1.0)
                epoch_time = time.time() - epoch_start_time
                
                if is_main_process(self.config):
                    print(f"\nEpoch {epoch+1} completed:")
                    print(f"  Average Loss: {avg_epoch_loss:.4f}")
                    print(f"  Epoch Time: {epoch_time:.2f}s")
                    print(f"  Number of batches: {int(global_num_batches)}")
                
                # Check for invalid loss values
                if not (torch.isfinite(torch.tensor(avg_epoch_loss)) and global_num_batches > 0):
                    if is_main_process(self.config):
                        print(f"  Warning: Invalid loss value ({avg_epoch_loss}) or no batches processed!")
                        if global_num_batches == 0:
                            print("  No batches were processed in this epoch. Check dataset sharding and batch_size configuration.")
                    continue

                if is_main_process(self.config) and use_swanlab and swanlab_module is not None:
                    swanlab_module.log(
                        {
                            "train/epoch_loss": avg_epoch_loss,
                            "train/epoch_time": epoch_time,
                        },
                        step=self.step_id,
                    )
                
                # Save best model
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    if is_main_process(self.config):
                        print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
                    self.save_checkpoint('best.pth', epoch, self.step_id, best_loss)
        finally:
            if use_swanlab and swanlab_module is not None and is_main_process(self.config):
                swanlab_module.finish()

        if is_main_process(self.config):
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

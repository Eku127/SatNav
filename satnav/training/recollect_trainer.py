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
                for batch_idx, batch in enumerate(dataloader):
                    batch_start_time = time.time()
                    
                    # Move batch to device
                    observations, prev_actions, not_done_masks, teacher_actions = batch
                    observations = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in observations.items()
                    }
                    prev_actions = prev_actions.to(self.device, non_blocking=True)
                    not_done_masks = not_done_masks.to(self.device, non_blocking=True)
                    teacher_actions = teacher_actions.to(self.device, non_blocking=True)
                    
                    # Gradient update
                    loss = self._update_agent(
                        observations,
                        prev_actions,
                        not_done_masks,
                        teacher_actions
                    )
                    
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
                    if batch_idx + 1 >= len(self.dataset.trajectories) // self.config.IL.batch_size:
                        break
                        
            except StopIteration:
                pass
            finally:
                pbar.close()
            
            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
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
        print(f"Best loss: {best_loss:.4f}")
        print("="*80)
        
        # Finish wandb
        if use_wandb:
            wandb.finish()


def main():
    """Main entry point for training."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Train SatNav VLN agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config file"
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Modify config options from command line"
    )
    
    args = parser.parse_args()
    
    # Load configs
    from omegaconf import OmegaConf
    
    # Load training config
    training_config = OmegaConf.load(args.config)
    
    # Auto-load task config if specified in training config
    # This follows VLN-CE style: task config path specified in training config
    task_config = None
    if "TASK_CONFIG_PATH" in training_config:
        task_config_path = training_config.TASK_CONFIG_PATH
        if os.path.exists(task_config_path):
            print(f"Loading task config from: {task_config_path}")
            task_config = OmegaConf.load(task_config_path)
        else:
            print(f"Warning: Task config not found at {task_config_path}, skipping")
    
    # Load model config
    model_config = OmegaConf.load(args.model_config)
    
    # Merge configs with priority: training < task < model
    # This allows model config to override task config if needed
    if task_config is not None:
        config = OmegaConf.merge(training_config, task_config, model_config)
    else:
        config = OmegaConf.merge(training_config, model_config)
    
    # Apply command line overrides (highest priority)
    if args.opts:
        override_config = OmegaConf.from_dotlist(args.opts)
        config = OmegaConf.merge(config, override_config)
    
    # Create trainer and train
    trainer = RecollectTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


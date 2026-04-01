#!/usr/bin/env python3
"""Offline trainer for pre-rendered trajectory_data imitation learning."""

import os
import time
from typing import Optional

import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from satnav.dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.distributed import is_main_process
from satnav.training.registry import register_trainer
from satnav.training.utils import collate_fn


@register_trainer("offline_trainer")
class OfflineTrainer(BaseILTrainer):
    """Trainer for offline imitation learning on pre-rendered trajectories."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.dataset: Optional[OfflineTrajectoryDataset] = None

    def train(self) -> None:
        print("=" * 80)
        print("Starting OfflineTrainer")
        print("=" * 80)

        self._make_checkpoint_dir()

        swanlab_module, use_swanlab = self._init_swanlab()

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("Creating offline dataset...")
            print("=" * 80)
        self.dataset = OfflineTrajectoryDataset(self.config)

        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                self.dataset,
                shuffle=True,
                drop_last=False,
            )

        offline_cfg = self.config.IL.OFFLINE
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.IL.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=int(getattr(offline_cfg, "num_workers", 0)),
            collate_fn=collate_fn,
            pin_memory=bool(getattr(offline_cfg, "pin_memory", True)),
            drop_last=False,
        )

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("Initializing policy...")
            print("=" * 80)
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            self.dataset.observation_space,
            self.dataset.action_space,
        )

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print(f"Training for {self.config.IL.epochs} epochs")
            print("=" * 80)

        best_loss = float("inf")

        try:
            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                num_batches = 0

                if sampler is not None:
                    sampler.set_epoch(epoch)

                local_max_batches = len(dataloader)
                max_batches = self._distributed_min(local_max_batches)

                if is_main_process(self.config):
                    pbar = tqdm.tqdm(
                        total=max_batches,
                        desc=f"Epoch {epoch + 1}/{self.config.IL.epochs}",
                        dynamic_ncols=True,
                    )
                else:
                    pbar = None

                try:
                    if max_batches == 0:
                        if is_main_process(self.config):
                            print("Warning: No offline batches to process")
                        continue

                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx >= max_batches:
                            break

                        batch_start_time = time.time()
                        observations, prev_actions, not_done_masks, teacher_actions, weights = batch

                        observations = {
                            key: value.to(self.device, non_blocking=True)
                            for key, value in observations.items()
                        }
                        prev_actions = prev_actions.to(self.device, non_blocking=True)
                        not_done_masks = not_done_masks.to(self.device, non_blocking=True)
                        teacher_actions = teacher_actions.to(self.device, non_blocking=True)
                        weights = weights.to(self.device, non_blocking=True)

                        loss = self._update_agent(
                            observations,
                            prev_actions,
                            not_done_masks,
                            teacher_actions,
                            epoch=epoch,
                            weights=weights,
                        )

                        if not torch.isfinite(torch.tensor(loss)):
                            print(f"Warning: Invalid loss value {loss} at batch {batch_idx}")
                            continue

                        epoch_loss += loss
                        num_batches += 1

                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "loss": f"{loss:.4f}",
                                    "batch_time": f"{time.time() - batch_start_time:.2f}s",
                                }
                            )

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
                finally:
                    if pbar is not None:
                        pbar.close()

                global_epoch_loss = self._distributed_sum(epoch_loss)
                global_num_batches = self._distributed_sum(num_batches)
                avg_epoch_loss = global_epoch_loss / max(global_num_batches, 1.0)
                epoch_time = time.time() - epoch_start_time

                if is_main_process(self.config):
                    print(f"\nEpoch {epoch + 1} completed:")
                    print(f"  Average Loss: {avg_epoch_loss:.4f}")
                    print(f"  Epoch Time: {epoch_time:.2f}s")
                    print(f"  Number of batches: {int(global_num_batches)}")

                if not (torch.isfinite(torch.tensor(avg_epoch_loss)) and global_num_batches > 0):
                    if is_main_process(self.config):
                        print("  Warning: Invalid loss value or no batches processed")
                    continue

                if is_main_process(self.config) and use_swanlab and swanlab_module is not None:
                    swanlab_module.log(
                        {
                            "train/epoch_loss": avg_epoch_loss,
                            "train/epoch_time": epoch_time,
                        },
                        step=self.step_id,
                    )

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    if is_main_process(self.config):
                        print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
                    self.save_checkpoint("best.pth", epoch, self.step_id, best_loss)
        finally:
            if use_swanlab and swanlab_module is not None and is_main_process(self.config):
                swanlab_module.finish()

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("Offline training completed!")
            if best_loss == float("inf"):
                print("Warning: no valid offline batches were processed")
            else:
                print(f"Best loss: {best_loss:.4f}")
            print("=" * 80)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self) -> None:
        """Evaluation entry point. Uses OfflineTrajectoryDataset for policy init (no simulator needed)."""
        from omegaconf import OmegaConf
        from satnav.training.evaluator import Evaluator

        print("=" * 80)
        print("Starting Evaluation")
        print("=" * 80)

        eval_split = OmegaConf.select(self.config, "EVAL.SPLIT", default="val_seen") or "val_seen"
        print(f"Evaluating on split: {eval_split}")

        try:
            self.config.defrost()
        except Exception:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            self.config = OmegaConf.create(config_dict)

        if not hasattr(self.config, "DATASET"):
            self.config.DATASET = OmegaConf.create({})
        self.config.DATASET.SPLIT = eval_split

        try:
            self.config.freeze()
        except Exception:
            pass

        eval_cfg = getattr(self.config, "EVAL", {})
        ckpt_path = getattr(eval_cfg, "CKPT_PATH", None) or getattr(
            getattr(self.config, "IL", {}), "ckpt_to_load", None
        )
        if not ckpt_path:
            print("Error: No checkpoint path specified. Set EVAL.CKPT_PATH in config.")
            return
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}")
            return

        if self.policy is None:
            print("\nInitializing policy from offline observation/action space...")
            if self.dataset is None:
                self.dataset = OfflineTrajectoryDataset(self.config)
            self._initialize_policy(
                self.config,
                load_from_ckpt=False,
                observation_space=self.dataset.observation_space,
                action_space=self.dataset.action_space,
            )

        evaluator = Evaluator(self.config, self.device)
        evaluator.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            policy=self.policy,
            checkpoint_index=0,
        )

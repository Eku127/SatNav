#!/usr/bin/env python3
"""DAgger trainer for SatNav online aggregation and supervised IL."""

import gc
import os
import time
from typing import Optional

import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from satnav.dataset.dagger_dataset import DaggerCollector, DaggerTrajectoryDataset
from satnav.training.base_il_trainer import BaseILTrainer
from satnav.training.distributed import barrier, get_rank, get_world_size, is_main_process
from satnav.training.registry import register_trainer
from satnav.training.utils import collate_fn


@register_trainer("dagger_trainer")
class DaggerTrainer(BaseILTrainer):
    """Trainer that alternates dataset aggregation and supervised updates."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.collector: Optional[DaggerCollector] = None

    def _make_dirs(self) -> None:
        self._make_checkpoint_dir()
        dagger_cfg = getattr(self.config.IL, "DAGGER", {})
        split = getattr(self.config.DATASET, "SPLIT", "train")
        storage_dir = str(
            getattr(dagger_cfg, "storage_dir", "output/seq2seq_dagger/datasets/{split}")
        ).format(split=split)
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)

    def _build_dataset(self) -> DaggerTrajectoryDataset:
        dagger_cfg = getattr(self.config.IL, "DAGGER", {})
        split = getattr(self.config.DATASET, "SPLIT", "train")
        storage_dir = str(
            getattr(dagger_cfg, "storage_dir", "output/seq2seq_dagger/datasets/{split}")
        ).format(split=split)

        return DaggerTrajectoryDataset(
            storage_dir=storage_dir,
            use_inflection_weighting=bool(
                getattr(self.config.IL, "use_inflection_weighting", False)
            ),
            inflection_weight_coef=float(
                getattr(self.config.IL, "inflection_weight_coef", 3.2)
            ),
            batch_size=int(self.config.IL.batch_size),
            preload_size=int(getattr(dagger_cfg, "preload_size", 100)),
            rank=get_rank(self.config),
            world_size=get_world_size(self.config),
        )

    def train(self) -> None:
        print("=" * 80)
        print("Starting DaggerTrainer")
        print("=" * 80)

        self._make_dirs()

        swanlab_module, use_swanlab = self._init_swanlab()

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("Creating DAgger collector...")
            print("=" * 80)

        self.collector = DaggerCollector(self.config)

        dagger_cfg = getattr(self.config.IL, "DAGGER", {})
        if bool(getattr(dagger_cfg, "reset_dataset_on_start", False)):
            # Only rank 0 deletes files; barrier ensures all other ranks wait
            # until the deletion is complete before proceeding.
            if is_main_process(self.config):
                print("Resetting existing DAgger aggregated dataset")
                self.collector.reset_storage()
            barrier(self.config)

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("Initializing policy...")
            print("=" * 80)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            self.collector.observation_space,
            self.collector.action_space,
        )

        iterations = int(getattr(dagger_cfg, "iterations", 1))
        epochs_per_iter = int(self.config.IL.epochs)
        num_workers = int(getattr(dagger_cfg, "num_workers", 0))
        pin_memory = bool(getattr(dagger_cfg, "pin_memory", True))

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print(f"Running {iterations} DAgger iterations")
            print("=" * 80)

        best_loss = float("inf")
        global_epoch_idx = 0

        try:
            for dagger_iter in range(iterations):
                if is_main_process(self.config):
                    print("\n" + "=" * 80)
                    print(f"DAgger Iteration {dagger_iter + 1}/{iterations}: collecting rollouts")
                    print("=" * 80)

                collect_stats = self.collector.collect(
                    self.policy,
                    self.device,
                    iteration=dagger_iter + (1 if self.config.IL.load_from_ckpt else 0),
                )

                if is_main_process(self.config):
                    print(
                        f"  beta={collect_stats['beta']:.4f}, "
                        f"collected={collect_stats['collected_episodes']}, "
                        f"skipped={collect_stats['skipped_episodes']}, "
                        f"dataset_size={collect_stats['dataset_size']}"
                    )
                    print(f"  expert_action_counts={collect_stats['expert_action_counts']}")
                    print(f"  policy_action_counts={collect_stats['policy_action_counts']}")
                    print(f"  executed_action_counts={collect_stats['executed_action_counts']}")
                    print(f"  stop_labels={collect_stats['stop_labels']}")

                if is_main_process(self.config) and use_swanlab and swanlab_module is not None:
                    swanlab_module.log(
                        {
                            "dagger/beta": collect_stats["beta"],
                            "dagger/collected_episodes": collect_stats["collected_episodes"],
                            "dagger/skipped_episodes": collect_stats["skipped_episodes"],
                            "dagger/dataset_size": collect_stats["dataset_size"],
                            "dagger/stop_labels": collect_stats["stop_labels"],
                        },
                        step=self.step_id,
                    )

                # Ensure all ranks finish writing before any rank starts reading for training.
                barrier(self.config)

                policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
                policy_module.train()

                if collect_stats["dataset_size"] == 0:
                    if is_main_process(self.config):
                        print("Warning: no aggregated trajectories available, skipping training")
                    continue

                dataset = self._build_dataset()
                # Align ranks after dataset scanning so no rank starts the first
                # DDP all-reduce while another rank is still enumerating files.
                barrier(self.config)
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.config.IL.batch_size),
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=collate_fn,
                    drop_last=False,
                )

                for epoch in range(epochs_per_iter):
                    epoch_start_time = time.time()
                    epoch_loss = 0.0
                    num_batches = 0

                    if is_main_process(self.config):
                        total_batches = max((len(dataset) + self.config.IL.batch_size - 1) // self.config.IL.batch_size, 1)
                        pbar = tqdm.tqdm(
                            total=total_batches,
                            desc=(
                                f"Dagger {dagger_iter + 1}/{iterations} "
                                f"Epoch {epoch + 1}/{epochs_per_iter}"
                            ),
                            dynamic_ncols=True,
                        )
                    else:
                        pbar = None

                    try:
                        for batch in dataloader:
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
                                epoch=global_epoch_idx,
                                weights=weights,
                            )

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
                                        "train/dagger_iteration": dagger_iter,
                                        "train/epoch": global_epoch_idx,
                                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                                    },
                                    step=self.step_id,
                                )

                            self.step_id += 1
                    finally:
                        if pbar is not None:
                            pbar.close()

                    avg_epoch_loss = epoch_loss / max(num_batches, 1)
                    epoch_time = time.time() - epoch_start_time

                    if is_main_process(self.config):
                        print(f"\nIteration {dagger_iter + 1}, epoch {epoch + 1} completed:")
                        print(f"  Average Loss: {avg_epoch_loss:.4f}")
                        print(f"  Epoch Time: {epoch_time:.2f}s")
                        print(f"  Number of batches: {num_batches}")

                    if is_main_process(self.config) and use_swanlab and swanlab_module is not None:
                        swanlab_module.log(
                            {
                                "train/epoch_loss": avg_epoch_loss,
                                "train/epoch_time": epoch_time,
                            },
                            step=self.step_id,
                        )

                    ckpt_name = f"ckpt.iter{dagger_iter:02d}.epoch{epoch:02d}.pth"
                    self.save_checkpoint(ckpt_name, global_epoch_idx, self.step_id, avg_epoch_loss)

                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        if is_main_process(self.config):
                            print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
                        self.save_checkpoint("best.pth", global_epoch_idx, self.step_id, best_loss)

                    global_epoch_idx += 1

                del dataloader
                del dataset
                gc.collect()
                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()

                # All ranks finish training before starting the next collect phase.
                barrier(self.config)
        finally:
            if use_swanlab and swanlab_module is not None and is_main_process(self.config):
                swanlab_module.finish()

        if is_main_process(self.config):
            print("\n" + "=" * 80)
            print("DAgger training completed!")
            if best_loss == float("inf"):
                print("Warning: no valid DAgger batches were processed")
            else:
                print(f"Best loss: {best_loss:.4f}")
            print("=" * 80)

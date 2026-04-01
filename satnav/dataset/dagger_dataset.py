#!/usr/bin/env python3
"""DAgger dataset and collector for SatNav IL training."""

import copy
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from satnav.core.config import get_success_distance_default
from satnav.navigation import ReferencePathFollower
from satnav.task.actions import Action
from satnav.training.distributed import barrier, get_rank, get_world_size, is_distributed_runtime
from satnav.utils.build_vocab import (
    VocabDict,
    build_vocab_from_dataset,
    tokenize_instruction_in_observation,
)


class DaggerCollector:
    """Collect on-policy trajectories with expert labels for DAgger."""

    def __init__(self, config: DictConfig, target_rgb_size: int = 224):
        self.config = config
        self.rank = get_rank(config)
        self.world_size = get_world_size(config)
        self.target_rgb_size = target_rgb_size

        dagger_cfg = getattr(config.IL, "DAGGER", {})
        split = getattr(config.DATASET, "SPLIT", "train")
        storage_dir = getattr(
            dagger_cfg,
            "storage_dir",
            "output/seq2seq_dagger/datasets/{split}",
        )
        self.storage_dir = Path(str(storage_dir).format(split=split))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / f"manifest_rank{self.rank:02d}.tsv"

        training_config = self._filter_topdown_map_for_training(config)

        from satnav.core.env import Env

        print(
            "Creating DAgger environment..."
            f" rank={self.rank}"
            f" world_size={self.world_size}"
        )
        self.env = Env(training_config, cycle=True)

        goal_radius = get_success_distance_default(config, default=10.0)
        turn_angle = getattr(config.SIMULATOR, "TURN_ANGLE", 15.0)
        self.expert = ReferencePathFollower(goal_radius=goal_radius, turn_angle=turn_angle)

        self.vocab = self._load_vocabulary()
        self.max_instruction_len = int(
            getattr(dagger_cfg, "max_instruction_len", 200)
        )
        self.max_traj_len = int(getattr(dagger_cfg, "max_traj_len", 500))
        self.target_rgb_size = int(
            getattr(
                dagger_cfg,
                "rgb_size",
                getattr(getattr(config.IL, "OFFLINE", {}), "rgb_size", target_rgb_size),
            )
        )
        self.deterministic_policy = bool(
            getattr(dagger_cfg, "deterministic_policy", False)
        )

        all_episode_indices = list(range(len(self.env._dataset.episodes)))
        self._episode_indices = all_episode_indices[self.rank::self.world_size]
        random.shuffle(self._episode_indices)
        self._current_episode_idx = 0

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _filter_topdown_map_for_training(self, config: DictConfig) -> DictConfig:
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_dict = copy.deepcopy(config_dict)

        if "TASK" in config_dict and "MEASUREMENTS" in config_dict["TASK"]:
            measurements = config_dict["TASK"]["MEASUREMENTS"]
            if isinstance(measurements, list):
                config_dict["TASK"]["MEASUREMENTS"] = [
                    m for m in measurements if m.upper() != "TOP_DOWN_MAP"
                ]

        return OmegaConf.create(config_dict)

    def _load_vocabulary(self) -> VocabDict:
        vocab_file = getattr(self.config.DATASET, "vocab_file", None)
        if vocab_file:
            print(f"Loading DAgger vocabulary from: {vocab_file}")
            vocab = VocabDict.load(vocab_file)
        else:
            dataset_path = self.config.DATASET.DATA_PATH.format(
                split=self.config.DATASET.SPLIT
            )
            print(f"Building DAgger vocabulary from dataset: {dataset_path}")
            vocab = build_vocab_from_dataset(dataset_path)

        print(f"Dagger vocabulary size: {len(vocab)}")
        return vocab

    def _tokenize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_instruction_in_observation(
            obs,
            self.vocab,
            max_length=self.max_instruction_len,
            output_format="tensor",
        )

    def _resize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[0] == self.target_rgb_size and rgb.shape[1] == self.target_rgb_size:
            return rgb

        rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
        rgb_resized = F.interpolate(
            rgb_chw.float(),
            size=(self.target_rgb_size, self.target_rgb_size),
            mode="bilinear",
            align_corners=False,
        )
        return rgb_resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)

    def _prepare_observation_for_storage(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs_dict: Dict[str, torch.Tensor] = {}

        if "rgb" in obs:
            rgb = obs["rgb"]
            if not isinstance(rgb, torch.Tensor):
                rgb = torch.from_numpy(rgb)
            if rgb.dtype != torch.uint8:
                rgb = rgb.to(torch.uint8)
            obs_dict["rgb"] = self._resize_rgb(rgb).cpu()

        if "instruction" in obs:
            instruction = obs["instruction"]
            if not isinstance(instruction, torch.Tensor):
                instruction = torch.as_tensor(instruction)
            obs_dict["instruction"] = instruction.cpu()

        return obs_dict

    def _prepare_observation_for_policy(
        self, obs: Dict[str, Any], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        obs_batch: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value.unsqueeze(0).to(device)
            else:
                tensor_value = torch.from_numpy(value).unsqueeze(0).to(device)
            obs_batch[key] = tensor_value
        return obs_batch

    def _get_next_episode(self):
        if not self._episode_indices:
            return None

        if self._current_episode_idx >= len(self._episode_indices):
            random.shuffle(self._episode_indices)
            self._current_episode_idx = 0

        episode_idx = self._episode_indices[self._current_episode_idx]
        self._current_episode_idx += 1
        return self.env._dataset.episodes[episode_idx]

    def get_beta(self, iteration: int) -> float:
        dagger_cfg = getattr(self.config.IL, "DAGGER", {})
        beta_mode = str(getattr(dagger_cfg, "beta_mode", "exponential")).lower()
        beta_start = float(getattr(dagger_cfg, "beta_start", 1.0))
        min_beta = float(getattr(dagger_cfg, "min_beta", 0.0))

        if beta_mode == "constant":
            beta = beta_start
        else:
            beta_decay = float(getattr(dagger_cfg, "beta_decay", 0.5))
            beta = beta_start * (beta_decay ** iteration)

        return max(min_beta, min(1.0, beta))

    def reset_storage(self) -> None:
        for path in self.storage_dir.glob("traj_*.pt"):
            path.unlink()
        for path in self.storage_dir.glob("manifest_rank*.tsv"):
            path.unlink()

    def count_trajectories(self) -> int:
        return len(list(self.storage_dir.glob("traj_*.pt")))

    def _append_manifest_entry(self, trajectory_path: Path, trajectory_len: int) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{trajectory_path.name}\t{trajectory_len}\n")

    def _save_trajectory(
        self,
        trajectory_id: int,
        episode,
        observations: List[Dict[str, torch.Tensor]],
        prev_actions: List[int],
        teacher_actions: List[int],
    ) -> Path:
        stacked_obs = defaultdict(list)
        for obs in observations:
            for sensor, value in obs.items():
                stacked_obs[sensor].append(value)

        payload_obs = {
            sensor: torch.stack(values, dim=0) for sensor, values in stacked_obs.items()
        }

        payload = {
            "episode_id": str(episode.episode_id),
            "trajectory_id": str(getattr(episode, "trajectory_id", "")),
            "trajectory_type": str(getattr(episode, "trajectory_type", "")),
            "observations": payload_obs,
            "prev_actions": torch.tensor(prev_actions, dtype=torch.long),
            "teacher_actions": torch.tensor(teacher_actions, dtype=torch.long),
        }

        output_path = self.storage_dir / f"traj_{trajectory_id:08d}.pt"
        torch.save(payload, output_path)
        return output_path

    def collect(
        self,
        policy,
        device: torch.device,
        iteration: int,
    ) -> Dict[str, Any]:
        dagger_cfg = getattr(self.config.IL, "DAGGER", {})
        global_update_size = int(getattr(dagger_cfg, "update_size", 0))
        if global_update_size <= 0:
            return {
                "beta": self.get_beta(iteration),
                "collected_episodes": 0,
                "skipped_episodes": 0,
                "expert_action_counts": {},
                "executed_action_counts": {},
                "policy_action_counts": {},
                "stop_labels": 0,
                "dataset_size": self.count_trajectories(),
            }

        update_size = (global_update_size + self.world_size - 1) // self.world_size

        beta = self.get_beta(iteration)
        policy_module = policy.module if hasattr(policy, "module") else policy
        policy_module.eval()

        expert_action_counts: Counter = Counter()
        executed_action_counts: Counter = Counter()
        policy_action_counts: Counter = Counter()

        collected = 0
        skipped = 0
        attempts = 0
        max_attempts = max(update_size * 10, len(self._episode_indices) * 2)
        # In multi-GPU mode, offset file IDs by rank to avoid name collisions.
        # Each rank writes traj_{rank * large_offset + local_id}.pt so files never overlap.
        _id_offset = self.rank * 10_000_000
        next_id = self.count_trajectories() + _id_offset
        stop_labels = 0

        while collected < update_size and attempts < max_attempts:
            attempts += 1
            episode = self._get_next_episode()
            if episode is None:
                break

            reference_path = getattr(episode, "reference_path", None)
            if not reference_path or len(reference_path) < 2:
                skipped += 1
                continue

            try:
                obs = self.env.reset_to_episode(episode)
            except Exception as exc:
                print(f"Warning: could not reset episode {episode.episode_id}: {exc}")
                skipped += 1
                continue

            obs = self._tokenize_observation(obs)
            self.expert.reset(reference_path)

            rnn_state = policy_module.net.get_initial_state(1, device)
            prev_action = torch.zeros(1, 1, device=device, dtype=torch.long)
            not_done_mask = torch.zeros(1, 1, device=device, dtype=torch.uint8)

            episode_obs: List[Dict[str, torch.Tensor]] = []
            episode_prev_actions: List[int] = []
            episode_teacher_actions: List[int] = []
            errored = False
            done = False
            step_count = 0

            while not done and step_count < self.max_traj_len:
                expert_action_str = self.expert.get_next_action(self.env._task._sim)
                expert_action_idx = Action.get_action_index(expert_action_str)
                expert_action_counts[expert_action_str] += 1

                obs_batch = self._prepare_observation_for_policy(obs, device)
                with torch.no_grad():
                    actions, rnn_state = policy_module.act(
                        obs_batch,
                        rnn_state,
                        prev_action,
                        not_done_mask,
                        deterministic=self.deterministic_policy,
                    )

                policy_action_idx = int(actions[0].item())
                policy_action_str = Action.get_action_from_index(policy_action_idx)
                policy_action_counts[policy_action_str] += 1

                episode_obs.append(self._prepare_observation_for_storage(obs))
                episode_prev_actions.append(int(prev_action.item()))
                episode_teacher_actions.append(expert_action_idx)

                if expert_action_str == Action.STOP:
                    stop_labels += 1
                    executed_action_idx = expert_action_idx
                elif random.random() < beta:
                    executed_action_idx = expert_action_idx
                else:
                    executed_action_idx = policy_action_idx

                executed_action_str = Action.get_action_from_index(executed_action_idx)
                executed_action_counts[executed_action_str] += 1

                try:
                    obs, done, _ = self.env.step(executed_action_idx)
                except Exception as exc:
                    print(
                        f"Warning: DAgger rollout failed for episode {episode.episode_id}: {exc}"
                    )
                    errored = True
                    break

                step_count += 1
                if expert_action_str == Action.STOP:
                    done = True
                    break

                obs = self._tokenize_observation(obs)
                prev_action.fill_(executed_action_idx)
                not_done_mask.fill_(0 if done else 1)

            if errored or len(episode_teacher_actions) == 0:
                skipped += 1
                continue

            self._save_trajectory(
                next_id,
                episode,
                episode_obs,
                episode_prev_actions,
                episode_teacher_actions,
            )
            self._append_manifest_entry(
                self.storage_dir / f"traj_{next_id:08d}.pt",
                len(episode_teacher_actions),
            )
            next_id += 1
            collected += 1

        return {
            "beta": beta,
            "collected_episodes": collected,
            "skipped_episodes": skipped,
            "expert_action_counts": dict(expert_action_counts),
            "executed_action_counts": dict(executed_action_counts),
            "policy_action_counts": dict(policy_action_counts),
            "stop_labels": stop_labels,
            "dataset_size": self.count_trajectories(),
        }


class DaggerTrajectoryDataset(torch.utils.data.IterableDataset):
    """Disk-backed aggregated trajectory dataset for DAgger training."""

    def __init__(
        self,
        storage_dir: str,
        use_inflection_weighting: bool,
        inflection_weight_coef: float,
        batch_size: int = 1,
        preload_size: int = 100,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.storage_dir = Path(storage_dir)
        self.use_inflection_weighting = use_inflection_weighting
        self.batch_size = batch_size
        self.preload_size = max(preload_size, batch_size)
        self.rank = rank
        self.world_size = max(world_size, 1)
        self._preload: List[Tuple[Path, int]] = []
        self.inflec_weights = torch.tensor(
            [1.0, inflection_weight_coef if use_inflection_weighting else 1.0]
        )
        self._entries = self._scan_entries()
        self.length = len(self._entries)
        self.load_ordering: List[int] = []

    def _scan_entries_from_manifest(self) -> List[Tuple[Path, int]]:
        entries: List[Tuple[Path, int]] = []
        manifest_paths = sorted(self.storage_dir.glob("manifest_rank*.tsv"))
        for manifest_path in manifest_paths:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    filename, traj_len = line.split("\t", 1)
                    path = self.storage_dir / filename
                    if not path.exists():
                        continue
                    length = int(traj_len)
                    if length <= 0:
                        continue
                    entries.append((path, length))
        return entries

    def _scan_entries(self) -> List[Tuple[Path, int]]:
        """Scan trajectory files. Reads only the teacher_actions tensor to get length
        without loading the full payload (rgb can be large)."""
        manifest_entries = self._scan_entries_from_manifest()
        if manifest_entries:
            return manifest_entries

        entries: List[Tuple[Path, int]] = []
        for path in sorted(self.storage_dir.glob("traj_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            traj_len = int(payload["teacher_actions"].size(0))
            if traj_len <= 0:
                continue
            entries.append((path, traj_len))
        return entries

    def __len__(self) -> int:
        return self.length

    def _block_shuffle(self, indices: List[int], block_size: int) -> List[int]:
        blocks = [indices[i : i + block_size] for i in range(0, len(indices), block_size)]
        random.shuffle(blocks)
        return [idx for block in blocks for idx in block]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload: List[Tuple[Path, int]] = []
            lengths: List[int] = []
            for _ in range(self.preload_size):
                if len(self.load_ordering) == 0:
                    break
                idx = self.load_ordering.pop()
                path, traj_len = self._entries[idx]
                new_preload.append((path, traj_len))
                lengths.append(traj_len)

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)
            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in self._block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        path, _ = self._preload.pop()
        return torch.load(path, map_location="cpu", weights_only=False)

    def __iter__(self):
        # First shard by DDP rank, then further shard among DataLoader workers.
        all_indices = list(range(self.length))
        rank_indices = all_indices[self.rank :: self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            local_indices = rank_indices
        else:
            per_worker = (len(rank_indices) + worker_info.num_workers - 1) // worker_info.num_workers
            w_start = per_worker * worker_info.id
            local_indices = rank_indices[w_start : w_start + per_worker]

        self.load_ordering = list(
            reversed(
                self._block_shuffle(local_indices, self.preload_size)
            )
        )
        self._preload = []
        return self

    def __next__(self):
        payload = self._load_next()
        obs = payload["observations"]
        prev_actions = payload["prev_actions"]
        teacher_actions = payload["teacher_actions"]

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (teacher_actions[1:] != teacher_actions[:-1]).long(),
            ]
        )
        weights = self.inflec_weights[inflections]

        return obs, prev_actions, teacher_actions, weights

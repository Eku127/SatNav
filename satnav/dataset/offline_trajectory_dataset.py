#!/usr/bin/env python3
"""Offline trajectory dataset backed by pre-rendered trajectory_data."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import DictConfig

from satnav.utils.build_vocab import VocabDict, build_vocab_from_dataset, tokenize


class OfflineTrajectoryDataset(torch.utils.data.Dataset):
    """Dataset for imitation learning on pre-rendered trajectories."""

    def __init__(self, config: DictConfig, target_rgb_size: int = 224):
        super().__init__()
        self.config = config
        self.target_rgb_size = int(
            getattr(getattr(config.IL, "OFFLINE", {}), "rgb_size", target_rgb_size)
        )
        self.max_instruction_len = int(
            getattr(getattr(config.IL, "OFFLINE", {}), "max_instruction_len", 200)
        )
        self.max_traj_len = int(
            getattr(
                getattr(config.IL, "OFFLINE", {}),
                "max_traj_len",
                getattr(config.IL.RECOLLECT_TRAINER, "max_traj_len", 500),
            )
        )

        offline_cfg = config.IL.OFFLINE
        self.annotations_path = Path(offline_cfg.annotations_path)
        self.images_root = Path(offline_cfg.images_root)

        self.vocab = self._load_vocabulary()
        self.annotations = self._load_annotations()

        self.observation_space = {
            "rgb": {
                "shape": (self.target_rgb_size, self.target_rgb_size, 3),
                "dtype": "uint8",
                "description": "Offline RGB image from pre-rendered trajectory data",
            },
            "instruction": {
                "shape": (self.max_instruction_len,),
                "dtype": "int64",
                "description": "Tokenized navigation instruction",
            },
        }
        self.action_space = {
            "actions": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
            "description": "Discrete action space for VLN navigation",
        }

        if getattr(config.IL, "use_inflection_weighting", False):
            coef = float(getattr(config.IL, "inflection_weight_coef", 3.2))
            self.inflec_weights = torch.tensor([1.0, coef], dtype=torch.float32)
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def _load_vocabulary(self) -> VocabDict:
        vocab_file = getattr(self.config.DATASET, "vocab_file", None)
        if vocab_file:
            return VocabDict.load(vocab_file)

        dataset_path = self.config.DATASET.DATA_PATH.format(split=self.config.DATASET.SPLIT)
        return build_vocab_from_dataset(dataset_path)

    def _load_annotations(self) -> List[Dict[str, Any]]:
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            raw_annotations = json.load(f)

        valid_annotations: List[Dict[str, Any]] = []
        for ann in raw_annotations:
            traj_len = self._trajectory_length(ann)
            if traj_len <= 0:
                continue
            if self.max_traj_len > 0 and traj_len > self.max_traj_len:
                continue
            valid_annotations.append(ann)

        if not valid_annotations:
            raise ValueError(
                f"No valid offline trajectories found in {self.annotations_path}"
            )

        return valid_annotations

    def _trajectory_length(self, ann: Dict[str, Any]) -> int:
        _, teacher_actions = self._action_sequences(ann)
        return len(teacher_actions)

    def _action_sequences(self, ann: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        actions = list(ann.get("actions", []))
        if not actions:
            return [], []

        if actions[0] == -1:
            prev_actions = actions[:-1]
            teacher_actions = actions[1:]
        else:
            prev_actions = [-1] + actions[:-1]
            teacher_actions = actions

        # STOP is a supervised terminal decision.  Keep its pre-action
        # observation, but discard any malformed post-STOP suffix (including
        # duplicated STOPs from older archives).  Archives that omit INIT or
        # STOP retain their existing boundary rather than gaining a fabricated
        # action.
        if 0 in teacher_actions:
            terminal_idx = teacher_actions.index(0) + 1
            teacher_actions = teacher_actions[:terminal_idx]
            prev_actions = prev_actions[:terminal_idx]

        return prev_actions, teacher_actions

    def _resolve_frame_dir(self, ann: Dict[str, Any]) -> Path:
        video_path = Path(ann["video"])
        if video_path.parts and video_path.parts[0] == "images":
            video_path = Path(*video_path.parts[1:])
        return self.images_root / video_path / "rgb"

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

    def _load_rgb_sequence(self, ann: Dict[str, Any], traj_len: int) -> torch.Tensor:
        frame_dir = self._resolve_frame_dir(ann)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if len(frame_paths) < traj_len:
            raise ValueError(
                f"Trajectory {ann.get('id')} has only {len(frame_paths)} frames for required length {traj_len}"
            )

        # Every teacher target uses its pre-action observation.  Canonical
        # annotations have actions [-1, ..., STOP] and one frame per action, so
        # this includes the frame before STOP while excluding the saved frame
        # after STOP.
        selected_paths = frame_paths[:traj_len]
        frames = []
        for path in selected_paths:
            with Image.open(path) as img:
                rgb = torch.from_numpy(np.array(img.convert("RGB"), copy=True))
            frames.append(self._resize_rgb(rgb))

        return torch.stack(frames, dim=0)

    def _build_action_tensors(self, ann: Dict[str, Any], traj_len: int) -> Dict[str, torch.Tensor]:
        prev_actions_raw, teacher_actions_raw = self._action_sequences(ann)
        prev_actions = torch.tensor(
            [max(a, 0) for a in prev_actions_raw[:traj_len]], dtype=torch.long
        )
        teacher_actions = torch.tensor(
            teacher_actions_raw[:traj_len], dtype=torch.long
        )
        return {
            "prev_actions": prev_actions,
            "teacher_actions": teacher_actions,
        }

    def _build_weights(self, teacher_actions: torch.Tensor) -> torch.Tensor:
        if torch.allclose(self.inflec_weights, torch.tensor([1.0, 1.0])):
            return torch.ones(len(teacher_actions), dtype=torch.float32)

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (teacher_actions[1:] != teacher_actions[:-1]).long(),
            ]
        )
        return self.inflec_weights[inflections]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        traj_len = self._trajectory_length(ann)

        rgb = self._load_rgb_sequence(ann, traj_len)
        action_tensors = self._build_action_tensors(ann, traj_len)

        instruction_text = ""
        instructions = ann.get("instructions", [])
        if instructions:
            instruction_text = instructions[0]
        tokens = tokenize(instruction_text)
        indices = self.vocab.tokens_to_indices(tokens)
        indices_t = torch.tensor(indices[:self.max_instruction_len], dtype=torch.long)
        if len(indices_t) < self.max_instruction_len:
            pad = torch.zeros(self.max_instruction_len - len(indices_t), dtype=torch.long)
            indices_t = torch.cat([indices_t, pad])
        instruction_tokens = indices_t
        instruction = instruction_tokens.unsqueeze(0).repeat(traj_len, 1)

        weights = self._build_weights(action_tensors["teacher_actions"])
        observations = {
            "rgb": rgb,
            "instruction": instruction,
        }
        return (
            observations,
            action_tensors["prev_actions"],
            action_tensors["teacher_actions"],
            weights,
        )

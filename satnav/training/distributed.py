#!/usr/bin/env python3
"""Utilities for optional distributed training."""

import os
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict


def _select(config, key: str, default=None):
    value = OmegaConf.select(config, key, default=default)
    return default if value is None else value


def setup_distributed(config, run_type: str) -> None:
    """Initialize distributed runtime metadata for the current process.

    Multi-GPU training is enabled only when:
    1. `run_type == "train"`
    2. `DISTRIBUTED.enabled == true`
    3. the process is launched with `WORLD_SIZE > 1` (for example via torchrun)
    """
    user_enabled = bool(_select(config, "DISTRIBUTED.enabled", False))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested_multi_gpu = run_type == "train" and user_enabled and world_size > 1

    if requested_multi_gpu and not torch.cuda.is_available():
        raise RuntimeError("Distributed multi-GPU training requires CUDA.")

    if requested_multi_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        backend = _select(config, "DISTRIBUTED.backend", "nccl")

        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        device_id = local_rank
    else:
        device_id = int(_select(config, "TORCH_GPU_ID", 0))
        local_rank = device_id
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

    with open_dict(config):
        if "DISTRIBUTED" not in config:
            config.DISTRIBUTED = OmegaConf.create({})
        config.DISTRIBUTED.runtime_enabled = requested_multi_gpu
        config.DISTRIBUTED.rank = rank
        config.DISTRIBUTED.world_size = world_size
        config.DISTRIBUTED.local_rank = local_rank
        config.TORCH_GPU_ID = device_id


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed_runtime(config) -> bool:
    """Return whether the current process is in DDP runtime."""
    return bool(_select(config, "DISTRIBUTED.runtime_enabled", False))


def get_rank(config) -> int:
    """Return the current global rank."""
    return int(_select(config, "DISTRIBUTED.rank", 0))


def get_world_size(config) -> int:
    """Return the current world size."""
    return int(_select(config, "DISTRIBUTED.world_size", 1))


def get_local_rank(config) -> int:
    """Return the current local rank."""
    return int(_select(config, "DISTRIBUTED.local_rank", _select(config, "TORCH_GPU_ID", 0)))


def is_main_process(config) -> bool:
    """Return whether the current process is rank 0."""
    return get_rank(config) == 0


def barrier(config) -> None:
    """Synchronize all ranks when running with DDP."""
    if is_distributed_runtime(config) and dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_tensor(config, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce a tensor across ranks and optionally average it."""
    if not is_distributed_runtime(config):
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= get_world_size(config)
    return reduced


def reduce_scalar(config, value: float, device: Optional[torch.device] = None, average: bool = True) -> float:
    """All-reduce a scalar value across ranks and optionally average it."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", int(_select(config, "TORCH_GPU_ID", 0)))
        else:
            device = torch.device("cpu")

    tensor = torch.tensor(float(value), device=device)
    reduced = reduce_tensor(config, tensor, average=average)
    return float(reduced.item())

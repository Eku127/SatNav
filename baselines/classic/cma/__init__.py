"""CMA baseline compatibility exports.

Importing this package intentionally opts into the classic PyTorch extra.
Existing ``satnav.models.baselines.cma_policy`` imports remain supported so old
checkpoints and scripts retain their state-dict names.
"""

from satnav.models.baselines.cma_policy import CMANet, CMAPolicy
from baselines.classic.cma.factory import build_cma_adapter

__all__ = ["CMANet", "CMAPolicy", "build_cma_adapter"]

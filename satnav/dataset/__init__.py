"""Dataset loading for SatNav.

Keep package imports lightweight.

Training-oriented datasets import PyTorch and are exposed lazily so importing
the core episode loader never loads baseline dependencies.
"""

from satnav.dataset.satnav_dataset import SatNavDataset
from satnav.dataset.scene_resolver import SceneResolver

__all__ = [
    "SatNavDataset",
    "SceneResolver",
    "RecollectionDataset",
    "OfflineTrajectoryDataset",
]


def __getattr__(name):
    if name == "OfflineTrajectoryDataset":
        from satnav.dataset.offline_trajectory_dataset import OfflineTrajectoryDataset

        return OfflineTrajectoryDataset
    if name == "RecollectionDataset":
        from satnav.dataset.recollect_dataset import RecollectionDataset

        return RecollectionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

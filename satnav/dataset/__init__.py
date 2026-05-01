"""Dataset loading for SatNav.

Keep package imports lightweight.

`RecollectionDataset` pulls in the training package, which in turn imports the
evaluator and environment stack. Import it lazily here to avoid circular import
failures when evaluation code only needs `SatNavDataset`.
"""

from satnav.dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
from satnav.dataset.satnav_dataset import SatNavDataset

__all__ = [
    "SatNavDataset",
    "RecollectionDataset",
    "OfflineTrajectoryDataset",
]


def __getattr__(name):
    if name == "RecollectionDataset":
        from satnav.dataset.recollect_dataset import RecollectionDataset

        return RecollectionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

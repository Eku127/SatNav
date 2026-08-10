"""Classic SatNav baseline entry points.

Non-learning adapters are dependency-light.  Neural policies are imported only
from their model-specific subpackages so importing this module does not import
PyTorch.
"""

from baselines.classic.agents import RandomAdapter, ReferenceFollowerAdapter
from baselines.classic.factory import build_classic_adapter

__all__ = [
    "RandomAdapter",
    "ReferenceFollowerAdapter",
    "build_classic_adapter",
]

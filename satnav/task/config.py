"""Public task-configuration helpers shared by applications and examples."""

from numbers import Real
from typing import Any, Mapping, Optional


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def get_episode_success_distance(
    config: Any,
    trajectory_type: Optional[str] = None,
    default: float = 10.0,
) -> float:
    """Resolve the configured success distance for an episode type.

    Both the legacy scalar form and the current ``DEFAULT`` / trajectory-type
    mapping are supported.
    """
    task_config = _get(config, "TASK")
    if task_config is None:
        return float(default)

    success_distance = _get(task_config, "SUCCESS_DISTANCE", default)
    if isinstance(success_distance, Real):
        return float(success_distance)

    if trajectory_type:
        typed_value = _get(success_distance, trajectory_type)
        if typed_value is not None:
            return float(typed_value)

    return float(_get(success_distance, "DEFAULT", default))

"""Classic CLI configuration loading with repository-relative inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(
    value: Union[str, Path], *, relative_to: Optional[Path] = None
) -> Path:
    """Resolve a public relative path without embedding a machine path."""

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if relative_to is not None:
        candidates.append(relative_to / path)
    candidates.extend((Path.cwd() / path, REPOSITORY_ROOT / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the repository-relative spelling for useful error messages.
    return (REPOSITORY_ROOT / path).resolve()


def _load_with_base(path: Path, seen: Optional[Iterable[Path]] = None) -> DictConfig:
    path = path.resolve()
    ancestors = set(seen or ())
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"cyclic config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    current = OmegaConf.load(path)
    base_value = OmegaConf.select(current, "_base_", default=None)
    if base_value is None:
        base_value = OmegaConf.select(current, "BASE_CONFIG_PATH", default=None)
    if base_value is None:
        return current
    base_path = resolve_path(str(base_value), relative_to=path.parent)
    base = _load_with_base(base_path, (*ancestors, path))
    current_dict = OmegaConf.to_container(current, resolve=False)
    current_dict.pop("_base_", None)
    current_dict.pop("BASE_CONFIG_PATH", None)
    return OmegaConf.merge(base, OmegaConf.create(current_dict))


def _override_config(overrides: Sequence[str]) -> DictConfig:
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"config override must use KEY=VALUE syntax, got {item!r}"
            )
    return OmegaConf.from_dotlist(list(overrides))


def load_classic_config(
    config_path: Union[str, Path], *, overrides: Sequence[str] = ()
) -> DictConfig:
    """Load base + experiment + task config, then apply CLI overrides.

    Overrides are inspected before loading ``BASE_TASK_CONFIG_PATH`` so a
    canonical run can replace an example task config from the command line.
    The same overrides are applied last and remain authoritative.
    """

    path = resolve_path(config_path)
    experiment = _load_with_base(path)
    override = _override_config(overrides)
    task_selector = OmegaConf.merge(experiment, override)
    task_value = OmegaConf.select(
        task_selector, "BASE_TASK_CONFIG_PATH", default=None
    )
    if task_value is not None:
        task_path = resolve_path(str(task_value), relative_to=path.parent)
        if not task_path.is_file():
            raise FileNotFoundError(f"task config does not exist: {task_path}")
        task = OmegaConf.load(task_path)
        config = OmegaConf.merge(task, experiment, override)
    else:
        config = OmegaConf.merge(experiment, override)
    return config

"""OpenFly source provenance and lazy local class registration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


OPENFLY_UPSTREAM = "https://github.com/SHAILAB-IPEC/OpenFly-Platform.git"
OPENFLY_REVISION = "c075075497a7122bad82f5b76b9be926ad5a81b3"
SWIFTVLN_ADAPTER_REVISION = "7b996303b05ca62791a2e5bf8f5fda729d6fdc07"


class OpenFlySourceError(RuntimeError):
    pass


def source_revision(repo: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def source_is_dirty(repo: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(repo)),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def validate_optional_upstream(value: Optional[os.PathLike] = None) -> Optional[Path]:
    """Validate an optional comparison checkout; runtime never imports it."""

    configured = value or os.environ.get("OPENFLY_PLATFORM_REPO")
    if not configured:
        return None
    repo = Path(configured).expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise OpenFlySourceError(f"not an OpenFly-Platform Git checkout: {repo}")
    revision = source_revision(repo)
    if revision != OPENFLY_REVISION:
        raise OpenFlySourceError(
            f"OpenFly upstream mismatch: expected {OPENFLY_REVISION}, found {revision}"
        )
    if source_is_dirty(repo):
        raise OpenFlySourceError("OpenFly comparison checkout is dirty")
    return repo


def register_openfly_classes() -> None:
    """Register the bundled adapted classes only on a model code path."""

    from baselines.vlm.openfly.openfly_core import register_openfly_auto_classes

    register_openfly_auto_classes()

"""Resolve the pinned external NaVILA checkout without cwd assumptions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


NAVILA_UPSTREAM = "https://github.com/AnjieCheng/NaVILA.git"
NAVILA_REVISION = "76b98f233dd0fff05dfcd69435eec6740febff9d"
SWIFTVLN_ADAPTER_REVISION = "7b996303b05ca62791a2e5bf8f5fda729d6fdc07"

BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]


class NaVILASourceError(RuntimeError):
    """Raised when the configured upstream checkout is incomplete or unpinned."""


def resolve_navila_repo(value: Optional[os.PathLike] = None) -> Path:
    """Resolve and validate an external NaVILA source checkout."""

    configured = value or os.environ.get("NAVILA_REPO")
    repo = (
        Path(configured).expanduser() if configured else SATNAV_ROOT.parent / "NaVILA"
    )
    repo = repo.resolve()
    required = (
        repo / "pyproject.toml",
        repo / "llava" / "model" / "builder.py",
        repo / "llava" / "train" / "train.py",
        repo / "llava" / "data" / "dataset.py",
    )
    missing = [path.relative_to(repo) for path in required if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise NaVILASourceError(
            f"Not a usable NaVILA checkout: {repo}; missing {rendered}"
        )
    return repo


def source_revision(repo: os.PathLike) -> Optional[str]:
    """Return a Git checkout revision, or ``None`` for a source archive."""

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
    revision = result.stdout.strip()
    return revision or None


def source_is_dirty(repo: os.PathLike) -> bool:
    """Return whether a Git checkout has tracked or untracked changes."""

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


def bootstrap_navila(
    value: Optional[os.PathLike] = None,
    *,
    require_pinned_revision: bool = True,
) -> Path:
    """Validate the source revision and prepend its root to ``sys.path``."""

    repo = resolve_navila_repo(value)
    revision = source_revision(repo)
    if require_pinned_revision and revision != NAVILA_REVISION:
        rendered = revision or "unknown (source archive or non-Git checkout)"
        raise NaVILASourceError(
            "NaVILA revision mismatch: "
            f"expected {NAVILA_REVISION}, found {rendered}. "
            "Use --allow-upstream-mismatch only for deliberate development runs."
        )
    if require_pinned_revision and source_is_dirty(repo):
        raise NaVILASourceError(
            "NaVILA checkout is dirty; pinned runs require an unmodified source tree. "
            "Use --allow-upstream-mismatch only for deliberate development runs."
        )
    rendered = str(repo)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)
    return repo

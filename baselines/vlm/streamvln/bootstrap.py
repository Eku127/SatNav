"""Resolve the pinned external StreamVLN source tree without cwd assumptions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


STREAMVLN_UPSTREAM = "https://github.com/InternRobotics/StreamVLN.git"
STREAMVLN_REVISION = "60476e81f4c01b29f1a51a7469f1cb4addbc1d62"
SATNAV_ADAPTER_UPSTREAM = "https://github.com/Eku127/SatNav.git"
SATNAV_ADAPTER_REVISION = "c0c0e72ea4575b36d74a5e8f777942172978938e"

BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]


class StreamVLNSourceError(RuntimeError):
    """Raised when the external pinned source tree cannot be used."""


def resolve_streamvln_repo(value: Optional[os.PathLike] = None) -> Path:
    """Return a validated StreamVLN checkout path.

    Resolution order is an explicit argument, ``STREAMVLN_REPO``, then a
    sibling ``StreamVLN`` checkout next to SatNav.  Unlike the legacy adapter,
    the fallback never depends on the process working directory.
    """

    configured = value or os.environ.get("STREAMVLN_REPO")
    repo = Path(configured).expanduser() if configured else SATNAV_ROOT.parent / "StreamVLN"
    repo = repo.resolve()
    required = (
        repo / "streamvln" / "streamvln_train.py",
        repo / "streamvln" / "model" / "stream_video_vln.py",
        repo / "streamvln" / "dataset" / "vln_action_dataset.py",
    )
    missing = [path.relative_to(repo) for path in required if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise StreamVLNSourceError(
            f"Not a usable StreamVLN checkout: {repo}; missing {rendered}"
        )
    return repo


def source_revision(repo: os.PathLike) -> Optional[str]:
    """Return the checkout revision, or ``None`` for a source archive."""

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


def bootstrap_streamvln(
    value: Optional[os.PathLike] = None,
    *,
    require_pinned_revision: bool = True,
) -> Path:
    """Add canonical and legacy upstream import roots to ``sys.path``.

    The pinned official revision contains both canonical imports such as
    ``streamvln.model`` and legacy imports such as ``model``/``utils``.  Its
    repository root and nested ``streamvln`` directory are therefore both
    required.  They are resolved once from an explicit checkout, not from
    ``../StreamVLN`` relative to an arbitrary launcher directory.
    """

    repo = resolve_streamvln_repo(value)
    revision = source_revision(repo)
    if require_pinned_revision and revision != STREAMVLN_REVISION:
        rendered = revision or "unknown (source archive or non-Git checkout)"
        raise StreamVLNSourceError(
            "StreamVLN revision mismatch: "
            f"expected {STREAMVLN_REVISION}, found {rendered}. "
            "Pass --allow-upstream-mismatch only for deliberate development runs."
        )

    for import_root in (repo, repo / "streamvln"):
        rendered = str(import_root)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    return repo

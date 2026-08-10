"""Resolve the pinned external Uni-NaVid checkout without cwd assumptions."""

from __future__ import annotations

import importlib.machinery
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional


UNINAVID_UPSTREAM = "https://github.com/jzhzhang/Uni-NaVid.git"
UNINAVID_REVISION = "79ef5ea3fea14c205342d1ab070563d84c7a966a"
SWIFTVLN_ADAPTER_REVISION = "7b996303b05ca62791a2e5bf8f5fda729d6fdc07"

BASELINE_DIR = Path(__file__).resolve().parent
SATNAV_ROOT = BASELINE_DIR.parents[2]


class UniNaVidSourceError(RuntimeError):
    """Raised when the configured upstream checkout is incomplete or unpinned."""


def resolve_uninavid_repo(value: Optional[os.PathLike] = None) -> Path:
    configured = value or os.environ.get("UNINAVID_REPO")
    repo = (
        Path(configured).expanduser()
        if configured
        else SATNAV_ROOT.parent / "Uni-NaVid"
    ).resolve()
    required = (
        repo / "pyproject.toml",
        repo / "uninavid" / "model" / "builder.py",
        repo / "uninavid" / "model" / "uninavid_arch.py",
        repo / "uninavid" / "train" / "train.py",
        repo / "uninavid" / "processor" / "clip-patch14-224" / "preprocessor_config.json",
    )
    missing = [path.relative_to(repo) for path in required if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise UniNaVidSourceError(
            f"Not a usable Uni-NaVid checkout: {repo}; missing {rendered}"
        )
    return repo


def source_revision(repo: os.PathLike) -> Optional[str]:
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


def source_is_dirty(repo: os.PathLike) -> bool:
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


def bootstrap_uninavid(
    value: Optional[os.PathLike] = None,
    *,
    require_pinned_revision: bool = True,
) -> Path:
    repo = resolve_uninavid_repo(value)
    revision = source_revision(repo)
    if require_pinned_revision and revision != UNINAVID_REVISION:
        raise UniNaVidSourceError(
            "Uni-NaVid revision mismatch: expected "
            f"{UNINAVID_REVISION}, found {revision or 'unknown'}"
        )
    if require_pinned_revision and source_is_dirty(repo):
        raise UniNaVidSourceError(
            "Uni-NaVid checkout is dirty; accepted runs require a clean pinned tree"
        )
    rendered = str(repo)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)
    return repo


def install_image_dataset_decord_stub() -> None:
    """Satisfy an unused upstream MP4 import for SatNav's JPEG-only loader.

    The only pinned upstream reference to decord is a module-level import in
    ``train.py``. SatNav replaces its data module and never constructs a
    ``VideoReader``. The public PyPI decord wheel is tagged for CPython 3.6 and
    fails ``pip check`` under Python 3.9, so accepted environments deliberately
    do not install it.
    """

    if "decord" in sys.modules:
        return
    module = types.ModuleType("decord")
    module.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
    module.__version__ = "0.0-satnav-jpeg-stub"

    class UnsupportedVideoReader:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("SatNav Uni-NaVid training accepts JPEG frames only")

    module.VideoReader = UnsupportedVideoReader
    module.cpu = lambda index=0: index
    sys.modules["decord"] = module

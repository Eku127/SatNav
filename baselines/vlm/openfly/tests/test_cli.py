from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


BASELINE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["trainer", "evaluate", "checkpoint", "dataset"])
def test_module_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"baselines.vlm.openfly.{module}", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script", ["bootstrap_env.sh", "validate_data.sh", "train.sh", "eval.sh"]
)
def test_shell_syntax(script: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(BASELINE / "scripts" / script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

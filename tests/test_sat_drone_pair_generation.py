"""Tests for the sat-drone pair generation launcher."""

from __future__ import annotations

import importlib
import sys

import pytest

from applications.sat_drone_pair_generation.main import main as launcher_main
from applications.sat_drone_pair_generation.registry import (
    get_registry,
    normalize_dataset_name,
    resolve_module,
)


def iter_registry_entries():
    """Yield all dataset-command pairs in the registry."""
    for dataset, commands in get_registry().items():
        for command in commands:
            yield dataset, command


@pytest.mark.parametrize(("dataset", "command"), list(iter_registry_entries()))
def test_registered_modules_import(dataset: str, command: str) -> None:
    module = importlib.import_module(resolve_module(dataset, command))
    assert hasattr(module, "main")


def test_normalize_dataset_name_supports_aliases() -> None:
    assert normalize_dataset_name("gta-uav") == "gta_uav"
    assert normalize_dataset_name("UAV-VisLoc") == "uavvisloc"


def test_launcher_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(resolve_module("denseuav", "build_pairs"))
    captured = {}

    def fake_main() -> None:
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(module, "main", fake_main)

    exit_code = launcher_main(["denseuav", "build_pairs", "--help"])

    assert exit_code == 0
    assert captured["argv"] == ["denseuav build_pairs", "--help"]


def test_launcher_reads_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = importlib.import_module(resolve_module("denseuav", "build_pairs"))
    captured = {}

    def fake_main() -> None:
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(module, "main", fake_main)

    config_path = tmp_path / "pair_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "DATASETS:",
                "  denseuav:",
                "    build_pairs:",
                "      input:",
                "        dataset_root: /data/denseuav",
                "      output:",
                "        output_dir: /output/denseuav",
                "      args:",
                "        workers: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = launcher_main(["--config", str(config_path), "denseuav", "build_pairs"])

    assert exit_code == 0
    assert captured["argv"] == [
        "denseuav build_pairs",
        "--dataset-root",
        "/data/denseuav",
        "--output-dir",
        "/output/denseuav",
        "--workers",
        "4",
    ]


def test_launcher_cli_overrides_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = importlib.import_module(resolve_module("denseuav", "build_pairs"))
    captured = {}

    def fake_main() -> None:
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(module, "main", fake_main)

    config_path = tmp_path / "pair_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "DATASETS:",
                "  denseuav:",
                "    build_pairs:",
                "      input:",
                "        dataset_root: /data/denseuav",
                "      output:",
                "        output_dir: /output/from_config",
                "      args:",
                "        workers: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = launcher_main(
        [
            "--config",
            str(config_path),
            "denseuav",
            "build_pairs",
            "--output-dir",
            "/output/from_cli",
            "--workers",
            "8",
        ]
    )

    assert exit_code == 0
    assert captured["argv"] == [
        "denseuav build_pairs",
        "--dataset-root",
        "/data/denseuav",
        "--output-dir",
        "/output/from_cli",
        "--workers",
        "8",
    ]


def test_launcher_rejects_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = launcher_main(["denseuav", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Unknown command" in captured.err
    assert "Supported dataset commands" in captured.err

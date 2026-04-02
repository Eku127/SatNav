"""Unified launcher for sat-drone pair generation scripts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from .registry import get_registry, normalize_dataset_name, resolve_module

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _format_supported_commands() -> str:
    lines = ["Supported dataset commands:"]
    for dataset, commands in get_registry().items():
        joined = ", ".join(sorted(commands))
        lines.append(f"  - {dataset}: {joined}")
    lines.append("")
    lines.append("Dataset aliases: gta-uav -> gta_uav, uav-visloc -> uavvisloc")
    return "\n".join(lines)


def _usage() -> str:
    return (
        "Usage:\n"
        "  python -m applications.sat_drone_pair_generation "
        "[--config path/to/config.yaml] <dataset> <command> [args...]\n\n"
        f"Default config path: {DEFAULT_CONFIG_PATH}\n\n"
        f"{_format_supported_commands()}"
    )


def run_command(dataset: str, command: str, forwarded_args: Sequence[str]) -> int:
    """Run a registered dataset command by forwarding argv to its main()."""
    module_path = resolve_module(dataset, command)
    module = importlib.import_module(module_path)
    module_main = getattr(module, "main", None)
    if module_main is None:
        raise AttributeError(f"Module {module_path} does not define main()")

    original_argv = list(sys.argv)
    dataset_key = normalize_dataset_name(dataset)
    try:
        sys.argv = [f"{dataset_key} {command}"] + list(forwarded_args)
        module_main()
    finally:
        sys.argv = original_argv
    return 0


def _flag_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _is_flag_overridden(flag: str, forwarded_args: Sequence[str]) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in forwarded_args)


def _append_cli_value(parts: List[str], key: str, value: Any) -> None:
    if value is None or value is False:
        return

    flag = _flag_name(key)
    if isinstance(value, bool):
        parts.append(flag)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            return
        parts.append(flag)
        parts.extend(str(item) for item in value)
        return

    parts.extend([flag, str(value)])


def _extract_global_options(args: Sequence[str]) -> Tuple[Path, List[str]]:
    config_path = DEFAULT_CONFIG_PATH
    remaining = list(args)

    while remaining:
        current = remaining[0]
        if current == "--config":
            if len(remaining) < 2:
                raise ValueError("--config requires a path")
            config_path = Path(remaining[1])
            remaining = remaining[2:]
            continue
        if current.startswith("--config="):
            config_path = Path(current.split("=", 1)[1])
            remaining = remaining[1:]
            continue
        break

    return config_path, remaining


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def _get_command_config(
    config: Mapping[str, Any],
    dataset: str,
    command: str,
) -> Mapping[str, Any]:
    datasets = config.get("DATASETS") or config.get("datasets") or {}
    if not isinstance(datasets, Mapping):
        raise ValueError("Config field DATASETS must be a mapping")

    normalized = {normalize_dataset_name(name): value for name, value in datasets.items()}
    dataset_cfg = normalized.get(normalize_dataset_name(dataset), {})
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError(f"Dataset config for {dataset} must be a mapping")

    command_cfg = dataset_cfg.get(command, {})
    if not isinstance(command_cfg, Mapping):
        raise ValueError(f"Command config for {dataset}.{command} must be a mapping")
    return command_cfg


def _build_config_args(command_cfg: Mapping[str, Any], forwarded_args: Sequence[str]) -> List[str]:
    if any(arg in {"-h", "--help"} for arg in forwarded_args):
        return []

    generated: List[str] = []
    for section_name in ("input", "output", "args"):
        section = command_cfg.get(section_name) or command_cfg.get(section_name.upper()) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"Config section {section_name} must be a mapping")
        for key, value in section.items():
            flag = _flag_name(key)
            if _is_flag_overridden(flag, forwarded_args):
                continue
            _append_cli_value(generated, key, value)
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the sat-drone pair launcher."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    try:
        config_path, args = _extract_global_options(raw_args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if not args or args[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0

    if len(args) < 2:
        print("Expected <dataset> and <command>.", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    dataset, command = args[0], args[1]
    forwarded_args: List[str] = args[2:]

    try:
        config = _load_config(config_path)
        config_args = _build_config_args(_get_command_config(config, dataset, command), forwarded_args)
        return run_command(dataset, command, config_args + forwarded_args)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

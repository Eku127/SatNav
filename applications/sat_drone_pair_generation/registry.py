"""Static registry for sat-drone pair generation commands."""

from __future__ import annotations

from typing import Dict, Mapping


DATASET_COMMANDS: Dict[str, Dict[str, str]] = {
    "denseuav": {
        "build_pairs": "applications.sat_drone_pair_generation.denseuav.build_pairs",
        "sample_preview": "applications.sat_drone_pair_generation.denseuav.sample_preview",
    },
    "gta_uav": {
        "build_pairs": "applications.sat_drone_pair_generation.gta_uav.build_pairs",
        "sample_preview": "applications.sat_drone_pair_generation.gta_uav.sample_preview",
    },
    "sues": {
        "build_pairs": "applications.sat_drone_pair_generation.sues.build_pairs",
        "pipeline": "applications.sat_drone_pair_generation.sues.pipeline",
        "center_recrop_pairs": "applications.sat_drone_pair_generation.sues.center_recrop_pairs",
        "sample_preview": "applications.sat_drone_pair_generation.sues.sample_preview",
        "merge_variants_dense_style": (
            "applications.sat_drone_pair_generation.sues.merge_variants_dense_style"
        ),
    },
    "uavvisloc": {
        "build_pairs": "applications.sat_drone_pair_generation.uavvisloc.build_pairs",
        "export_selected": "applications.sat_drone_pair_generation.uavvisloc.export_selected",
        "center_recrop_pairs": "applications.sat_drone_pair_generation.uavvisloc.center_recrop_pairs",
        "sample_preview": "applications.sat_drone_pair_generation.uavvisloc.sample_preview",
    },
}

DATASET_ALIASES = {
    "denseuav": "denseuav",
    "gta_uav": "gta_uav",
    "gta-uav": "gta_uav",
    "sues": "sues",
    "uavvisloc": "uavvisloc",
    "uav-visloc": "uavvisloc",
}


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names accepted by the launcher."""
    key = name.strip().lower()
    return DATASET_ALIASES.get(key, key.replace("-", "_"))


def get_registry() -> Mapping[str, Mapping[str, str]]:
    """Return the command registry."""
    return DATASET_COMMANDS


def resolve_module(dataset: str, command: str) -> str:
    """Resolve dataset + command into a Python module path."""
    dataset_key = normalize_dataset_name(dataset)
    commands = DATASET_COMMANDS.get(dataset_key)
    if commands is None:
        raise KeyError(f"Unknown dataset: {dataset}")

    module_path = commands.get(command)
    if module_path is None:
        raise KeyError(f"Unknown command for {dataset_key}: {command}")

    return module_path

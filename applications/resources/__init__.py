"""Runtime-resolved access to SatNav's public example resources."""

from pathlib import Path

from omegaconf import OmegaConf


def load_example_task_config():
    """Load the bundled task config with source-relative resource paths."""

    resource_root = Path(__file__).resolve().parent
    config = OmegaConf.load(resource_root / "satnav_example_task.yaml")
    config.DATASET.DATA_PATH = str(resource_root / "satnav_example_episodes.json")
    config.DATASET.SCENES_DIR = str(resource_root)
    return config


__all__ = ["load_example_task_config"]

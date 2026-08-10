"""Import-boundary tests for the dependency-light SatNav public API."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CoreImportTests(unittest.TestCase):
    def test_env_and_episode_loader_do_not_import_training_or_torch(self):
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(repository_root), environment.get("PYTHONPATH", "")],
            )
        )
        code = """
import json
import sys
from satnav.core import Env
from satnav.dataset import SatNavDataset, SceneResolver
print(json.dumps({
    'torch': 'torch' in sys.modules,
    'training': 'satnav.training' in sys.modules,
    'offline_dataset': 'satnav.dataset.offline_trajectory_dataset' in sys.modules,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=str(repository_root),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {"torch": False, "training": False, "offline_dataset": False},
        )

    def test_dataset_package_has_no_training_import(self):
        dataset_root = Path(__file__).resolve().parents[1] / "satnav" / "dataset"
        offenders = []
        for source_path in dataset_root.glob("*.py"):
            if "satnav.training" in source_path.read_text(encoding="utf-8"):
                offenders.append(source_path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

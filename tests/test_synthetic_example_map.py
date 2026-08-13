"""Provenance and reproducibility checks for the bundled synthetic map."""

from __future__ import annotations

import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from affine import Affine
import numpy as np
import rasterio

from applications.resources import load_example_task_config
from satnav.core.env import Env


class SyntheticExampleMapTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    map_path = repository_root / "applications" / "resources" / "map.tif"
    generator = repository_root / "scripts" / "generate_synthetic_example_map.py"
    expected_transform = Affine(
        0.5971642834776122,
        0.0,
        12696898.933590123,
        0.0,
        -0.5971642834780654,
        2577851.0207853806,
    )

    def test_metadata_declares_synthetic_cc0_provenance(self):
        self.assertEqual(stat.S_IMODE(self.map_path.stat().st_mode), 0o644)
        with rasterio.open(self.map_path) as source:
            self.assertEqual((source.width, source.height), (3441, 3169))
            self.assertEqual(source.count, 3)
            self.assertEqual(source.dtypes, ("uint8", "uint8", "uint8"))
            self.assertEqual(source.crs.to_string(), "EPSG:3857")
            self.assertEqual(source.transform, self.expected_transform)
            tags = source.tags()
        self.assertEqual(tags["SATNAV_ASSET_KIND"], "procedural-synthetic-example")
        self.assertEqual(tags["SATNAV_LICENSE"], "CC0-1.0")
        self.assertEqual(
            tags["SATNAV_GENERATOR"],
            "scripts/generate_synthetic_example_map.py",
        )
        self.assertIn("no third-party", tags["SATNAV_PROVENANCE"].lower())

    def test_generator_reproduces_the_tracked_raster_pixel_for_pixel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = Path(temporary_directory) / "map.tif"
            subprocess.run(
                [sys.executable, str(self.generator), "--output", str(generated)],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(stat.S_IMODE(generated.stat().st_mode), 0o644)
            with rasterio.open(self.map_path) as tracked_source:
                with rasterio.open(generated) as generated_source:
                    self.assertEqual(generated_source.crs, tracked_source.crs)
                    self.assertEqual(
                        generated_source.transform, tracked_source.transform
                    )
                    self.assertEqual(generated_source.tags(), tracked_source.tags())
                    for _, window in tracked_source.block_windows():
                        np.testing.assert_array_equal(
                            generated_source.read(window=window),
                            tracked_source.read(window=window),
                        )

    def test_resource_helper_resets_the_example_environment(self):
        config = load_example_task_config()
        self.assertTrue(Path(config.DATASET.DATA_PATH).is_file())
        self.assertTrue(Path(config.DATASET.SCENES_DIR).is_dir())
        environment = Env(config)
        try:
            observation = environment.reset()
            self.assertEqual(observation["rgb"].shape, (224, 224, 3))
            self.assertEqual(len(environment.episodes), 2)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()

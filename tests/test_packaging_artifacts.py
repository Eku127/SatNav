"""Build/install regression tests for SatNav distribution artifacts."""

from contextlib import contextmanager
import os
from pathlib import Path
import stat
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unittest
import zipfile


@contextmanager
def _temporary_umask(mask):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


class PackagingArtifactTests(unittest.TestCase):
    local_canary = "SATNAV_PACKAGING_" + "LOCAL_CANARY_" + "DO_NOT_SHIP"
    nested_probe_stem = "deep_" + "canary"
    forbidden_directories = {
        ".local",
        "checkpoints",
        "logs",
        "output",
        "rebuttal",
        "results",
        "runs",
        "runtime",
        "wandb",
    }

    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temporary_directory.name)
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.source_root = cls.temp_root / "source"
        cls.dist_root = cls.temp_root / "dist"
        cls.dist_root.mkdir()

        cls.source_root.mkdir()
        candidate_output = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=str(cls.repository_root),
            check=True,
            capture_output=True,
        ).stdout
        candidates = [
            Path(raw.decode("utf-8"))
            for raw in candidate_output.split(b"\0")
            if raw
        ]
        for relative in candidates:
            if "rebuttal" in relative.parts:
                continue
            source = cls.repository_root / relative
            # ``git ls-files --cached`` also reports tracked files deleted in
            # the current worktree. Distribution checks must model the files
            # that actually exist in the candidate tree.
            if not source.exists() and not source.is_symlink():
                continue
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"non-regular release candidate: {relative}")
            destination = cls.source_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        canary_paths = (
            cls.source_root / ".local" / "env.sh",
            cls.source_root / "baselines" / "classic" / ".local" / "env.sh",
            cls.source_root / "baselines" / "vlm" / "streamvln" / ".local" / "env.sh",
            cls.source_root / "baselines" / "vlm" / "navila" / ".local" / "env.sh",
            cls.source_root / "baselines" / "vlm" / "uninavid" / ".local" / "env.sh",
            cls.source_root / "baselines" / "vlm" / "openfly" / ".local" / "env.sh",
            cls.source_root / "scripts" / "seq2seq" / ".local" / "env.sh",
            cls.source_root / "scripts" / "cma" / ".local" / "env.sh",
            cls.source_root / "configs" / "local_packaging_canary.yaml",
            cls.source_root
            / "baselines"
            / "vlm"
            / "streamvln"
            / "results"
            / "canary.json",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / ".local"
            / "deeper"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "foo"
            / "output"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / "runtime"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "resources"
            / "private_release_canary.json",
            cls.source_root
            / "applications"
            / "resources"
            / "private_release_canary.tif",
            cls.source_root
            / "baselines"
            / "vlm"
            / "openfly"
            / "configs"
            / "private_release_canary.json",
            cls.source_root
            / "applications"
            / "resources"
            / "runtime"
            / "__init__.py",
            cls.source_root
            / "applications"
            / "resources"
            / "runtime"
            / "secret.py",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / "checkpoints"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / "logs"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / "runs"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "applications"
            / "resources"
            / "nested"
            / "wandb"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "baselines"
            / "classic"
            / "runtime"
            / f"{cls.nested_probe_stem}.json",
            cls.source_root
            / "scripts"
            / "deep"
            / "component"
            / ".local"
            / "deeper"
            / f"{cls.nested_probe_stem}.py",
        )
        for canary_path in canary_paths:
            canary_path.parent.mkdir(parents=True, exist_ok=True)
            canary_path.write_text(cls.local_canary, encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for distribution_command in ("sdist", "bdist_wheel"):
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "setup.py",
                    distribution_command,
                    "--dist-dir",
                    str(cls.dist_root),
                ],
                cwd=str(cls.source_root),
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        cls.sdist_path = next(cls.dist_root.glob("*.tar.gz"))
        cls.wheel_path = next(cls.dist_root.glob("*.whl"))
        cls.rebuilt_dist_root = cls.temp_root / "rebuilt-dist"
        cls.rebuilt_dist_root.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(cls.rebuilt_dist_root),
                str(cls.sdist_path),
            ],
            cwd=str(cls.temp_root),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.rebuilt_wheel_path = next(cls.rebuilt_dist_root.glob("*.whl"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_sdist_contains_public_runtime_resources(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()
            members = archive.getmembers()
            map_member = next(
                member
                for member in members
                if member.name.endswith("/applications/resources/map.tif")
            )
        self.assertTrue(any(name.endswith("/pyproject.toml") for name in names))
        self.assertTrue(
            any(name.endswith("/docs/MODEL_INTEGRATION.md") for name in names)
        )
        self.assertTrue(
            any(
                name.endswith(
                    "/satnav/utils/assets/maps_topdown_agent_sprite/100x100.png"
                )
                for name in names
            )
        )
        self.assertTrue(
            any(name.endswith("/applications/resources/map.tif") for name in names)
        )
        self.assertTrue(
            any(name.endswith("/applications/resources/README.md") for name in names)
        )
        self.assertTrue(
            any(name.endswith("/baselines/vlm/openfly/configs/zero2.json") for name in names)
        )
        self.assertTrue(
            any(name.endswith("/baselines/vlm/uninavid/configs/zero1.json") for name in names)
        )
        self.assertEqual(stat.S_IMODE(map_member.mode), 0o644)
        for member in members:
            if member.isdir():
                self.assertEqual(stat.S_IMODE(member.mode), 0o755, member.name)
            elif member.isfile():
                self.assertEqual(stat.S_IMODE(member.mode), 0o644, member.name)

    def test_wheel_contains_resources_and_dependency_extras(self):
        with zipfile.ZipFile(self.wheel_path) as archive:
            names = archive.namelist()
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = archive.read(metadata_name).decode("utf-8")

        self.assertIn(
            "satnav/utils/assets/maps_topdown_agent_sprite/100x100.png", names
        )
        self.assertIn("applications/resources/map.tif", names)
        self.assertIn("applications/resources/README.md", names)
        self.assertIn("baselines/vlm/openfly/configs/zero2.json", names)
        self.assertIn("baselines/vlm/uninavid/configs/zero1.json", names)
        for wheel_path in (self.wheel_path, self.rebuilt_wheel_path):
            with zipfile.ZipFile(wheel_path) as archive:
                map_info = archive.getinfo("applications/resources/map.tif")
                for info in archive.infolist():
                    if not info.is_dir():
                        mode = stat.S_IMODE(info.external_attr >> 16)
                        self.assertEqual(mode & 0o444, 0o444, (wheel_path, info.filename))
            self.assertEqual(
                stat.S_IMODE(map_info.external_attr >> 16),
                0o644,
                wheel_path,
            )
        self.assertTrue(
            any(
                Path(name).name == "DATA_LICENSE.md"
                and any(part.endswith(".dist-info") for part in Path(name).parts)
                for name in names
            )
        )
        self.assertIn("License-File: DATA_LICENSE.md", metadata)
        self.assertIn("Provides-Extra: core", metadata)
        self.assertIn("Provides-Extra: classic", metadata)
        self.assertIn("Provides-Extra: vlm", metadata)
        imageio_requirements = [
            line
            for line in metadata.splitlines()
            if line.lower().startswith("requires-dist: imageio")
        ]
        self.assertTrue(imageio_requirements)
        self.assertTrue(
            any("extra ==" not in requirement for requirement in imageio_requirements)
        )
        torch_requirements = [
            line
            for line in metadata.splitlines()
            if line.lower().startswith("requires-dist: torch")
        ]
        self.assertTrue(torch_requirements)
        self.assertTrue(all("extra ==" in line for line in torch_requirements))

    def test_distribution_artifacts_exclude_local_state(self):
        canary = self.local_canary.encode("utf-8")
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                self.assertTrue(
                    self.forbidden_directories.isdisjoint(parts), member.name
                )
                self.assertFalse(
                    Path(member.name).name.startswith("local_packaging_canary"),
                    member.name,
                )
                if member.isfile():
                    extracted = archive.extractfile(member)
                    self.assertIsNotNone(extracted)
                    content = extracted.read()
                    self.assertNotIn(canary, content, member.name)
                    self.assertNotIn(
                        self.nested_probe_stem.encode("ascii"), content, member.name
                    )

        for wheel_path in (self.wheel_path, self.rebuilt_wheel_path):
            with zipfile.ZipFile(wheel_path) as archive:
                for name in archive.namelist():
                    self.assertTrue(
                        self.forbidden_directories.isdisjoint(Path(name).parts), name
                    )
                    self.assertFalse(
                        Path(name).name.startswith("local_packaging_canary"), name
                    )
                    content = archive.read(name)
                    self.assertNotIn(canary, content, name)
                    self.assertNotIn(
                        self.nested_probe_stem.encode("ascii"), content, name
                    )

    def test_wheel_installs_import_core_without_torch(self):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        code = """
from pathlib import Path
import satnav
import applications
import sys
from applications.resources import load_example_task_config
from satnav.core import Env
root = Path(satnav.__file__).resolve().parent
assert str(root).startswith(sys.argv[1])
assert (root / 'utils/assets/maps_topdown_agent_sprite/100x100.png').is_file()
resource_root = Path(applications.__file__).resolve().parent / 'resources'
assert str(resource_root).startswith(sys.argv[1])
config = load_example_task_config()
assert (resource_root / 'map.tif').stat().st_mode & 0o444 == 0o444
environment = Env(config)
try:
    observation = environment.reset()
    assert observation['rgb'].shape == (224, 224, 3)
    assert len(environment.episodes) == 2
finally:
    environment.close()
assert 'torch' not in sys.modules
"""
        for index, wheel_path in enumerate((self.wheel_path, self.rebuilt_wheel_path)):
            target = self.temp_root / f"wheel-target-{index}"
            with _temporary_umask(0o022):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--target",
                        str(target),
                        str(wheel_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            environment["PYTHONPATH"] = str(target)
            subprocess.run(
                [sys.executable, "-B", "-c", code, str(target)],
                cwd=str(self.temp_root),
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_editable_install_imports_the_selected_checkout(self):
        editable_target = self.temp_root / "editable-target"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(self.source_root),
                "--target",
                str(editable_target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        code = """
from pathlib import Path
import site
import sys
site.addsitedir(sys.argv[2])
# ``-S`` prevents the test interpreter from loading the developer checkout's
# existing editable .pth.  Add dependency packages as a plain path only after
# installing the selected checkout's editable finder.
sys.path.append(sys.argv[3])
import satnav
from satnav.core import Env
resolved = Path(satnav.__file__).resolve()
assert str(resolved).startswith(sys.argv[1]), resolved
assert 'torch' not in sys.modules
"""
        subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                "-c",
                code,
                str(self.source_root),
                str(editable_target),
                sysconfig.get_paths()["purelib"],
            ],
            cwd=str(self.temp_root),
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()

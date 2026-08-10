"""Regression tests for the release-tree hygiene gate."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ReleaseHygieneTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    source_script = repository_root / "scripts" / "check_release_hygiene.sh"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(
            self.source_script, self.root / "scripts" / self.source_script.name
        )
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _track(self, path: str, content: str) -> Path:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", path], cwd=self.root, check=True)
        return destination

    def _run(self, **environment_overrides: str):
        environment = os.environ.copy()
        environment.update(environment_overrides)
        return subprocess.run(
            ["bash", "scripts/check_release_hygiene.sh"],
            cwd=self.root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_clean_tree_passes(self):
        self._track("README.md", "public release\n")
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[OK]", result.stdout)

    def test_new_private_path_and_token_shapes_are_rejected_and_redacted(self):
        probes = (
            "/" + "data3/user/private/model.bin",
            "/" + "srv/internal/checkpoints/model.bin",
            "github" + "_pat_" + "A" * 32,
            "sk-" + "proj-" + "B" * 32,
        )
        for index, probe in enumerate(probes):
            with self.subTest(probe_index=index):
                path = f"probe_{index}.txt"
                self._track(path, f"prefix {probe} suffix\n")
                result = self._run()
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{path}:1", result.stderr)
                self.assertNotIn(probe, result.stderr)
                subprocess.run(
                    ["git", "rm", "-q", "-f", "--", path],
                    cwd=self.root,
                    check=True,
                )

    def test_ignored_overlay_is_silent_normally_and_blocked_in_release_mode(self):
        self._track(".gitignore", ".local/\n")
        local_marker = "IGNORED_LOCAL_MARKER_MUST_NOT_BE_ECHOED"
        local_file = self.root / ".local" / "env.sh"
        local_file.parent.mkdir()
        local_file.write_text(local_marker, encoding="utf-8")

        normal = self._run()
        self.assertEqual(normal.returncode, 0, normal.stderr)
        self.assertNotIn(local_marker, normal.stdout + normal.stderr)

        strict = self._run(SATNAV_RELEASE_TREE="1")
        self.assertEqual(strict.returncode, 1)
        self.assertIn("1 ignored local-only file(s)", strict.stderr)
        self.assertNotIn(local_marker, strict.stdout + strict.stderr)
        self.assertNotIn("env.sh", strict.stdout + strict.stderr)

    def test_symlink_escape_is_rejected(self):
        outside = self.root.parent / f"{self.root.name}-outside.txt"
        outside.write_text("sk-" + "proj-" + "Z" * 32 + "\n", encoding="utf-8")
        try:
            (self.root / "escape.txt").symlink_to(outside)
            subprocess.run(
                ["git", "add", "--", "escape.txt"], cwd=self.root, check=True
            )
            result = self._run()
            self.assertEqual(result.returncode, 1)
            self.assertIn("symlink points outside repository", result.stderr)
            self.assertNotIn("escape.txt:1", result.stderr)
        finally:
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

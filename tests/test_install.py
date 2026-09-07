"""Check installation orchestration without installing or downloading anything."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallationTests(unittest.TestCase):
    def run_install(self, *, existing=False, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            shutil.copyfile(ROOT / "scripts/install.sh", root / "scripts/install.sh")
            log = root / "commands.log"
            uv = root / "bin/uv"
            uv.write_text('#!/bin/bash\nprintf "uv %s\\n" "$*" >> "$INSTALL_TEST_LOG"\nexit "${INSTALL_TEST_FAIL:-0}"\n')
            uv.chmod(0o700)
            prepare = root / "scripts/prepare_models.sh"
            prepare.write_text('#!/bin/bash\nprintf "models %s\\n" "$*" >> "$INSTALL_TEST_LOG"\n'
                               'test "$PYTHON" = "$PWD/.envs/spatialmqa/bin/python"\n'
                               'test "$ASSET_ROOT" = "$PWD/.cache/assets"\n')
            prepare.chmod(0o700)
            if existing:
                for name in ("spatialmqa", "molmoact", "qwen36"):
                    python = root / f".envs/{name}/bin/python"
                    python.parent.mkdir(parents=True)
                    python.write_text("unchanged")
                    python.chmod(0o700)
            result = subprocess.run(
                ["bash", str(root / "scripts/install.sh")], cwd=tmp,
                env=dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}",
                         INSTALL_TEST_LOG=str(log), INSTALL_TEST_FAIL="7" if fail else "0"),
                text=True, capture_output=True,
            )
            if existing:
                for name in ("spatialmqa", "molmoact", "qwen36"):
                    self.assertEqual((root / f".envs/{name}/bin/python").read_text(), "unchanged")
            return result, log.read_text().splitlines()

    def test_fresh_install_prepares_all_required_model_groups(self):
        result, calls = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 8)
        self.assertTrue(calls[0].startswith("uv venv --python 3.10 "))
        self.assertTrue(calls[2].startswith("uv venv --python 3.11 "))
        self.assertTrue(calls[4].startswith("uv venv --python 3.11 "))
        for index, requirements in ((1, "requirements-molmo2.txt"), (3, "requirements-molmoact2.txt"), (5, "requirements-qwen.txt")):
            self.assertTrue(calls[index].endswith(f"-r {requirements}"))
        self.assertEqual(calls[-2:], ["models all", "models action_supervision"])

    def test_existing_environments_are_not_recreated(self):
        result, calls = self.run_install(existing=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 5)
        self.assertFalse(any("uv venv" in call for call in calls))

    def test_dependency_failure_stops_before_model_download(self):
        result, calls = self.run_install(existing=True, fail=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("uv pip install"))

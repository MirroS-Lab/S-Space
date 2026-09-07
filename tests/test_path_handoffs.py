"""CPU regressions for fresh-checkout data and CLI path handoffs."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sspace.experiments.analysis.contextual_cases.scripts import prepare_mmsi_eightway as mmsi
from sspace.experiments.analysis.object_coordinate_lens.cli import _worker_python
from sspace.experiments.analysis.object_coordinate_lens.inference import _local_asset_base
from sspace.experiments.multi_view.hstar import hstar_multiview as hstar


ROOT = Path(__file__).resolve().parents[1]


class HStarHandoffTests(unittest.TestCase):
    def test_rebuilt_published_rows_reuse_audit_without_fabricating_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            pairs.write_text('{"image_a_sha256":"original"}\n')
            digest = hashlib.sha256(pairs.read_bytes()).hexdigest()
            manifest = {"dataset_fingerprint": digest, "review": {"status": "pending"}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            checksums = {"pairs.jsonl": digest}
            with patch.object(hstar, "PUBLISHED_PAIRS_SHA256", digest):
                self.assertIsNone(hstar._validated_review_ids(path, manifest, checksums))
                self.assertEqual(manifest["review"], {"status": "pending"})
                pairs.write_text('{"image_a_sha256":"changed"}\n')
                with self.assertRaisesRegex(ValueError, "visual review"):
                    hstar._validated_review_ids(path, manifest, checksums)

    def test_unpublished_data_still_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "visual review"):
                hstar._validated_review_ids(path, {"dataset_fingerprint": "changed"}, {})


class MMSIHandoffTests(unittest.TestCase):
    def setUp(self):
        self.reference = [json.loads(line) for line in mmsi.REFERENCE_MANIFEST.read_text().splitlines()]

    def test_exact_reference_accepted(self):
        self.assertEqual(len(self.reference), 191)
        mmsi.validate_reference(self.reference, skip_images=False)

    def test_changed_question_label_image_and_selection_rejected(self):
        for key in ("prompt", "answer_letter", "image_sha256"):
            rows = copy.deepcopy(self.reference)
            rows[0][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "191-question"):
                mmsi.validate_reference(rows, skip_images=False)
        with self.assertRaises(ValueError):
            mmsi.validate_reference(self.reference[:-1], skip_images=False)

    def test_skip_images_still_checks_questions(self):
        rows = [{**r, "image_sha256": [""] * len(r["image_paths"])} for r in self.reference]
        mmsi.validate_reference(rows, skip_images=True)

    def test_download_pins_revision_and_checks_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "source.parquet"
            parquet.write_bytes(b"verified-source")
            digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
            with patch("huggingface_hub.hf_hub_download", return_value=str(parquet)) as download:
                with patch.object(mmsi, "SOURCE_SHA256", digest):
                    self.assertEqual(mmsi.source_parquet(), parquet)
                self.assertEqual(download.call_args.kwargs["revision"], mmsi.SOURCE_REVISION)
                self.assertEqual(download.call_args.kwargs["repo_type"], "dataset")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    mmsi.source_parquet()


class LauncherPathTests(unittest.TestCase):
    def test_prepare_dry_run_in_either_argument_position(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "python"
            fake.write_text(
                "#!/bin/bash\n"
                '[[ "$*" == *sspace.data.assets_cli* && "$*" == *--dry-run* ]] || exit 42\n'
                'printf "%s\\n" "$*"\n'
            )
            fake.chmod(0o700)
            for args in (["--dry-run", "--workers", "1"], ["--workers", "1", "--dry-run"]):
                with self.subTest(args=args):
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts/prepare_data.sh"), *args],
                        cwd=directory, env={**os.environ, "PYTHON": str(fake)},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(sum(line.startswith("-m ") for line in result.stdout.splitlines()), 6)

    def test_lens_python_override_preserves_environment_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.symlink_to(sys.executable)
            self.assertEqual(_worker_python("molmo2_er", python), python)
            self.assertNotEqual(python, python.resolve())

    def test_lens_cli_relative_assets_and_web_paths(self):
        self.assertEqual(_local_asset_base("."), ".")
        self.assertEqual(_local_asset_base("/runs/example/"), "/runs/example")
        for value in ("..", "//remote", "https://remote"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _local_asset_base(value)


if __name__ == "__main__":
    unittest.main()

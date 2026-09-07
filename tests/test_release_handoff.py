"""CPU regressions for portable artifacts and the CoT score-to-plot boundary."""

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sspace.core.artifacts import SSpaceArtifact
from sspace.experiments.analysis import plot_quantitative
from sspace.experiments.spinbench.evolving_cot.scripts import (
    score_prefilled_spinbench_cot_stages as cot_score,
)
from sspace.run_records import atomic_json


ROOT = Path(__file__).resolve().parents[1]


class PortableArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = SSpaceArtifact.load(
            ROOT / "sspace/core/artifacts/pretrained/"
            "molmo2_er_coco6000_final_logit_axes_v2"
        )

    def with_construction(self, **updates):
        manifest = replace(
            self.artifact.manifest,
            construction={**self.artifact.manifest.construction, **updates},
        )
        return replace(self.artifact, manifest=manifest)

    def test_local_manifest_hash_is_provenance_not_dataset_identity(self):
        manifest = {"source": {"annotation_path": "/another-checkout/instances.json"}}
        digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        self.assertNotEqual(
            digest, self.artifact.manifest.construction["dataset_manifest_sha256"]
        )
        self.with_construction(dataset_manifest_sha256=digest).validate()

    def test_changed_dataset_identity_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
            self.with_construction(dataset_fingerprint="0" * 64).validate()

    def test_invalid_manifest_digest_remains_rejected(self):
        for digest in (None, "", "g" * 64, "a" * 63):
            with self.subTest(digest=digest), self.assertRaisesRegex(
                ValueError, "dataset_manifest_sha256"
            ):
                self.with_construction(dataset_manifest_sha256=digest).validate()


class CotPlotHandoffTests(unittest.TestCase):
    def test_real_scorer_summary_can_be_plotted(self):
        row = {"target_view": "back", "target_property": "closer"}
        for stage in cot_score.STAGES:
            for mapping in cot_score.MAPPINGS:
                row[f"{stage}_{mapping}_correct"] = True
                row[f"{stage}_{mapping}_gold_aligned_margin"] = 1.0
        summary = cot_score._summary([row])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "summary.json"
            output = root / "plots"
            atomic_json(source, summary)
            with patch.object(sys, "argv", [
                "plot_quantitative", "--protocol", "cot", "--inputs",
                str(source), "--output-dir", str(output),
            ]):
                plot_quantitative.main()
            for name in ("cot.png", "cot.svg", "plot_source.csv", "provenance.json"):
                self.assertTrue((output / name).is_file(), name)
            table = pd.read_csv(output / "plot_source.csv")
            self.assertEqual(len(table), 6)
            self.assertTrue(table.accuracy.eq(1.0).all())


if __name__ == "__main__":
    unittest.main()

"""Keep the released CoT text identical to final historical experiment records."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sspace.experiments.spinbench.evolving_cot.template import render_prefilled_cot
from sspace.experiments.spinbench.evolving_cot.scripts import score_prefilled_spinbench_cot_stages as scoring


class CotTemplateTests(unittest.TestCase):
    def test_final_historical_text_for_all_transforms_and_answer_orders(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/cot_natural_template.json").read_text())
        self.assertEqual(len(fixture["cases"]), 12)
        for case in fixture["cases"]:
            with self.subTest(sample=case["sample_id"]):
                self.assertEqual(render_prefilled_cot(SimpleNamespace(**case["sample"])), case["rationale"])

    def test_scoring_rejects_old_template_before_loading_records(self):
        artifact = SimpleNamespace(manifest=SimpleNamespace(model=SimpleNamespace(selected_layer=43)), axes=lambda layer: None)
        with patch.object(Path, "exists", return_value=False), patch.object(scoring.SSpaceArtifact, "load", return_value=artifact), patch.object(Path, "read_text", return_value='{"sample_count":146}'), patch.object(scoring, "_read_jsonl") as read:
            with self.assertRaisesRegex(ValueError, "final native_exact_winner"):
                scoring.main([])
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

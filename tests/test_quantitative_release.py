"""CPU checks for migrated quantitative readouts and their plot sources."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from sspace.experiments.analysis.contextual_cases.token_matching import find_direction_occurrences
from sspace.experiments.analysis.contextual_cases.scripts.plot_qwen_mmsi_direction_centroids import summarize
from sspace.experiments.analysis.plot_quantitative import retention_table
from sspace.experiments.intervention.spatial_causality.cli import summarize_layers
from sspace.experiments.intervention.spatial_causality.spatial_jacobian import dataset as causality_dataset
from sspace.experiments.spinbench.perspective_taking.tools import run_spinbench_layerwise_sweep as sweep
from sspace.run_records import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class QuantitativeReleaseTests(unittest.TestCase):
    def test_released_model_loaders_select_sdpa(self):
        from sspace.core.models.runtime import RUNTIME_SPECS
        from sspace.experiments.analysis.object_coordinate_lens.models import _MODEL_SPECS

        self.assertEqual({spec.attention_backend for spec in RUNTIME_SPECS.values()}, {"sdpa"})
        self.assertEqual({spec.attention_implementation for spec in _MODEL_SPECS.values()}, {"sdpa"})

    def test_pair_loader_does_not_swallow_implementation_errors(self):
        frame = pd.DataFrame([dict(category="left", question="test", index=0)])
        with patch.object(causality_dataset.pd, "read_csv", return_value=frame), patch.object(
            causality_dataset, "extract_objects", side_effect=RuntimeError("unexpected bug")
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
                causality_dataset.load_swap_pairs(Path("unused.tsv"), seed=42)

    def test_reference_integrity_and_recomputed_centroids(self):
        root = ROOT / "sspace/data/contextual_cases/mmsi_eightway"
        manifest = json.loads((root / "reference_manifest.json").read_text())
        self.assertEqual(file_sha256(root / "manifest.jsonl"), manifest["manifest"]["sha256"])
        self.assertEqual(file_sha256(root / "selection.json"), manifest["selection_sha256"])
        for name, digest in manifest["reference_outputs"].items():
            self.assertEqual(file_sha256(root / "reference" / name), digest)
        artifact = ROOT / "sspace/core/artifacts/pretrained" / manifest["artifact_id"]
        self.assertEqual(file_sha256(artifact / "tensors.safetensors"), manifest["artifact_tensors_sha256"])
        tokens = pd.read_csv(root / "reference/direction_token_projections.csv")
        self.assertEqual(len(tokens), 1942)
        self.assertEqual(tokens.mmsi_id.nunique(), 160)
        actual = summarize(tokens).set_index("word").sort_index()
        expected = pd.read_csv(root / "reference/direction_centroids_up_positive.csv").set_index("word").sort_index()
        self.assertEqual(int(actual.occurrence_count.sum()), 1311)
        np.testing.assert_allclose(actual, expected[actual.columns], rtol=1e-12, atol=1e-12)

    def test_compound_direction_tokens_are_not_counted_twice(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": list(map(ord, text))}

            def decode(self, ids, **kwargs):
                return "".join(map(chr, ids))

        result = find_direction_occurrences(Tokenizer(), list(map(ord, "north west; south-east; left")))
        self.assertEqual([r["word"] for r in result], ["northwest", "southeast", "left"])

    def test_retention_uses_predictions_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary_all.csv"
            row = dict(experiment="relative_swap", source_layer=17, parameter="alpha", value=1,
                       scope="overall", n=10, changed_count=3, changed_rate=.3, target_opposite_rate=.9)
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertAlmostEqual(retention_table(path).answer_retention.iloc[0], .7)
            row["changed_rate"] = .4
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                retention_table(path)

    def test_retention_reads_strength_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_summary.csv"
            pd.DataFrame([
                dict(experiment="absolute_shift", source_layer=17, edit_scale=6,
                     scope="overall", n=800, effect_count=472, effect_rate=.59),
                dict(experiment="relative_swap", source_layer=17, edit_scale=6,
                     scope="overall", n=800, effect_count=449, effect_rate=.56125),
            ]).to_csv(path, index=False)
            actual = retention_table(path).set_index("experiment")
            self.assertAlmostEqual(actual.loc["absolute_shift", "answer_retention"], .41)
            self.assertAlmostEqual(actual.loc["relative_swap", "answer_retention"], .43875)

    def test_shift_directions_share_one_edit_scale(self):
        frames = []
        for value, changes in ((-5, (True, False)), (0, (False, False)), (5, (False, True))):
            frames.append(pd.DataFrame({
                "experiment": ["absolute_shift"] * 2,
                "source_layer": [17] * 2,
                "value": [value] * 2,
                "case_index": [0, 1],
                "group": ["horizontal", "vertical"],
                "changed_vs_baseline": changes,
            }))
        overall = summarize_layers(frames)
        overall = overall[overall.scope.eq("overall")].set_index("edit_scale")
        self.assertEqual(overall.index.tolist(), [0.0, 1.0])
        self.assertEqual(overall.loc[1.0, "values"], "[-5.0, 5.0]")
        self.assertEqual(overall.loc[1.0, "effect_count"], 2)

    def test_each_hidden_layer_uses_its_matching_axis(self):
        artifact = SimpleNamespace(
            manifest=SimpleNamespace(model=SimpleNamespace(layer_ids=(0, 2))),
            axes_tensor=np.array([np.eye(3), np.eye(3)[[1, 0, 2]]], dtype=np.float32),
        )
        runtime = SimpleNamespace(spec=SimpleNamespace(hidden_size=3), blocks=[None] * 3, model=None)
        sample = SimpleNamespace(sample_id="test", premise_mode="without", source_group="horizontal",
                                 target_view="back", target_property="left", object_a="a", object_b="b", answer="A")
        config = SimpleNamespace(probe_context="none", checkpoint_every=1, model_adapter="fixture", premise_mode="without")
        difference = torch.tensor([[[1., 2., 3.], [10., 20., 30.], [4., 5., 6.]],
                                   [[-1., -2., -3.], [-10., -20., -30.], [-4., -5., -6.]]])
        prompts = (SimpleNamespace(text="a b"), SimpleNamespace(text="b a"))
        with tempfile.TemporaryDirectory() as tmp, patch.object(sweep, "render_spinbench_projection_prompts", return_value=prompts), \
             patch.object(sweep, "open_spinbench_image"), patch.object(sweep, "LayerwisePostBlockStates", return_value=MagicMock()), \
             patch.object(sweep, "_extract_prompt_pair_differences", return_value=(difference, (), ())):
            margins, _ = sweep._extract_layerwise_margins(runtime, artifact, [sample], config, Path(tmp), {"test": True})
            np.testing.assert_array_equal(margins, [[[1, 5], [-1, -5]]])

    def test_layerwise_csv_allows_groups_absent_from_a_smoke_subset(self):
        summary = {
            "overall": {"margin": {"accuracy": 1.0}},
            "by_source_group": {},
            "by_target_view": {"back": {"margin": {"accuracy": 1.0}}},
            "by_target_property": {"closer": {"margin": {"accuracy": 1.0}}},
            "by_transform": {},
        }
        record = SimpleNamespace(to_dict=lambda: {"sample_id": "test"})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(sweep, "score_spinbench_projection", return_value=[record]), \
             patch.object(sweep, "summarize_spinbench_projection", return_value=summary):
            result = sweep._score_layers(
                [SimpleNamespace()], np.zeros((1, 2, 1)), [SimpleNamespace()],
                [0], "smoke", Path(tmp)
            )
            row = pd.read_csv(Path(tmp) / "layerwise_metrics.csv").iloc[0]
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(row.back_accuracy, 1.0)
            self.assertTrue(pd.isna(row.left_view_accuracy))
            self.assertTrue(pd.isna(row.right_view_accuracy))
            self.assertTrue(pd.isna(row.left_property_accuracy))


if __name__ == "__main__":
    unittest.main()

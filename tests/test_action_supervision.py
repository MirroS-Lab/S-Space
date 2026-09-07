"""CPU-only contracts for the InstructPart action-supervision experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sspace.data.instructpart.acquisition import select_rows
from sspace.experiments.action_supervision.instructpart.config import (
    DEFAULT_CONFIG,
    MODEL_ORDER,
    load_config,
)
from sspace.experiments.action_supervision.instructpart.prompts import (
    PROMPT_TEMPLATES,
    render_prompt_ensemble,
)
from sspace.experiments.action_supervision.instructpart.scoring import (
    _prompt_summary,
    signed_correct,
)


class PromptProtocolTests(unittest.TestCase):
    def test_exact_ten_templates_and_separate_forwards(self) -> None:
        pairs = render_prompt_ensemble("open", "bottle", "cap")
        self.assertEqual(len(pairs), 10)
        self.assertEqual(len(PROMPT_TEMPLATES), 10)
        object_prompt, part_prompt = pairs[5]
        self.assertEqual(
            object_prompt.text,
            'Given the instruction "open the bottle", locate the bottle.',
        )
        self.assertEqual(
            part_prompt.text,
            'Given the instruction "open the bottle", locate the cap.',
        )
        for prompt, target in ((object_prompt, "bottle"), (part_prompt, "cap")):
            start, end = prompt.object_spans["target"]
            self.assertEqual(prompt.text[start:end], target)

    def test_zero_margin_is_not_a_directional_prediction(self) -> None:
        margins = np.asarray([-1.0, 0.0, 1.0, 0.0])
        ground_truth = np.asarray([-1.0, -1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            signed_correct(margins, ground_truth),
            np.asarray([True, False, True, False]),
        )


class DataAndConfigTests(unittest.TestCase):
    def test_scorer_emits_prompt_summary_and_preserves_paired_comparisons(self) -> None:
        from sspace.experiments.action_supervision.instructpart import scoring
        from sspace.experiments.action_supervision.instructpart.prompts import PROMPT_PROTOCOL

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{
                "showcase_id": 1, "score_candidate": True,
                "horizontal_delta": 1., "vertical_delta": 1., "depth_gap_normalized": 1.,
                "horizontal_label": "right", "vertical_label": "below", "depth_label": "close",
            }]
            gt = root / "ground_truth.json"
            gt.write_text(json.dumps({"rows": rows, "review_status": "pending"}))
            templates = [{"template_id": item.template_id, "group": item.group, "text": item.text}
                         for item in PROMPT_TEMPLATES]
            for model in MODEL_ORDER:
                directory = root / model
                directory.mkdir()
                (directory / "summary.json").write_text("{}")
                (directory / "resolved_config.json").write_text(json.dumps({
                    "protocol": PROMPT_PROTOCOL, "templates": templates, "layer_ids": [18],
                }))
                (directory / "records.jsonl").write_text('{"showcase_id":1,"coordinate_index":0}\n')
                np.save(directory / "template_part_minus_object.npy", np.ones((1, 10, 1, 3)))
            output = root / "results.json"
            with patch.object(scoring, "_comparison", return_value={"paired": True}):
                scoring.score(gt, root, output, [18])
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "provisional")
            for model in MODEL_ORDER:
                layer = result["model_results"][model]["requested_layers"][0]
                self.assertEqual(layer["across_prompt_summary"]["overall"]["ci95"], [1., 1.])
                self.assertEqual(len(layer["per_template"]), 10)
            self.assertEqual(result["paired_comparisons_vs_molmo2_er"]["molmoact2"]["18"], {"paired": True})

    def test_prompt_intervals_match_published_depth_bars(self) -> None:
        # Historical counts out of 225 valid depth labels, in T01--T10 order.
        cases = [
            ([144, 137, 138, 139, 142, 146, 144, 148, 143, 138], 63.0666666667, [61.8728165363, 64.2605168687]),
            ([150, 152, 151, 152, 151, 150, 146, 155, 156, 147], 67.1111111111, [66.1283041040, 68.0939181481]),
            ([157, 154, 154, 155, 155, 157, 153, 159, 157, 152], 69.0222222222, [68.3345840091, 69.7098605487]),
        ]
        for counts, mean, interval in cases:
            rows = [
                {"overall_accuracy": count / 225, "per_axis": [
                    {"axis": axis, "accuracy": count / 225}
                    for axis in ("horizontal", "vertical", "distance")
                ]}
                for count in counts
            ]
            result = _prompt_summary(rows)
            depth = result["per_axis"][2]
            self.assertEqual(result["degrees_of_freedom"], 9)
            self.assertEqual(result["template_count"], 10)
            self.assertAlmostEqual(depth["mean_accuracy"] * 100, mean, places=8)
            np.testing.assert_allclose(np.array(depth["ci95"]) * 100, interval, rtol=0, atol=1e-7)

    def test_identical_prompts_have_zero_interval_width(self) -> None:
        rows = [{"overall_accuracy": .75, "per_axis": [
            {"axis": axis, "accuracy": .75}
            for axis in ("horizontal", "vertical", "distance")
        ]}] * 10
        result = _prompt_summary(rows)
        self.assertEqual(result["overall"]["ci95"], [.75, .75])
        self.assertEqual(result["overall"]["standard_error"], 0.)

    def test_group_round_robin_is_deterministic(self) -> None:
        rows = [
            {
                "row": {
                    "question_id": question_id,
                    "object": object_name,
                    "part": "handle",
                    "affordance": "hold",
                }
            }
            for question_id, object_name in (
                (1, "cup"),
                (2, "cup"),
                (3, "bag"),
                (4, "bag"),
            )
        ]
        first = select_rows(rows, 4, 42)
        second = select_rows(list(reversed(rows)), 4, 42)
        self.assertEqual(
            [row["row"]["question_id"] for row in first],
            [row["row"]["question_id"] for row in second],
        )

    def test_release_config_declares_three_models(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["classification"], "quantitative")
        self.assertEqual(
            set(config["evaluation"]["models"]),
            set(MODEL_ORDER),
        )
        self.assertEqual(config["evaluation"]["layers"], list(range(17, 23)))


if __name__ == "__main__":
    unittest.main()

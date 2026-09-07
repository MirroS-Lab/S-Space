"""CPU regression checks for declared Beyond-the-Visible contrasts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sspace.experiments.analysis.contextual_cases.schema import (
    load_case_manifest,
)
from sspace.experiments.analysis.contextual_cases.scoring import (
    summarize_beyond_visible,
    summarize_beyond_visible_conditions,
    summarize_web_political,
)
from sspace.experiments.analysis.contextual_cases.scripts.build_manifest import (
    build_paired_prompts,
    build_web_showcase,
    build_political_templates,
)

ROOT = Path(__file__).resolve().parents[1]


def ball_frame(*, blank: bool = True) -> pd.DataFrame:
    rows = []
    for prompt, values in enumerate(((-3, 1, 100), (-5, 3, -100))):
        for condition, target, reference in zip(
            ("original", "mirror", "blank"), values, (1, 2, 0), strict=True
        ):
            if condition == "blank" and not blank:
                continue
            rows.append(
                {
                    "theme": "beyond_visible",
                    "family": "offscreen_ball",
                    "condition": condition,
                    "meta_prompt_key": f"ball_{prompt}",
                    "meta_effect_sign": -1,
                    "target_horizontal": float(target),
                    "pair_horizontal": float(target - reference),
                }
            )
    return pd.DataFrame(rows)


def train_frame() -> pd.DataFrame:
    rows = []
    for prompt, values in enumerate(((2, 4, 10), (4, 6, 12))):
        for condition, target, reference in zip(
            ("original", "mirror", "blank"), values, (3, 1, 1), strict=True
        ):
            rows.append(
                {
                    "theme": "beyond_visible",
                    "family": "offscreen_train",
                    "condition": condition,
                    "meta_prompt_key": f"train_{prompt}",
                    "meta_effect_sign": 1,
                    "meta_effect_axis": "distance",
                    "meta_effect_conditions_a": ["blank"],
                    "meta_effect_conditions_b": ["original", "mirror"],
                    # Deliberately unrelated H values expose an incorrect axis choice.
                    "target_horizontal": float(1000 - target * 100),
                    "pair_horizontal": float(2000 - target * 300),
                    "target_distance": float(target),
                    "pair_distance": float(target - reference),
                }
            )
    return pd.DataFrame(rows)


class BeyondVisibleTests(unittest.TestCase):
    def test_legacy_mirror_effect_and_paired_interval(self) -> None:
        summary, points = summarize_beyond_visible(ball_frame(blank=False), draws=1000)
        raw = summary.set_index("readout").loc["raw_target"]
        self.assertEqual(raw.expected_effect_mean, 6.0)
        self.assertEqual((raw.ci95_low, raw.ci95_high), (4.0, 8.0))
        self.assertEqual(raw.n_prompts, 2)
        np.testing.assert_array_equal(
            points[points.readout.eq("target_minus_reference")].effect, [3.0, 7.0]
        )

    def test_blank_is_reported_without_changing_mirror_contrast(self) -> None:
        expected = summarize_beyond_visible(ball_frame(blank=False), draws=1000)
        actual = summarize_beyond_visible(ball_frame(), draws=1000)
        for left, right in zip(expected, actual, strict=True):
            pd.testing.assert_frame_equal(left, right)
        conditions = summarize_beyond_visible_conditions(ball_frame())
        raw = conditions[conditions.readout.eq("raw_target")].set_index("condition")
        self.assertEqual(raw.loc["blank", "mean"], 0.0)
        self.assertEqual(raw.loc["original", "mean"], -4.0)
        self.assertEqual(raw.loc["mirror", "mean"], 2.0)

    def test_train_uses_distance_and_within_prompt_condition_mean(self) -> None:
        summary, points = summarize_beyond_visible(train_frame(), draws=1000)
        raw = summary.set_index("readout").loc["raw_target"]
        self.assertEqual(raw.effect_axis, "distance")
        self.assertEqual(raw.conditions_a, "blank")
        self.assertEqual(raw.conditions_b, "original+mirror")
        self.assertEqual(
            (raw.expected_effect_mean, raw.ci95_low, raw.ci95_high), (7, 7, 7)
        )
        np.testing.assert_array_equal(
            points[points.readout.eq("target_minus_reference")].effect, [8.0, 8.0]
        )
        conditions = summarize_beyond_visible_conditions(train_frame())
        raw_conditions = conditions[conditions.readout.eq("raw_target")].set_index(
            "condition"
        )
        self.assertEqual(raw_conditions.loc["blank", "mean"], 11.0)
        self.assertEqual(raw_conditions.loc["original", "mean"], 3.0)
        self.assertEqual(raw_conditions.loc["mirror", "mean"], 5.0)

    def test_mixed_legacy_and_distance_metadata(self) -> None:
        frame = pd.concat([ball_frame(blank=False), train_frame()], ignore_index=True)
        summary, _ = summarize_beyond_visible(frame, draws=1000)
        raw = summary[summary.readout.eq("raw_target")].set_index("family")
        self.assertEqual(raw.loc["offscreen_ball", "expected_effect_mean"], 6.0)
        self.assertEqual(raw.loc["offscreen_train", "expected_effect_mean"], 7.0)

    def test_incomplete_or_duplicate_conditions_are_rejected(self) -> None:
        frame = train_frame()
        for invalid in (frame.drop(index=1), pd.concat([frame, frame.iloc[[0]]])):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                summarize_beyond_visible(invalid, draws=10)
        # A blank control missing from only one ball prompt must also fail.
        with self.assertRaisesRegex(ValueError, "complete matched"):
            summarize_beyond_visible(ball_frame().drop(index=2), draws=10)

    def test_inconsistent_contrast_metadata_is_rejected(self) -> None:
        for column, value in (
            ("meta_effect_axis", "horizontal"),
            ("meta_effect_conditions_a", ["original"]),
            ("meta_effect_sign", -1),
        ):
            frame = train_frame()
            frame.at[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                summarize_beyond_visible(frame, draws=10)

    def test_missing_axis_or_nonfinite_controls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_distance"):
            summarize_beyond_visible(
                train_frame().drop(columns="target_distance"), draws=10
            )
        frame = ball_frame()
        frame.at[2, "target_horizontal"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_beyond_visible(frame, draws=10)

    def test_new_specs_build_180_cases_with_checked_spans_and_contrasts(self) -> None:
        specs = [
            json.loads((ROOT / "configs/experiments/analysis/cases" / name).read_text())
            for name in (
                "beyond_visible_ball_owner_with_blank.json",
                "beyond_visible_train.json",
            )
        ]
        groups = [group for spec in specs for group in spec["groups"]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Only image identity/checksum handling is exercised in this CPU test.
            for name in {
                image
                for group in groups
                for image in group["condition_images"].values()
            }:
                (root / name).write_bytes(b"test-only image identity")
            rows = [
                row for group in groups for row in build_paired_prompts(group, root)
            ]
            manifest = root / "cases.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cases = load_case_manifest(manifest, root)
            self.assertEqual(len(cases), 180)
            for group in groups:
                selected = [case for case in cases if case.family == group["family"]]
                self.assertEqual(len(selected), 60)
                for case in selected:
                    for role in ("target", "reference"):
                        start, end = case.object_spans[role]
                        self.assertEqual(case.prompt[start:end], group[role])
                    self.assertEqual(
                        case.metadata["effect_axis"],
                        group.get("effect_axis", "horizontal"),
                    )
            owner = next(
                case
                for case in cases
                if case.family == "offscreen_owner" and case.condition == "original"
            )
            self.assertEqual(
                owner.images[0].path.name, "tight-leash-corridor-mirror.jpg"
            )
            self.assertIn("image-right", owner.metadata["condition_semantics"])
            train = next(case for case in cases if case.family == "offscreen_train")
            self.assertEqual(train.metadata["effect_conditions_a"], ["blank"])
            self.assertEqual(
                train.metadata["effect_conditions_b"], ["original", "mirror"]
            )




class WebCaseTests(unittest.TestCase):
    def test_bundled_web_cases_validate_and_include_every_token(self):
        image_root = ROOT / "sspace/data/contextual_cases/s-space-web"
        spec = json.loads((ROOT / "configs/experiments/analysis/cases/offscreen_observer.json").read_text())
        rows = [row for group in spec["groups"] for row in build_web_showcase(group, image_root)]
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({row["family"] for row in rows}), 6)
        for row in rows:
            start, end = row["object_spans"]["target"]
            self.assertEqual(row["prompt"][start:end], row["metadata"]["token_text"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(len(load_case_manifest(path, image_root)), 13)
        political = json.loads((ROOT / "configs/experiments/analysis/cases/political_semantics.json").read_text())
        rows = build_political_templates(political["groups"][0], image_root)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["metadata"]["target_word"] for r in rows], ["socialist", "moderate", "conservative"])
        self.assertTrue(all("moderator member." in r["prompt"] for r in rows))

    def test_political_web_population_normalization_and_incomplete_rejection(self):
        frame = pd.DataFrame({
            "family": ["political_semantics"] * 3,
            "condition": ["parliament"] * 3,
            "selected_layer": [17] * 3,
            "meta_target_word": ["socialist", "moderate", "conservative"],
            "target_horizontal": [-5.533, -3.338, -2.404],
        })
        result = summarize_web_political(frame)
        self.assertAlmostEqual(result.within_model_zscore.mean(), 0)
        self.assertAlmostEqual(result.within_model_zscore.std(ddof=0), 1)
        shifted = frame.copy()
        shifted["target_horizontal"] = shifted.target_horizontal * 3 + 10
        np.testing.assert_allclose(result.within_model_zscore, summarize_web_political(shifted).within_model_zscore)
        with self.assertRaises(ValueError):
            summarize_web_political(frame.iloc[:2])


if __name__ == "__main__":
    unittest.main()

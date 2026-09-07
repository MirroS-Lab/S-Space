"""Regression checks for parsing, recovery and release asset selection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sspace.core.pipeline import wait_for_worker_shards
from sspace.data.assets import DATASET_ASSETS, selected_assets
from sspace.experiments.analysis.contextual_cases.scripts import extract_qwen_mmsi_direction_tokens as mmsi
from sspace.experiments.analysis.contextual_cases.token_matching import find_direction_occurrences
from sspace.experiments.common.direct import generation
from sspace.experiments.spinbench.perspective_taking.tools import run_spinbench_layerwise_sweep as sweep
from sspace.run_records import atomic_text, canonical_fingerprint


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class ReviewFixTests(unittest.TestCase):
    def test_single_token_answers_remain_supported(self):
        for answer in "ABCD":
            for text in (answer, f"({answer})", f"{answer}.", f"Answer: {answer}", f"The answer is {answer}."):
                with self.subTest(text=text):
                    self.assertEqual(mmsi.parse_answer(text), answer)

    def test_prose_is_not_an_answer_letter(self):
        for text in ("The answer could depend on perspective.", "The option appears plausible.", "Answer: Below", "Answer is ambiguous"):
            with self.subTest(text=text):
                self.assertIsNone(mmsi.parse_answer(text))

    def test_direction_words_require_boundaries(self):
        text = "rightfully leftover topology northwestern upright left; (right), north west; south-east"
        matches = find_direction_occurrences(CharacterTokenizer(), list(map(ord, text)))
        self.assertEqual([row["word"] for row in matches], ["left", "right", "northwest", "southeast"])

    def test_unparsed_answers_count_in_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mmsi.write_outputs(root, [
                {"prediction": "A", "correct": True},
                {"prediction": None, "correct": None},
            ])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["accuracy"], .5)
            self.assertEqual(summary["parsed_accuracy"], 1.)
            self.assertEqual(summary["parse_failures"], 1)

    def test_first_image_failure_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(model_adapter="fixture", shard_count=1, to_dict=lambda: {"model_adapter": "fixture"})
            runtime = SimpleNamespace(spec=SimpleNamespace(key="fixture"), model=object())
            recorder = SimpleNamespace(directory=Path(tmp))
            sample = SimpleNamespace(validate=lambda: None)
            with patch.object(generation, "_eos_ids", return_value={0}), patch.object(generation, "_image", side_effect=RuntimeError("image failure")):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, "image failure"):
                        generation.run_direct_generation_shard(runtime, [sample], "fixture", config, 0, recorder)

    def test_atomic_report_preserves_previous_on_failure_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text("previous")
            with self.assertRaises(RuntimeError):
                with atomic_text(path) as stream:
                    stream.write("partial")
                    raise RuntimeError("interrupted")
            self.assertEqual(path.read_text(), "previous")
            self.assertFalse(path.with_suffix(".csv.tmp").exists())
            with atomic_text(path) as stream:
                stream.write("complete")
            self.assertEqual(path.read_text(), "complete")

    def test_stale_failure_does_not_block_completed_shard(self):
        self._check_worker_failure("old", False)

    def test_current_failure_still_aborts(self):
        self._check_worker_failure("current", True)

    def test_worker_waits_for_current_launch_ready_marker(self):
        from sspace.core import cli

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SSPACE_ATTEMPT_ID": "current"}):
            root = Path(tmp)
            resolved = {"config": "same"}
            ready = root / "run_ready.json"
            ready.write_text(json.dumps({"attempt_id": "old", "run_fingerprint": canonical_fingerprint(resolved)}))
            (root / "failure.json").write_text('{"attempt_id":"old"}')

            def initialize(_):
                ready.write_text(json.dumps({"attempt_id": "current", "run_fingerprint": canonical_fingerprint(resolved)}))

            with patch.object(cli.time, "sleep", side_effect=initialize) as sleep:
                cli._initialize_run(root, resolved, 1)
                sleep.assert_called_once()

    def _check_worker_failure(self, attempt, raises):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SSPACE_ATTEMPT_ID": "current"}):
            root = Path(tmp)
            (root / "rank_failures").mkdir()
            (root / "rank_failures/rank_000.json").write_text(json.dumps({"attempt_id": attempt, "message": "worker failure"}))
            shard = root / "shards/rank_000"
            shard.mkdir(parents=True)
            (shard / "manifest.json").write_text('{"status":"complete"}')
            if raises:
                with self.assertRaisesRegex(RuntimeError, "worker failure"):
                    wait_for_worker_shards(root, 1)
            else:
                wait_for_worker_shards(root, 1)

    def test_all_assets_include_models_and_datasets(self):
        assets = selected_assets(("all",), tuple(DATASET_ASSETS))
        self.assertEqual({asset.repo_type for asset in assets}, {"model", "dataset"})
        self.assertEqual(len(assets), len({asset.key for asset in assets}))

    def test_dataset_only_selection_does_not_add_models(self):
        assets = selected_assets((), tuple(DATASET_ASSETS))
        self.assertEqual({asset.repo_type for asset in assets}, {"dataset"})

    def test_instructpart_cache_requires_retained_selection(self):
        from sspace.experiments.action_supervision.instructpart import prepare
        from sspace.experiments.action_supervision.instructpart.config import DEFAULT_CONFIG, load_config

        config = load_config(DEFAULT_CONFIG)
        ground_truth = config["pseudo_ground_truth"]
        digests = {
            "selection.json": ground_truth["expected_filtered_selection_sha256"],
            "object_boxes.json": ground_truth["expected_object_boxes_sha256"],
            "retained_selection.json": ground_truth["expected_retained_selection_sha256"],
        }
        for missing in (False, True):
            with self.subTest(missing=missing), patch("sys.argv", ["prepare"]), \
                 patch.object(prepare, "load_config", return_value=config), patch.object(prepare, "_run") as run, \
                 patch.object(Path, "is_file", lambda path: not (missing and path.name == "retained_selection.json")), \
                 patch.object(prepare, "file_sha256", side_effect=lambda path: digests[path.name]), \
                 patch.object(prepare, "_value_fingerprint", return_value=ground_truth["expected_ground_truth_values_sha256"]):
                prepare.main()
                self.assertEqual(run.call_count, 3 if missing else 1)

    def test_runtime_identity_includes_backend_revision_and_source(self):
        from sspace.experiments.identity import runtime_identity
        from sspace.core.models.runtime import RUNTIME_SPECS

        with patch("sspace.experiments.identity.python_source_identity", return_value={"hash": "source"}):
            identity = runtime_identity("qwen36_27b", Path.cwd())
        self.assertEqual(identity["python_source"], {"hash": "source"})
        self.assertEqual(identity["model_revision"], RUNTIME_SPECS["qwen36_27b"].revision)
        self.assertEqual(identity["attention_backend"], "sdpa")
        self.assertTrue(identity["torch_version"])

    def test_layerwise_completion_checks_identity_before_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text("{}")
            config = SimpleNamespace(annotation_path=root / "data", annotation_sha256="digest", image_root=root,
                                     premise_mode="without", expected_sample_count=2, expected_dataset_fingerprint="dataset",
                                     model_adapter="fixture", artifact_dir=root, to_dict=lambda: {})
            artifact = SimpleNamespace(manifest=SimpleNamespace(artifact_id="axis", model=SimpleNamespace(layer_ids=[0])))
            resolved = dict(runtime_identity={}, config_path=str(config_path), output_dir=str(root),
                            dataset_fingerprint="dataset", artifact_id="axis", artifact_manifest_sha256="digest",
                            protocol=sweep.PROTOCOL, layer_ids=[0], limit=1)
            manifest = {"status": "complete", "sample_count": 1, "run_fingerprint": canonical_fingerprint(resolved)}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with patch.object(sweep, "file_sha256", return_value="digest"), patch.object(sweep, "load_spinbench", return_value=[1, 2]), \
                 patch.object(sweep, "spinbench_fingerprint", return_value="dataset"), patch.object(sweep, "runtime_identity", return_value={}), \
                 patch.object(sweep, "bind_launch_identity"), patch.object(sweep, "_extract_layerwise_margins") as extract:
                sweep._run_condition(None, artifact, config, config_path, root, root, limit=1)
                extract.assert_not_called()
                with self.assertRaisesRegex(ValueError, "different run"):
                    sweep._run_condition(None, artifact, config, config_path, root, root, limit=None)
                manifest["sample_count"] = 2
                (root / "run_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "sample count"):
                    sweep._run_condition(None, artifact, config, config_path, root, root, limit=1)


if __name__ == "__main__":
    unittest.main()

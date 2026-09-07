"""CPU-only contracts for the minimal release workflow boundary."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sspace.data import assets_cli
from sspace.data.assets import AUXILIARY_MODEL_ASSETS
from sspace.data.assets import ACTION_SUPERVISION_MODEL_ASSETS
from sspace.experiments.analysis.object_coordinate_lens.models import (
    MODEL_SPECS as OBJECT_LENS_MODEL_SPECS,
)
from sspace.experiments.analysis.object_coordinate_lens.cli import (
    WORKER_ENVIRONMENTS,
)
from sspace.experiments.spinbench.evolving_cot import runtime as cot_runtime
from sspace.experiments.common.direct.merge_cli import (
    _merge_records_atomically,
    _validate_record,
    _validate_shard_manifest,
)
from sspace.experiments.common.direct.schema import (
    DirectGenerationRecord,
    DirectGenerationSample,
)
from sspace.experiments.intervention.spatial_causality.cli import DEFAULT_AXES
from sspace.experiments.intervention.spatial_causality.spatial_jacobian.constants import (
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_MODEL_PATH,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample() -> DirectGenerationSample:
    return DirectGenerationSample(
        benchmark="unit",
        sample_id="sample-0",
        split="test",
        group="horizontal",
        query="cup",
        reference="book",
        gold_endpoint="right",
        prompt="Where is the cup relative to the book?",
        image_bytes=b"unit-image",
        subset_tags=("tiny",),
    )


def _record() -> DirectGenerationRecord:
    sample = _sample()
    return DirectGenerationRecord(
        evaluation_id="unit-evaluation",
        benchmark=sample.benchmark,
        sample_id=sample.sample_id,
        global_sample_index=0,
        shard_rank=0,
        shard_count=1,
        split=sample.split,
        subset_tags=sample.subset_tags,
        group=sample.group,
        query=sample.query,
        reference=sample.reference,
        gold_endpoint=sample.gold_endpoint,
        prompt=sample.prompt,
        image_sha256="0" * 64,
        thinking=False,
        output_tokens=1,
        ended_with_eos=True,
        thinking_completed=None,
        reasoning_response="",
        final_response="RIGHT.",
        raw_response="RIGHT.",
        generated_token_sha256="1" * 64,
        predicted_endpoint="right",
        parse_failure=False,
        correct=True,
        generation_seconds=0.1,
        metadata={},
    )


class AssetSelectionTests(unittest.TestCase):
    def test_asset_cli_requires_an_explicit_selection(self) -> None:
        arguments = [
            "assets_cli",
            "--asset-root",
            str(ROOT / ".cache/assets"),
            "--dry-run",
        ]
        with patch.object(sys, "argv", arguments), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                assets_cli.main()
        self.assertEqual(context.exception.code, 2)

    def test_asset_cli_selects_exactly_one_model(self) -> None:
        arguments = [
            "assets_cli",
            "--asset-root",
            str(ROOT / ".cache/assets"),
            "--model",
            "qwen35_4b",
            "--dry-run",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(assets_cli, "acquire_hub_assets") as acquire,
        ):
            assets_cli.main()
        selected = tuple(acquire.call_args.args[1])
        self.assertEqual([asset.key for asset in selected], ["qwen35_4b"])
        self.assertTrue(acquire.call_args.kwargs["dry_run"])

    def test_asset_cli_selects_one_object_lens_dependency(self) -> None:
        arguments = [
            "assets_cli",
            "--asset-root",
            str(ROOT / ".cache/assets"),
            "--model",
            "grounding_dino",
            "--dry-run",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(assets_cli, "acquire_hub_assets") as acquire,
        ):
            assets_cli.main()
        selected = tuple(acquire.call_args.args[1])
        self.assertEqual([asset.key for asset in selected], ["grounding_dino"])


class ScriptBoundaryTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ, PYTHON="/bin/echo")
        return subprocess.run(
            [str(ROOT / "scripts/reproduce_small.sh"), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_small_run_forces_isolated_output(self) -> None:
        result = self._run("configs/unit.json", "3", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-m sspace.experiments.scripts.run_suite", result.stdout)
        self.assertIn("--config " + str(ROOT / "configs/unit.json"), result.stdout)
        self.assertIn(
            "--output-root " + str(ROOT / "outputs/small/limit_3"),
            result.stdout,
        )
        self.assertIn("--limit 3", result.stdout)

    def test_small_run_rejects_output_override(self) -> None:
        result = self._run("configs/unit.json", "3", "--output-root", "outputs/formal")
        self.assertEqual(result.returncode, 2)
        self.assertIn("controls --output-root", result.stderr)

    def test_prepare_models_routes_one_explicit_model(self) -> None:
        environment = dict(os.environ, PYTHON="/bin/echo")
        result = subprocess.run(
            [str(ROOT / "scripts/prepare_models.sh"), "qwen35_4b", "--dry-run"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model qwen35_4b", result.stdout)
        self.assertNotIn("--group models", result.stdout)

    def test_prepare_models_routes_object_lens_dependencies(self) -> None:
        environment = dict(os.environ, PYTHON="/bin/echo")
        result = subprocess.run(
            [str(ROOT / "scripts/prepare_models.sh"), "object_lens", "--dry-run"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--group object_lens", result.stdout)

    def test_prepare_models_routes_action_supervision_dependencies(self) -> None:
        environment = dict(os.environ, PYTHON="/bin/echo")
        result = subprocess.run(
            [
                str(ROOT / "scripts/prepare_models.sh"),
                "action_supervision",
                "--dry-run",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--group action_supervision", result.stdout)

    def test_controller_scripts_do_not_fall_back_to_system_python(self) -> None:
        expected = "PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}"
        for name in (
            "list_experiments.sh",
            "prepare_data.sh",
            "prepare_models.sh",
            "reproduce_small.sh",
            "run_all_dry.sh",
            "run_experiment.sh",
            "run_smoke.sh",
        ):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn(expected, source)
            self.assertNotIn("PYTHON=${PYTHON:-python3}", source)

    def test_extra_scripts_route_project_environments(self) -> None:
        runner = (ROOT / "scripts/run_step.sh").read_text(encoding="utf-8")
        listing = (ROOT / "scripts/list_steps.sh").read_text(encoding="utf-8")
        self.assertIn("evolving-cot-*|mmsi-extract", runner)
        self.assertIn('.envs/qwen36/bin/python"', runner)
        self.assertIn('.envs/spatialmqa/bin/python"', runner)
        self.assertNotIn("PYTHON=${PYTHON:-python3}", runner + listing)

    def test_action_supervision_runner_uses_project_environments(self) -> None:
        runner_path = ROOT / "scripts/run_action_supervision.sh"
        self.assertTrue(os.access(runner_path, os.X_OK))
        runner = runner_path.read_text(encoding="utf-8")
        self.assertIn(".envs/spatialmqa/bin/python", runner)
        self.assertIn(".envs/molmoact/bin/python", runner)
        self.assertIn(
            "configs/experiments/action_supervision/instructpart_prompt_ensemble.json",
            runner,
        )
        self.assertIn(
            "sspace.experiments.action_supervision.instructpart.prepare",
            runner,
        )


class DirectMergeTests(unittest.TestCase):
    def test_different_launch_identity_is_rejected(self) -> None:
        manifest = {
            "status": "complete",
            "dataset_fingerprint": "data-fingerprint",
            "config": {
                "launch_fingerprint": "old-model-or-protocol",
                "rank": 0,
                "effective_shard_count": 1,
            },
        }
        with self.assertRaisesRegex(ValueError, "launch identity differs"):
            _validate_shard_manifest(
                manifest,
                "data-fingerprint",
                "current-model-and-protocol",
                0,
                1,
            )

    def test_final_response_controls_redundant_semantics(self) -> None:
        sample = _sample()
        with self.assertRaisesRegex(ValueError, "parsed semantics"):
            _validate_record(
                replace(_record(), correct=False),
                sample,
                0,
                0,
                1,
                "unit-evaluation",
                False,
            )

    def test_failed_merge_preserves_final_and_retry_replaces_it(self) -> None:
        scratch = ROOT / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            records = Path(directory) / "records.jsonl"
            records.write_text("previous-complete-records\n", encoding="utf-8")
            invalid = replace(_record(), predicted_endpoint="left", correct=False)
            invalid_line = json.dumps(invalid.to_dict()) + "\n"
            with self.assertRaisesRegex(ValueError, "parsed semantics"):
                _merge_records_atomically(
                    records,
                    [io.StringIO(invalid_line)],
                    [_sample()],
                    1,
                    "unit-evaluation",
                    False,
                )
            self.assertEqual(
                records.read_text(encoding="utf-8"), "previous-complete-records\n"
            )
            self.assertTrue((Path(directory) / "records.jsonl.tmp").is_file())

            valid_line = json.dumps(_record().to_dict()) + "\n"
            _merge_records_atomically(
                records,
                [io.StringIO(valid_line)],
                [_sample()],
                1,
                "unit-evaluation",
                False,
            )
            self.assertEqual(
                json.loads(records.read_text(encoding="utf-8"))["correct"], True
            )
            self.assertFalse((Path(directory) / "records.jsonl.tmp").exists())


class DependencyAndPathTests(unittest.TestCase):
    def test_qwen_runtime_pins_torchvision(self) -> None:
        requirements = (ROOT / "requirements-qwen.txt").read_text(encoding="utf-8")
        self.assertIn("torchvision==0.28.0\n", requirements)

    def test_spatial_causality_default_axes_exists(self) -> None:
        self.assertTrue(DEFAULT_AXES.is_file(), DEFAULT_AXES)

    def test_retained_extras_use_prepared_asset_paths(self) -> None:
        model_root = ROOT / ".cache/assets/models"
        self.assertEqual(cot_runtime.MODEL_PATH, model_root / "qwen36_27b")
        self.assertEqual(DEFAULT_MODEL_PATH, model_root / "molmo2_er")
        self.assertEqual(
            DEFAULT_ANNOTATION_PATH,
            ROOT / ".cache/assets/datasets/embspatial_annotations/data/"
            "test-00000-of-00001.parquet",
        )
        self.assertEqual(
            {asset.key for asset in AUXILIARY_MODEL_ASSETS},
            {
                "grounding_dino",
                "sam_vit_base",
                "slimsam_50_uniform",
                "depth_anything_v2_small_hf",
            },
        )
        self.assertEqual(
            {asset.key for asset in ACTION_SUPERVISION_MODEL_ASSETS},
            {
                "grounding_dino",
                "slimsam_50_uniform",
                "depth_anything_v2_small_hf",
            },
        )
        for key, spec in OBJECT_LENS_MODEL_SPECS.items():
            self.assertEqual(spec.checkpoint_path, model_root / key)
        self.assertEqual(
            WORKER_ENVIRONMENTS,
            {
                "molmo2_er": "spatialmqa",
                "molmoact2": "molmoact",
                "molmoact2_pretrain": "molmoact",
                "qwen35_4b": "qwen36",
                "qwen36_27b": "qwen36",
            },
        )


if __name__ == "__main__":
    unittest.main()

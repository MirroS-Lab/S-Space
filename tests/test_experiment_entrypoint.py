"""Exercise unified launcher routes without loading models."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExperimentEntrypointTests(unittest.TestCase):
    def run_entry(self, *args):
        return subprocess.run(
            [str(ROOT / "scripts/run_experiment.sh"), *args],
            cwd=ROOT,
            env=dict(os.environ, PYTHON=sys.executable,
                     CONTROLLER_PYTHON=sys.executable, MOLMO2_PYTHON=sys.executable,
                     MOLMOACT_PYTHON=sys.executable),
            text=True, capture_output=True, check=False,
        )

    def test_list_includes_all_workflow_types(self):
        result = self.run_entry("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("Configs", "action_supervision", "benchmarks-generalization",
                     "mmsi-extract", "spinbench-layerwise", "evolving-cot-extract"):
            self.assertIn(name, result.stdout)
        self.assertNotIn("evolving-cot-replay", result.stdout)

    def test_step_dry_run_is_consumed_by_launcher(self):
        result = self.run_entry("--step", "mmsi-extract", "--dry-run",
                                "--manifest", "not-a-real-input.jsonl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("extract_qwen_mmsi_direction_tokens", result.stdout)
        self.assertIn("--manifest not-a-real-input.jsonl", result.stdout)
        self.assertNotIn("--dry-run", result.stdout)

    def test_config_and_suite_routes_keep_model_dispatch(self):
        config = self.run_entry("configs/experiments/analysis/cases/run.example.json", "--dry-run")
        self.assertEqual(config.returncode, 0, config.stderr)
        self.assertIn("contextual_cases.cli", config.stdout)
        suite = self.run_entry("--suite", "spinbench-projection", "--dry-run",
                               "--model", "molmo2_er", "--python", f"molmo2_er={sys.executable}")
        self.assertEqual(suite.returncode, 0, suite.stderr)
        self.assertIn("perspective_taking.projection_cli", suite.stdout)

    def test_workflow_dry_run_and_unknown_workflow(self):
        result = self.run_entry("--workflow", "action_supervision", "--dry-run", "--limit", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        for stage in ("instructpart.prepare", "instructpart.projection", "instructpart.scoring"):
            self.assertIn(stage, result.stdout)
        self.assertIn("--limit 2", result.stdout)
        invalid = self.run_entry("--workflow", "unknown")
        self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()

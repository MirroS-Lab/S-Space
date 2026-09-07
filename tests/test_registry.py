"""Formal Config and command-graph gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sspace.experiments.identity import experiment_launch_identity
from sspace.experiments.scripts import run_suite
from sspace.registry import load_registry
from sspace import run_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_only_extra_spatialtunnel_intervention_is_removed(self) -> None:
        from sspace.experiments.launch import RUNNER_MODULES

        registry = load_registry()
        self.assertNotIn("projection_swap", RUNNER_MODULES)
        self.assertNotIn("projection_swap", registry["runner_modules"])
        self.assertFalse(any("projection-swap" in entry["suites"] for entry in registry["formal_configs"]))
        self.assertTrue(any("spatialtunnel" in entry["path"] for entry in registry["formal_configs"]))
        self.assertTrue(any(entry["id"] == "spatial-causality" for entry in registry["formal_workflows"]))

    def test_explicit_formal_experiment_contract(self) -> None:
        registry = load_registry()
        self.assertEqual(len(registry["formal_configs"]), registry["expected_config_count"])
        self.assertGreater(registry["expected_command_count"], 0)
        self.assertEqual(len(registry["formal_workflows"]), registry["expected_workflow_count"])
        self.assertEqual(
            registry["formal_workflows"][0]["id"],
            "action_supervision/instructpart_prompt_ensemble",
        )

    def test_static_command_graph_does_not_consult_completed_outputs(self) -> None:
        overrides = {
            model: Path(sys.executable) for model in run_suite.DEFAULT_MODEL_PYTHONS
        }
        with patch.object(
            run_suite, "_completed_and_valid", return_value=True
        ) as completed:
            commands = run_suite._job_commands(
                run_suite.ALL_JOBS,
                overrides,
                require_python=True,
                skip_completed=False,
            )
        self.assertEqual(len(commands), load_registry()["expected_command_count"])
        completed.assert_not_called()

    def test_single_registered_config_accepts_output_root(self) -> None:
        job = run_suite.ALL_JOBS[0]
        relative = job.config_path.relative_to(PROJECT_ROOT)
        output_root = Path("outputs/smoke/single")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_suite",
                    "--config",
                    str(relative),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
            ),
            patch.object(
                run_suite, "_with_output_root", return_value=(job,)
            ) as rewrite,
            patch.object(run_suite, "_job_commands", return_value=()) as commands,
            patch.object(run_suite, "run_commands") as execute,
        ):
            run_suite.main()
        rewrite.assert_called_once_with((job,), output_root, {})
        commands.assert_called_once()
        execute.assert_called_once_with((), True)

    def test_launch_identity_binds_source_and_model_marker_not_weights(self) -> None:
        temporary_root = PROJECT_ROOT / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            package = root / "sspace"
            package.mkdir()
            source = package / "algorithm.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
            model = root / f"models/{revision}"
            model.mkdir(parents=True)
            marker = model / ".sspace_hub_asset.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "repo_id": "Qwen/Qwen3.5-4B",
                        "repo_type": "model",
                        "revision": revision,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "experiment.json"
            config.write_text(
                json.dumps(
                    {
                        "model_adapter": "qwen35_4b",
                        "model_path": f"models/{revision}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            initial = experiment_launch_identity(config, root, None)
            (model / "model.safetensors").write_bytes(b"large-weight-placeholder")
            after_weight = experiment_launch_identity(config, root, None)
            self.assertEqual(initial, after_weight)

            marker.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "repo_id": "Qwen/Qwen3.5-4B",
                        "repo_type": "model",
                        "revision": "revision-b",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "marker does not match"):
                experiment_launch_identity(config, root, None)

            marker.unlink()
            after_marker = experiment_launch_identity(config, root, None)
            self.assertEqual(
                after_marker["model"]["identity_source"],
                "resolved_revision_directory",
            )

            source.write_text("VALUE = 2\n", encoding="utf-8")
            after_source = experiment_launch_identity(config, root, None)
            self.assertNotEqual(after_marker, after_source)

            unpinned = root / "models/unpinned-copy"
            unpinned.mkdir()
            config.write_text(
                json.dumps(
                    {
                        "model_adapter": "qwen35_4b",
                        "model_path": "models/unpinned-copy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no asset marker"):
                experiment_launch_identity(config, root, None)

    def test_single_config_rejects_registry_digest_mismatch(self) -> None:
        job = run_suite.ALL_JOBS[0]
        with (
            patch.object(run_suite, "file_sha256", return_value="wrong"),
            self.assertRaisesRegex(ValueError, "Config digest changed"),
        ):
            run_suite._registered_config_job(job.config_path)

    def test_git_metadata_is_optional_and_scoped_to_current_project(self) -> None:
        revision = subprocess.CompletedProcess(
            ["git", "rev-parse"],
            0,
            stdout=f"{PROJECT_ROOT.resolve()}\nabc123\n",
            stderr="",
        )
        status = subprocess.CompletedProcess(
            ["git", "status"], 0, stdout="?? untracked.py\n", stderr=""
        )
        with patch.object(
            run_records.subprocess, "run", side_effect=[revision, status]
        ) as run:
            identity = run_records.code_revision(PROJECT_ROOT)
        self.assertEqual(identity["commit"], "abc123")
        self.assertTrue(identity["project_worktree_dirty"])
        self.assertIn("--show-toplevel", run.call_args_list[0].args[0])
        status_command = run.call_args_list[1].args[0]
        self.assertIn("--untracked-files=all", status_command)
        self.assertEqual(status_command[-2:], ["--", "."])

        with patch.object(run_records.subprocess, "run", side_effect=FileNotFoundError):
            unavailable = run_records.code_revision(PROJECT_ROOT)
        self.assertEqual(
            unavailable,
            {
                "available": False,
                "commit": None,
                "project_worktree_dirty": None,
            },
        )

        ancestor = subprocess.CompletedProcess(
            ["git", "rev-parse"],
            0,
            stdout=f"{PROJECT_ROOT.parent.resolve()}\nabc123\n",
            stderr="",
        )
        with patch.object(run_records.subprocess, "run", return_value=ancestor):
            unavailable = run_records.code_revision(PROJECT_ROOT)
        self.assertFalse(unavailable["available"])


if __name__ == "__main__":
    unittest.main()

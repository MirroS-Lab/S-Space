"""Guard complete planning and fail before execution when case inputs are missing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sspace.experiments import workflows
from sspace.registry import REGISTRY_PATH, load_registry, sync_registry


class ExperimentWorkflowTests(unittest.TestCase):
    def test_all_plan_includes_benchmarks_and_every_workflow(self):
        with patch.object(workflows.subprocess, "run") as execute:
            groups = workflows.plan(True, None, [], True)
        execute.assert_not_called()
        names = {name for name, _ in groups}
        self.assertEqual(names, {entry["id"] for entry in load_registry()["formal_workflows"]} | {"benchmark-configs"})
        self.assertEqual(groups[0][0], "sspace-construction")
        self.assertEqual(len(dict(groups)["benchmark-configs"]), load_registry()["expected_command_count"] - 6)
        self.assertIn("sspace.experiments.analysis.contextual_cases.scripts.build_manifest", dict(groups)["contextual-cases"][0])
        commands = dict(groups)["context-scaling"]
        self.assertEqual(len(commands), 10)
        self.assertIn(".envs/qwen36/bin/python", commands[-2][0])
        self.assertIn(".envs/molmoact/bin/python", commands[2][0])

    def test_default_cases_build_before_inference_without_user_config(self):
        commands = workflows.plan(False, "contextual-cases", [], True)[0][1]
        self.assertEqual(len(commands), 14)
        built = set()
        for command in commands:
            if "--spec" in command:
                built.add(command[command.index("--output") + 1])
            if "--config" in command:
                path = workflows.PROJECT_ROOT / command[command.index("--config") + 1]
                config = json.loads(path.read_text())
                self.assertIn(config["manifest_path"], built)

    def test_unknown_experiment_does_not_launch(self):
        with self.assertRaisesRegex(ValueError, "Unknown experiment"):
            workflows.plan(False, "misspelled", [], True)

    def test_instructpart_binds_registered_config(self):
        commands = workflows.plan(False, "action_supervision", [], True)[0][1]
        self.assertIn("--config", commands[0])
        self.assertEqual(commands[0][-1], "configs/experiments/action_supervision/instructpart_prompt_ensemble.json")

    def test_sync_refreshes_metadata_and_remains_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            value = json.loads(REGISTRY_PATH.read_text())
            value["formal_configs"][0]["sha256"] = "outdated"
            value["expected_workflow_count"] = 1
            path.write_text(json.dumps(value))
            synced = sync_registry(path)
            self.assertEqual(load_registry(path), synced)
            self.assertEqual(synced["expected_workflow_count"], len(load_registry()["formal_workflows"]))


if __name__ == "__main__":
    unittest.main()

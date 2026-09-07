"""Keep contextual checkpoints bound to their code and runtime settings."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sspace.experiments.analysis.contextual_cases import cli
from sspace.run_records import RunRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContextualIdentityTests(unittest.TestCase):
    def test_limit_is_validated_before_loading_config(self):
        with (
            patch.object(sys, "argv", ["contextual", "--config", "unused", "--limit", "0"]),
            patch.object(cli, "configure_deterministic_cuda"),
            patch.object(cli, "load_case_config") as load,
        ):
            with self.assertRaisesRegex(ValueError, "--limit must be positive"):
                cli.main()
            load.assert_not_called()

    def test_limit_slices_samples_and_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text("{}")
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n")
            config = SimpleNamespace(
                model_adapter="qwen36_27b", manifest_path=manifest,
                output_dir=root, artifact_dir=root, image_root=root,
                model_path=root, device="cpu", seed=0, image_max_pixels=None,
                to_dict=lambda: {},
            )
            artifact = Mock()
            artifact.manifest.model.selected_layer = 43
            samples = [object(), object(), object()]
            with (
                patch.object(sys, "argv", ["contextual", "--config", str(config_path), "--limit", "1"]),
                patch.object(cli, "configure_deterministic_cuda"),
                patch.object(cli, "load_case_config", return_value=config),
                patch.object(cli, "RunRecorder") as recorder,
                patch.object(cli.SSpaceArtifact, "load", return_value=artifact),
                patch.object(cli, "load_case_manifest", return_value=samples),
                patch.object(cli, "load_model_runtime"),
                patch.object(cli, "configure_image_max_pixels"),
                patch.object(cli.torch, "manual_seed"),
                patch.object(cli.torch.cuda, "manual_seed_all"),
                patch.object(cli.torch, "use_deterministic_algorithms"),
                patch.object(cli, "run_case_evaluation", return_value={"sample_count": 1}) as evaluate,
            ):
                cli.main()
            self.assertEqual(evaluate.call_args.args[2], samples[:1])
            resolved = recorder.call_args.args[1]
            self.assertEqual(resolved["sample_limit"], 1)
            self.assertEqual(resolved["launch_identity"]["limit"], 1)

    def test_backend_change_rejects_old_checkpoint_directory(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"case": "identity-test"}))
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n")
            config = SimpleNamespace(
                model_adapter="qwen36_27b",
                manifest_path=manifest,
                output_dir=root / "run",
                to_dict=lambda: {"case": "identity-test"},
            )

            def resolved_for(backend):
                spec = replace(
                    cli.RUNTIME_SPECS["qwen36_27b"], attention_backend=backend
                )
                with (
                    patch.object(sys, "argv", ["contextual", "--config", str(config_path)]),
                    patch.object(cli, "load_case_config", return_value=config),
                    patch.object(cli, "configure_deterministic_cuda"),
                    patch.dict(cli.RUNTIME_SPECS, {"qwen36_27b": spec}),
                    patch.object(cli, "RunRecorder", side_effect=RuntimeError("stop before model load")) as recorder,
                ):
                    with self.assertRaisesRegex(RuntimeError, "stop before model load"):
                        cli.main()
                return recorder.call_args.args[1]

            eager = resolved_for("eager")
            sdpa = resolved_for("sdpa")
            self.assertIn("launch_fingerprint", sdpa)
            self.assertIn("python_source", sdpa["launch_identity"])
            self.assertEqual(sdpa["attention_backend"], "sdpa")
            self.assertEqual(sdpa["dtype"], "torch.bfloat16")
            RunRecorder(config.output_dir, eager, PROJECT_ROOT)
            with self.assertRaisesRegex(ValueError, "different resolved config"):
                RunRecorder(config.output_dir, sdpa, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()

"""Exercise recovery before SpinBench's first generated record."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sspace.experiments.spinbench.perspective_taking.direct import run_spinbench_direct


class SpinBenchResumeTest(unittest.TestCase):
    def test_first_preprocessing_failure_can_resume(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project_root / ".tmp") as directory:
            recorder = SimpleNamespace(directory=Path(directory))
            config = SimpleNamespace(
                model_adapter="qwen36_27b",
                expected_sample_count=1,
                premise_mode="without",
                thinking=False,
                to_dict=lambda: {"evaluation_id": "resume-test"},
            )
            runtime = SimpleNamespace(
                spec=SimpleNamespace(key="qwen36_27b"),
                model=SimpleNamespace(
                    generation_config=SimpleNamespace(eos_token_id=2)
                ),
                processor=object(),
                prepare_generation=Mock(side_effect=RuntimeError("processor failed")),
            )
            samples = [SimpleNamespace(premise_mode="without", direct_prompt="A or B?")]
            with patch(
                "sspace.experiments.spinbench.perspective_taking.direct._load_image",
                side_effect=RuntimeError("image failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "image failed"):
                    run_spinbench_direct(runtime, samples, "dataset", config, recorder)
            with patch(
                "sspace.experiments.spinbench.perspective_taking.direct._load_image",
                return_value=(object(), "image-sha"),
            ):
                with self.assertRaisesRegex(RuntimeError, "processor failed"):
                    run_spinbench_direct(runtime, samples, "dataset", config, recorder)
            runtime.prepare_generation.assert_called_once()
            checkpoint = json.loads((Path(directory) / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["next_sample_index"], 0)
            self.assertEqual(checkpoint["records_offset"], 0)
            self.assertEqual((Path(directory) / "records.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

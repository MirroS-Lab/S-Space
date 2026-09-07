"""Build the independent COCO-1800 layer-selection dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sspace.run_records import RunRecorder

from .config import load_coco_validation_config
from .validation_dataset import prepare_coco_validation


def main() -> None:
    """Build COCO-1800 from one exact configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = load_coco_validation_config(args.config, project_root, args.asset_root)
    resolved = {
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(config).items()
        },
        "workflow": "coco_validation1800",
        "config_path": str(args.config.resolve()),
    }
    recorder = RunRecorder(args.run_dir, resolved, project_root)
    stage = "prepare_coco_validation"
    try:
        recorder.event(stage, "started")
        manifest = prepare_coco_validation(config)
        recorder.event(
            stage,
            "complete",
            dataset_id=manifest["dataset_id"],
            dataset_fingerprint=manifest["dataset_fingerprint"],
            counts=manifest["counts"],
        )
        recorder.complete(
            dataset_id=manifest["dataset_id"],
            dataset_fingerprint=manifest["dataset_fingerprint"],
            output_dir=str(config.output_dir),
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

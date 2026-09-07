"""Run the formal COCO preprocessing workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sspace.run_records import RunRecorder

from .config import load_coco_training_config
from .pipeline import prepare_coco_training
from .validation import validate_preprocessed_coco


def main() -> None:
    """Run the single configured COCO preprocessing path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if (args.config is None) == (args.validate is None):
        parser.error("provide exactly one of --config or --validate")
    if args.config is not None and args.run_dir is None:
        parser.error("--run-dir is required with --config")
    if args.config is not None and args.asset_root is None:
        parser.error("--asset-root is required with --config")
    if args.validate is not None and args.run_dir is not None:
        parser.error("--run-dir is only valid with --config")
    if args.validate is not None and args.asset_root is not None:
        parser.error("--asset-root is only valid with --config")
    project_root = Path(__file__).resolve().parents[3]
    if args.validate is not None:
        manifest = validate_preprocessed_coco(args.validate)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return

    config = load_coco_training_config(args.config, project_root, args.asset_root)
    resolved = {
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(config).items()
        },
        "workflow": "coco_training6000",
        "config_path": str(args.config.resolve()),
    }
    recorder = RunRecorder(args.run_dir, resolved, project_root)
    stage = "prepare_coco_training"
    try:
        recorder.event(stage, "started")
        manifest = prepare_coco_training(config)
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

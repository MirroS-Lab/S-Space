"""Acquire official COCO images required by COCO-1800."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sspace.run_records import RunRecorder, file_sha256

from .acquisition import acquire_validation_images
from .config import load_coco_validation_config


def main() -> None:
    """Acquire COCO-1800 images and save checksum-bound provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    asset_root = args.asset_root.expanduser().resolve()
    config = load_coco_validation_config(
        args.config, project_root, asset_root, require_output_absent=False
    )
    temporary = (
        args.temporary_directory
        if args.temporary_directory.is_absolute()
        else project_root / args.temporary_directory
    )
    if not temporary.resolve().is_relative_to((asset_root / ".tmp").resolve()):
        raise ValueError(
            "Acquisition temporary directory must be under asset-root .tmp"
        )
    resolved = {
        "workflow": "coco_validation1800_image_acquisition",
        "config_path": str(args.config.resolve()),
        "annotation_path": str(config.annotation_path),
        "annotation_sha256": file_sha256(config.annotation_path),
        "image_root": str(config.image_root),
        "training_dataset": str(config.training_dataset),
        "protocol": config.protocol,
        "depth_candidate_count": config.depth_candidate_count,
        "official_image_base_url": "http://images.cocodataset.org/train2017",
        "temporary_directory": str(temporary),
        "workers": args.workers,
    }
    recorder = RunRecorder(args.run_dir, resolved, project_root)
    stage = "acquire_validation_images"
    try:
        recorder.event(stage, "started")
        records = acquire_validation_images(config, temporary, args.workers)
        records_path = args.run_dir / "image_records.jsonl"
        with records_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        counts = Counter(record["status"] for record in records)
        group_counts = Counter(record["group"] for record in records)
        recorder.event(
            stage,
            "complete",
            image_count=len(records),
            status_counts=dict(sorted(counts.items())),
            group_counts=dict(sorted(group_counts.items())),
            records_sha256=file_sha256(records_path),
        )
        recorder.complete(
            image_count=len(records),
            status_counts=dict(sorted(counts.items())),
            group_counts=dict(sorted(group_counts.items())),
            records_sha256=file_sha256(records_path),
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()

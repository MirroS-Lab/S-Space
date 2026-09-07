"""Publish the exact audited COCO-6000 and COCO-1800 relation tables."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path

import numpy as np

from sspace.run_records import atomic_json

from .assets import COCO_ANNOTATION_SHA256
from .config import COCO_TRAINING_PROTOCOL, COCO_VALIDATION_PROTOCOL
from .image_integrity import selected_image_fingerprint
from .validation import validate_preprocessed_coco
from .validation_dataset import (
    TRAINING_DATASET_ID,
    VALIDATION_DATASET_ID,
    validate_coco_validation_dataset,
)


FROZEN_ROOT = Path(__file__).with_name("manifests") / "frozen"
ARCHIVE_SHA256 = {
    "coco6000_samples.jsonl.gz": "76d13191abed7410b5ebe5e282bb9aed632e47a5657361bbf3af2df5f2a3492b",
    "coco6000_audit.csv.gz": "977326743749e463f5c700c4e5b11dae4865980b675eb10f113df815ec4a4fd2",
    "coco1800_samples.jsonl.gz": "58cf388265fedce705dfd01b25241744b37e2425675f4fa7f35d9d511ae16170",
    "coco1800_audit.csv.gz": "be63195f770a1fa2b905f801466b54e2ee068dbe28d87442138f9baf9905df25",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(name: str, destination: Path) -> None:
    source = FROZEN_ROOT / name
    if _sha256(source) != ARCHIVE_SHA256[name]:
        raise ValueError(f"Frozen COCO archive checksum changed: {source}")
    with gzip.open(source, "rb") as compressed, destination.open("xb") as output:
        while chunk := compressed.read(1024 * 1024):
            output.write(chunk)


def _fingerprint(samples: Path, audit: Path, image_fingerprint: str) -> str:
    digest = hashlib.sha256()
    for value in (_sha256(samples), _sha256(audit), image_fingerprint):
        digest.update(f"{value}\n".encode())
    return digest.hexdigest()


def _selected_ids(path: Path) -> set[int]:
    return {
        int(json.loads(line)["image_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
    }


def _publish_split(
    output: Path,
    image_root: Path,
    annotation_path: Path,
    *,
    prefix: str,
    dataset_id: str,
    split: str,
    samples_per_endpoint: int,
    protocol: str,
    training_manifest: dict[str, object] | None,
) -> dict[str, object]:
    if output.exists():
        return validate_preprocessed_coco(output)
    output.mkdir(parents=True)
    samples = output / "samples.jsonl"
    audit = output / "audit.csv"
    rejections = output / "rejections.jsonl"
    _extract(f"{prefix}_samples.jsonl.gz", samples)
    _extract(f"{prefix}_audit.csv.gz", audit)
    rejections.touch(exist_ok=False)
    rows = [
        json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()
    ]
    with audit.open(encoding="utf-8") as stream:
        audit_count = sum(1 for _ in csv.DictReader(stream))
    expected_count = 6 * samples_per_endpoint
    if len(rows) != expected_count or audit_count != expected_count:
        raise ValueError(f"Frozen {dataset_id} row count changed")
    endpoint_counts = Counter(row["label"] for row in rows)
    image_fingerprint = selected_image_fingerprint(rows, image_root)
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_fingerprint": _fingerprint(samples, audit, image_fingerprint),
        "source": {
            "release": "COCO 2017 train instances",
            "annotation_path": str(annotation_path.resolve()),
            "annotation_sha256": COCO_ANNOTATION_SHA256,
            "image_root": os.path.relpath(image_root.resolve(), output.resolve()),
            "image_root_base": "dataset_dir",
            "selected_image_fingerprint": image_fingerprint,
        },
        "construction": {
            "protocol": protocol,
            "split": split,
            "seed": 42,
            "samples_per_endpoint": samples_per_endpoint,
            "depth_confidence_threshold": 0.12,
            "source": "audited_frozen_relation_tables",
        },
        "counts": {
            "rows": expected_count,
            "images": expected_count,
            "by_split": {split: expected_count},
            "by_label": dict(sorted(endpoint_counts.items())),
        },
        "files": {
            "samples": {"name": samples.name, "sha256": _sha256(samples)},
            "audit": {"name": audit.name, "sha256": _sha256(audit)},
            "rejections": {
                "name": rejections.name,
                "sha256": _sha256(rejections),
            },
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    if training_manifest is not None:
        manifest["training_exclusion"] = {
            "dataset_id": training_manifest["dataset_id"],
            "dataset_fingerprint": training_manifest["dataset_fingerprint"],
        }
    atomic_json(output / "manifest.json", manifest)
    return validate_preprocessed_coco(output)


def publish_frozen_coco_datasets(
    asset_root: Path, project_root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Publish and fully validate the exact separate COCO train/validation sets."""
    root = asset_root.expanduser().resolve()
    annotation = root / "datasets/coco2017/annotations/instances_train2017.json"
    images = root / "datasets/coco2017/train2017"
    if _sha256(annotation) != COCO_ANNOTATION_SHA256:
        raise ValueError("COCO annotation checksum changed")
    processed = project_root.resolve() / "data/processed"
    train_dir = processed / TRAINING_DATASET_ID
    validation_dir = processed / VALIDATION_DATASET_ID
    train = _publish_split(
        train_dir,
        images,
        annotation,
        prefix="coco6000",
        dataset_id=TRAINING_DATASET_ID,
        split="train",
        samples_per_endpoint=1000,
        protocol=COCO_TRAINING_PROTOCOL,
        training_manifest=None,
    )
    validation = _publish_split(
        validation_dir,
        images,
        annotation,
        prefix="coco1800",
        dataset_id=VALIDATION_DATASET_ID,
        split="validation",
        samples_per_endpoint=300,
        protocol=COCO_VALIDATION_PROTOCOL,
        training_manifest=train,
    )
    validate_coco_validation_dataset(validation_dir, train_dir)
    overlap = _selected_ids(train_dir / "samples.jsonl") & _selected_ids(
        validation_dir / "samples.jsonl"
    )
    if overlap:
        raise ValueError(f"Frozen COCO split overlap: {len(overlap)}")
    return train, validation

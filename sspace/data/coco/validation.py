"""Validate one frozen, single-split COCO dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .image_integrity import resolve_image_root, selected_image_fingerprint


ENDPOINTS = {"left", "right", "above", "below", "far", "close"}
GROUP_ENDPOINTS = {
    "horizontal": {"left", "right"},
    "vertical": {"above", "below"},
    "distance": {"far", "close"},
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "far": "close",
    "close": "far",
}
OBJECT_FIELDS = {"name", "annotation_id", "bbox_xyxy"}


def _bbox(value: Any, width: int, height: int, sample_id: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Invalid bbox for {sample_id}")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or not (
        0.0 <= box[0] < box[2] <= width and 0.0 <= box[1] < box[3] <= height
    ):
        raise ValueError(f"Invalid bbox for {sample_id}")
    return box


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(
    samples_path: Path, audit_path: Path, image_fingerprint: str
) -> str:
    digest = hashlib.sha256()
    for value in (_sha256(samples_path), _sha256(audit_path), image_fingerprint):
        digest.update(f"{value}\n".encode())
    return digest.hexdigest()


def validate_preprocessed_coco(directory: Path) -> dict[str, Any]:
    """Validate checksums, balance, split identity, and spatial labels.

    Args:
        directory: Frozen preprocessing output containing manifest, samples,
            audit records, and declared rejections.

    Returns:
        The parsed manifest after every check passes.

    Raises:
        FileNotFoundError: A declared output file is absent.
        ValueError: A checksum, count, split, uniqueness, bbox, depth, or label
            invariant fails.

    Side effects:
        Reads files without changing them.

    Example:
        ``manifest = validate_preprocessed_coco(Path("data/processed/run"))``
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"].values():
        path = directory / record["name"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path.name}")
    samples_path = directory / manifest["files"]["samples"]["name"]
    audit_path = directory / manifest["files"]["audit"]["name"]
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    image_root = resolve_image_root(directory, manifest["source"])
    image_fingerprint = selected_image_fingerprint(samples, image_root)
    if image_fingerprint != manifest["source"]["selected_image_fingerprint"]:
        raise ValueError("Selected image fingerprint mismatch")
    expected_fingerprint = _dataset_fingerprint(
        samples_path,
        audit_path,
        image_fingerprint,
    )
    if manifest["dataset_fingerprint"] != expected_fingerprint:
        raise ValueError("Dataset fingerprint mismatch")
    with audit_path.open(encoding="utf-8") as stream:
        audit = list(csv.DictReader(stream))
    if len(samples) != manifest["counts"]["rows"] or len(audit) != len(samples):
        raise ValueError("Sample, audit, and manifest row counts differ")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Sample IDs are not unique")
    if len({int(row["image_id"]) for row in samples}) != len(samples):
        raise ValueError("One-pair-per-image invariant failed")
    construction = manifest["construction"]
    split = construction.get("split")
    if split not in {"train", "validation"}:
        raise ValueError(f"Unknown dataset split {split!r}")
    samples_per_endpoint = int(construction["samples_per_endpoint"])
    counts = Counter((row["split"], row["label"]) for row in samples)
    expected = {(split, endpoint): samples_per_endpoint for endpoint in ENDPOINTS}
    if counts != expected:
        raise ValueError(f"Endpoint split balance failed: {dict(counts)}")
    audit_by_id = {row["sample_id"]: row for row in audit}
    if set(audit_by_id) != {row["sample_id"] for row in samples}:
        raise ValueError("Audit rows do not match sample IDs")
    threshold = float(manifest["construction"]["depth_confidence_threshold"])
    for sample in samples:
        expected_fields = {
            "sample_id",
            "image_id",
            "image_path",
            "group",
            "label",
            "split",
            "query",
            "reference",
            "swapped_label",
        }
        if set(sample) != expected_fields:
            raise ValueError(
                f"Sample fields for {sample['sample_id']} differ from the frozen schema"
            )
        sample_id = str(sample["sample_id"])
        group = sample["group"]
        label = sample["label"]
        if (
            group not in GROUP_ENDPOINTS
            or label not in GROUP_ENDPOINTS[group]
            or sample["swapped_label"] != OPPOSITE[label]
            or sample["split"] != split
        ):
            raise ValueError(f"Invalid group, label, or swap for {sample_id}")
        query = sample["query"]
        reference = sample["reference"]
        if (
            not isinstance(query, dict)
            or set(query) != OBJECT_FIELDS
            or not isinstance(reference, dict)
            or set(reference) != OBJECT_FIELDS
            or not str(query["name"]).strip()
            or not str(reference["name"]).strip()
            or query["name"] == reference["name"]
            or isinstance(query["annotation_id"], bool)
            or not isinstance(query["annotation_id"], int)
            or isinstance(reference["annotation_id"], bool)
            or not isinstance(reference["annotation_id"], int)
            or query["annotation_id"] == reference["annotation_id"]
        ):
            raise ValueError(f"Invalid object schema for {sample_id}")
        image_path = (image_root / str(sample["image_path"])).resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        query_box = _bbox(query["bbox_xyxy"], width, height, sample_id)
        reference_box = _bbox(reference["bbox_xyxy"], width, height, sample_id)
        record = audit_by_id[sample_id]
        expected_audit = {
            "image_id": str(sample["image_id"]),
            "group": str(group),
            "label": str(label),
            "split": str(split),
            "query": str(query["name"]),
            "reference": str(reference["name"]),
            "query_annotation_id": str(query["annotation_id"]),
            "reference_annotation_id": str(reference["annotation_id"]),
        }
        if any(record.get(key) != value for key, value in expected_audit.items()):
            raise ValueError(f"Audit row differs from sample {sample_id}")
        if sample["label"] == "left" and not query_box[2] < reference_box[0]:
            raise ValueError(f"Invalid left relation {sample['sample_id']}")
        if sample["label"] == "right" and not query_box[0] > reference_box[2]:
            raise ValueError(f"Invalid right relation {sample['sample_id']}")
        if sample["label"] == "above" and not query_box[3] < reference_box[1]:
            raise ValueError(f"Invalid above relation {sample['sample_id']}")
        if sample["label"] == "below" and not query_box[1] > reference_box[3]:
            raise ValueError(f"Invalid below relation {sample['sample_id']}")
        if (
            sample["group"] == "distance"
            and float(record["depth_confidence"]) < threshold
        ):
            raise ValueError(f"Low-confidence distance relation {sample['sample_id']}")
        if sample["label"] == "close" and not float(record["query_depth_mean"]) > float(
            record["reference_depth_mean"]
        ):
            raise ValueError(f"Invalid close relation {sample['sample_id']}")
        if sample["label"] == "far" and not float(record["query_depth_mean"]) < float(
            record["reference_depth_mean"]
        ):
            raise ValueError(f"Invalid far relation {sample['sample_id']}")
    return manifest

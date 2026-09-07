"""Build the independent COCO-1800 layer-selection dataset."""

from __future__ import annotations

import csv
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .annotations import (
    load_coco_instances,
    scan_pair_candidates,
    scan_validation_vertical_candidates,
    take_unique,
)
from .config import COCO_VALIDATION_PROTOCOL, CocoValidationConfig
from .pipeline import (
    OPPOSITE,
    _dataset_fingerprint,
    _rows_for_group,
    _sha256,
    _write_csv,
    _write_jsonl,
)
from .validation import validate_preprocessed_coco
from .image_integrity import resolve_image_root


TRAINING_DATASET_ID = "coco_train2017_mask_depth_train6000_v2"
VALIDATION_DATASET_ID = "coco_train2017_mask_depth_validation1800_v2"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_audit(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bbox_iou(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, first)
    bx0, by0, bx1, by1 = map(float, second)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    first_area = (ax1 - ax0) * (ay1 - ay0)
    second_area = (bx1 - bx0) * (by1 - by0)
    return intersection / (first_area + second_area - intersection)


def _bbox_area_ratio(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, first)
    bx0, by0, bx1, by1 = map(float, second)
    ratio = ((ax1 - ax0) * (ay1 - ay0)) / ((bx1 - bx0) * (by1 - by0))
    return max(ratio, 1.0 / ratio)


def _validate_validation_geometry(
    sample: Mapping[str, Any], audit: Mapping[str, str]
) -> None:
    """Validate the frozen geometry for one COCO-1800 row."""
    group = str(sample["group"])
    axis_gap = float(audit["axis_gap"])
    cross_gap = float(audit["cross_gap"])
    boundary_gap = float(audit["boundary_gap"])
    query_box = list(sample["query"]["bbox_xyxy"])
    reference_box = list(sample["reference"]["bbox_xyxy"])
    iou = _bbox_iou(query_box, reference_box)
    area_ratio = _bbox_area_ratio(query_box, reference_box)
    if iou > 0.08:
        raise ValueError(f"Validation sample {sample['sample_id']} exceeds IoU 0.08")
    if group == "horizontal":
        valid = (
            axis_gap >= 0.26
            and cross_gap <= 0.14
            and boundary_gap > 0.0
            and area_ratio <= 2.5
        )
    elif group == "vertical":
        valid = (
            axis_gap >= 0.26
            and cross_gap <= 0.20
            and boundary_gap > 0.0
            and area_ratio <= 2.5
        )
    elif group == "distance":
        valid = axis_gap >= 0.12 and area_ratio <= 4.0
    else:
        raise ValueError(f"Unknown validation group {group!r}")
    if not valid:
        raise ValueError(
            f"Validation sample {sample['sample_id']} violates frozen geometry"
        )


def _load_training_identity(
    training_dataset: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = validate_preprocessed_coco(training_dataset)
    if (
        manifest.get("dataset_id") != TRAINING_DATASET_ID
        or manifest.get("construction", {}).get("split") != "train"
        or manifest.get("counts", {}).get("rows") != 6000
    ):
        raise ValueError("COCO-1800 requires the separate COCO-6000 training dataset")
    rows = _load_rows(training_dataset / manifest["files"]["samples"]["name"])
    return manifest, rows


def validate_coco_validation_dataset(
    directory: Path, training_dataset: Path
) -> dict[str, Any]:
    """Validate COCO-1800 identity, geometry, and training-image disjointness."""
    manifest = validate_preprocessed_coco(directory)
    training_manifest, training_rows = _load_training_identity(training_dataset)
    binding = manifest.get("training_exclusion")
    if (
        manifest.get("dataset_id") != VALIDATION_DATASET_ID
        or manifest.get("construction", {}).get("protocol") != COCO_VALIDATION_PROTOCOL
        or manifest.get("construction", {}).get("split") != "validation"
        or manifest.get("counts", {}).get("rows") != 1800
        or not isinstance(binding, dict)
        or binding.get("dataset_id") != training_manifest["dataset_id"]
        or binding.get("dataset_fingerprint")
        != training_manifest["dataset_fingerprint"]
    ):
        raise ValueError("COCO-1800 manifest differs from the frozen protocol")

    samples = _load_rows(directory / manifest["files"]["samples"]["name"])
    audit = _load_audit(directory / manifest["files"]["audit"]["name"])
    training_images = {int(row["image_id"]) for row in training_rows}
    validation_images = {int(row["image_id"]) for row in samples}
    if training_images & validation_images:
        raise ValueError("COCO-1800 images overlap COCO-6000 training images")
    for sample, record in zip(samples, audit, strict=True):
        if sample["sample_id"] != record["sample_id"]:
            raise ValueError("COCO-1800 sample and audit order differs")
        _validate_validation_geometry(sample, record)
    return manifest


def prepare_coco_validation(config: CocoValidationConfig) -> dict[str, Any]:
    """Build 1,800 validation-only rows disjoint from COCO-6000 (PRE-02).

    Horizontal and distance candidates use the training geometry. Vertical
    candidates use the declared 0.20 cross-axis center-gap limit. Training rows
    are read only to exclude their image IDs and are never copied to the output.
    """
    from .depth import infer_mask_depth

    config.validate()
    training_manifest, training_rows = _load_training_identity(config.training_dataset)
    if (
        Path(training_manifest["source"]["annotation_path"]).resolve()
        != config.annotation_path.resolve()
        or resolve_image_root(config.training_dataset, training_manifest["source"])
        != config.image_root.resolve()
    ):
        raise ValueError("COCO-6000 and COCO-1800 must use the same COCO source")

    images, objects = load_coco_instances(config.annotation_path)
    standard_candidates = scan_pair_candidates(images, objects)
    vertical_candidates = scan_validation_vertical_candidates(images, objects)
    used = {int(row["image_id"]) for row in training_rows}
    per_group = 2 * config.samples_per_endpoint
    vertical = take_unique(vertical_candidates, per_group, used, config.image_root)
    horizontal = take_unique(
        standard_candidates["horizontal"], per_group, used, config.image_root
    )
    distance_pool = take_unique(
        standard_candidates["distance"],
        config.depth_candidate_count,
        used,
        config.image_root,
    )
    depth_metrics = infer_mask_depth(
        distance_pool,
        config.image_root,
        config.depth_model_path,
        config.device,
        config.depth_batch_size,
    )
    accepted_distance = [
        candidate
        for candidate in distance_pool
        if depth_metrics[candidate.image.image_id].confidence >= config.depth_confidence
    ]
    accepted_distance.sort(
        key=lambda candidate: (
            -depth_metrics[candidate.image.image_id].confidence,
            candidate.image.image_id,
        )
    )

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for group, selected, metrics in (
        ("horizontal", horizontal, None),
        ("vertical", vertical, None),
        ("distance", accepted_distance, depth_metrics),
    ):
        group_rows, group_audit = _rows_for_group(
            selected,
            group,
            config.samples_per_endpoint,
            range(config.samples_per_endpoint),
            "validation",
            config.seed,
            metrics,
        )
        rows.extend(group_rows)
        audit.extend(group_audit)

    endpoint_counts = Counter(row["label"] for row in rows)
    expected = Counter({endpoint: config.samples_per_endpoint for endpoint in OPPOSITE})
    if endpoint_counts != expected or len(rows) != config.sample_count:
        raise ValueError(f"COCO-1800 endpoint balance differs: {dict(endpoint_counts)}")
    if len({int(row["image_id"]) for row in rows}) != config.sample_count:
        raise ValueError("COCO-1800 requires one unique image per row")

    config.output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = config.output_dir / "samples.jsonl"
    audit_path = config.output_dir / "audit.csv"
    rejections_path = config.output_dir / "rejections.jsonl"
    _write_jsonl(samples_path, rows)
    _write_csv(audit_path, audit)
    rejected_depth = [
        {
            "image_id": candidate.image.image_id,
            "reason": "depth_confidence_below_threshold",
            "confidence": depth_metrics[candidate.image.image_id].confidence,
        }
        for candidate in distance_pool
        if depth_metrics[candidate.image.image_id].confidence < config.depth_confidence
    ]
    _write_jsonl(rejections_path, rejected_depth)
    image_fingerprint, dataset_fingerprint = _dataset_fingerprint(
        samples_path, audit_path, config.image_root, rows
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": VALIDATION_DATASET_ID,
        "source": {
            "release": "COCO 2017 train instances",
            "annotation_path": str(config.annotation_path),
            "annotation_sha256": _sha256(config.annotation_path),
            "image_root": str(config.image_root),
            "selected_image_fingerprint": image_fingerprint,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "construction": {
            "protocol": config.protocol,
            "seed": config.seed,
            "split": "validation",
            "samples_per_endpoint": config.samples_per_endpoint,
            "sampling_unit": "one unique image and one original QA relation",
            "original_endpoint_balance": True,
            "orientations_per_sample": 2,
            "one_pair_per_image": True,
            "horizontal_vertical_label": "strict_bbox_separation",
            "distance_label": "mean_depth_inside_native_coco_instance_mask",
            "depth_model_path": str(config.depth_model_path),
            "depth_candidate_count": config.depth_candidate_count,
            "depth_confidence_threshold": config.depth_confidence,
            "depth_value_convention": "larger_is_closer",
            "crowd_policy": "reject_iscrowd",
            "vertical_cross_center_gap_max": 0.20,
        },
        "training_exclusion": {
            "dataset_id": training_manifest["dataset_id"],
            "dataset_fingerprint": training_manifest["dataset_fingerprint"],
            "excluded_image_count": len(training_rows),
        },
        "counts": {
            "rows": len(rows),
            "images": len(rows),
            "by_label": dict(sorted(endpoint_counts.items())),
            "by_split": {"validation": len(rows)},
            "orientation_prompts": 2 * len(rows),
            "train_orientation_prompts": 0,
            "validation_orientation_prompts": 2 * len(rows),
            "candidate_by_group": {
                key: len(value) for key, value in standard_candidates.items()
            },
            "vertical_candidates": len(vertical_candidates),
            "depth_accepted": len(accepted_distance),
            "depth_rejected": len(rejected_depth),
        },
        "files": {
            "samples": {"name": samples_path.name, "sha256": _sha256(samples_path)},
            "audit": {"name": audit_path.name, "sha256": _sha256(audit_path)},
            "rejections": {
                "name": rejections_path.name,
                "sha256": _sha256(rejections_path),
            },
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_coco_validation_dataset(config.output_dir, config.training_dataset)

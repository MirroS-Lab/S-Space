"""Orchestrate formal balanced COCO preprocessing stages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import load_coco_instances, scan_pair_candidates, take_unique
from .config import CocoTrainingConfig
from .image_integrity import selected_image_fingerprint
from .schema import DepthMetric, PairCandidate


ENDPOINTS = {
    "horizontal": ("left", "right"),
    "vertical": ("above", "below"),
    "distance": ("far", "close"),
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "far": "close",
    "close": "far",
}
FROZEN_BASE_ROWS_PER_ENDPOINT = 1050
FROZEN_TRAIN_ROWS_PER_ENDPOINT = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(
    samples_path: Path,
    audit_path: Path,
    image_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    """Bind frozen rows, audit records, and selected image bytes (PRE-06).

    The image fingerprint hashes sorted ``image_id:file_sha256`` records. The
    dataset fingerprint then hashes the sample checksum, audit checksum, and
    image fingerprint. This identifies the exact construction dataset without
    copying source images into the output directory.

    Returns:
        ``(image_fingerprint, dataset_fingerprint)`` as SHA-256 hex strings.

    Raises:
        FileNotFoundError: A selected source image is absent.

    Side effects:
        Reads every selected image once.
    """
    image_fingerprint = selected_image_fingerprint(rows, image_root)
    dataset_digest = hashlib.sha256()
    for value in (_sha256(samples_path), _sha256(audit_path), image_fingerprint):
        dataset_digest.update(f"{value}\n".encode())
    return image_fingerprint, dataset_digest.hexdigest()


def _ordered_objects(
    candidate: PairCandidate, group: str, depth: DepthMetric | None
) -> tuple[Any, Any]:
    if group == "horizontal":
        return tuple(
            sorted(
                (candidate.first, candidate.second), key=lambda value: value.center_x
            )
        )
    if group == "vertical":
        return tuple(
            sorted(
                (candidate.first, candidate.second), key=lambda value: value.center_y
            )
        )
    if depth is None:
        raise ValueError("Distance ordering requires mask depth")
    by_id = {
        candidate.first.annotation_id: candidate.first,
        candidate.second.annotation_id: candidate.second,
    }
    return by_id[depth.far_annotation_id], by_id[depth.near_annotation_id]


def _rows_for_group(
    candidates: list[PairCandidate],
    group: str,
    endpoint_pool_size: int,
    endpoint_indices: range,
    split: str,
    seed: int,
    depth_metrics: dict[int, DepthMetric] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign balanced endpoint signs and render original/swapped rows (PRE-05).

    Candidate ordering defines the negative and positive endpoint for an axis.
    The deterministic shuffle decides which physical ordering is rendered as
    the original question, preventing one object role from encoding the label.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"Unknown COCO split {split!r}")
    if not endpoint_indices or endpoint_indices.start < 0:
        raise ValueError("Endpoint indices must be non-empty and nonnegative")
    if endpoint_indices.stop > endpoint_pool_size or endpoint_indices.step != 1:
        raise ValueError("Endpoint indices exceed the frozen candidate pool")
    required = 2 * endpoint_pool_size
    if len(candidates) < required:
        raise ValueError(
            f"Need {required} accepted {group} candidates, found {len(candidates)}"
        )
    selected = candidates[:required]
    random.Random(
        seed + {"horizontal": 1, "vertical": 2, "distance": 3}[group]
    ).shuffle(selected)
    endpoints = ENDPOINTS[group]
    rows, audit = [], []
    for index, candidate in enumerate(selected):
        category_index = index % endpoint_pool_size
        if category_index not in endpoint_indices:
            continue
        label = endpoints[0] if index < endpoint_pool_size else endpoints[1]
        metric = (
            depth_metrics[candidate.image.image_id]
            if depth_metrics is not None
            else None
        )
        negative, positive = _ordered_objects(candidate, group, metric)
        query, reference = (
            (negative, positive) if label == endpoints[0] else (positive, negative)
        )
        depth_by_annotation = (
            {
                candidate.first.annotation_id: metric.first_mean,
                candidate.second.annotation_id: metric.second_mean,
            }
            if metric is not None
            else {}
        )
        rows.append(
            {
                "sample_id": f"coco_train2017_{candidate.image.image_id}_{group}",
                "image_id": candidate.image.image_id,
                "image_path": str(candidate.image.file_name),
                "group": group,
                "label": label,
                "split": split,
                "query": {
                    "name": query.name,
                    "annotation_id": query.annotation_id,
                    "bbox_xyxy": query.bbox,
                },
                "reference": {
                    "name": reference.name,
                    "annotation_id": reference.annotation_id,
                    "bbox_xyxy": reference.bbox,
                },
                "swapped_label": OPPOSITE[label],
            }
        )
        audit.append(
            {
                "sample_id": rows[-1]["sample_id"],
                "image_id": candidate.image.image_id,
                "group": group,
                "label": label,
                "split": split,
                "query": query.name,
                "reference": reference.name,
                "query_annotation_id": query.annotation_id,
                "reference_annotation_id": reference.annotation_id,
                "axis_gap": candidate.axis_gap,
                "cross_gap": candidate.cross_gap,
                "boundary_gap": candidate.boundary_gap,
                "selection_score": candidate.score,
                "depth_first_mean": metric.first_mean if metric else "",
                "depth_second_mean": metric.second_mean if metric else "",
                "depth_normalized_gap": metric.normalized_gap if metric else "",
                "depth_confidence": metric.confidence if metric else "",
                "query_depth_mean": depth_by_annotation.get(query.annotation_id, ""),
                "reference_depth_mean": depth_by_annotation.get(
                    reference.annotation_id, ""
                ),
            }
        )
    return rows, audit


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_coco_training(config: CocoTrainingConfig) -> dict[str, Any]:
    """Build the immutable, balanced COCO-6000 training dataset (PRE-01).

    The output contains 1,000 training rows for each of left, right, above,
    below, far, and close. It contains no validation rows.
    Horizontal and vertical labels come from strict bbox separation. Distance
    labels come from Depth Anything mean depth inside native COCO masks.

    Raises:
        ValueError: Any source, candidate, depth, balance, or uniqueness
            invariant fails.
        FileExistsError: The immutable output already exists.

    Side effects:
        Runs depth inference and writes JSONL, CSV, and JSON outputs.
    """
    from .depth import infer_mask_depth

    config.validate()
    images, objects = load_coco_instances(config.annotation_path)
    candidates = scan_pair_candidates(images, objects)
    used: set[int] = set()
    if config.samples_per_endpoint != FROZEN_TRAIN_ROWS_PER_ENDPOINT:
        raise ValueError("COCO-6000 training count differs from the frozen protocol")
    per_group = 2 * FROZEN_BASE_ROWS_PER_ENDPOINT
    # Vertical candidates are the scarcest under the frozen geometry rules.
    # Allocate them first so cross-axis image uniqueness cannot starve V while
    # the larger horizontal and distance pools still cover their quotas.
    vertical = take_unique(candidates["vertical"], per_group, used, config.image_root)
    horizontal = take_unique(
        candidates["horizontal"], per_group, used, config.image_root
    )
    distance_pool = take_unique(
        candidates["distance"], config.depth_candidate_count, used, config.image_root
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

    rows, audit = [], []
    for group, selected, metrics in (
        ("horizontal", horizontal, None),
        ("vertical", vertical, None),
        ("distance", accepted_distance, depth_metrics),
    ):
        group_rows, group_audit = _rows_for_group(
            selected,
            group,
            FROZEN_BASE_ROWS_PER_ENDPOINT,
            range(FROZEN_TRAIN_ROWS_PER_ENDPOINT),
            "train",
            config.seed,
            metrics,
        )
        rows.extend(group_rows)
        audit.extend(group_audit)
    if len(rows) != config.sample_count or len(
        {row["image_id"] for row in rows}
    ) != len(rows):
        raise ValueError("Output count or one-pair-per-image invariant failed")
    counts = Counter(row["label"] for row in rows)
    if set(counts.values()) != {config.samples_per_endpoint} or set(counts) != set(
        OPPOSITE
    ):
        raise ValueError(f"Output endpoint balance failed: {dict(counts)}")
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != {"train": config.sample_count}:
        raise ValueError(f"Output split balance failed: {dict(split_counts)}")

    config.output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = config.output_dir / "samples.jsonl"
    audit_path = config.output_dir / "audit.csv"
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
    _write_jsonl(config.output_dir / "rejections.jsonl", rejected_depth)
    image_fingerprint, dataset_fingerprint = _dataset_fingerprint(
        samples_path, audit_path, config.image_root, rows
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "coco_train2017_mask_depth_train6000_v2",
        "source": {
            "release": "COCO 2017 train instances",
            "annotation_path": str(config.annotation_path),
            "annotation_sha256": _sha256(config.annotation_path),
            "image_root": os.path.relpath(config.image_root, config.output_dir),
            "image_root_base": "dataset_dir",
            "selected_image_fingerprint": image_fingerprint,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "construction": {
            "protocol": config.protocol,
            "seed": config.seed,
            "split": "train",
            "samples_per_endpoint": config.samples_per_endpoint,
            "sampling_unit": "one unique image and one original QA relation",
            "original_endpoint_balance": True,
            "orientations_per_sample": 2,
            "orientation_protocol": "each original QA is expanded with one query-reference swap during training",
            "prompt_fields_stored": False,
            "prompt_protocol": None,
            "prompt_selection": "explicit_train_configuration_required",
            "one_pair_per_image": True,
            "horizontal_vertical_label": "strict_bbox_separation",
            "distance_label": "mean_depth_inside_native_coco_instance_mask",
            "depth_model_path": str(config.depth_model_path),
            "depth_candidate_count": config.depth_candidate_count,
            "depth_confidence_threshold": config.depth_confidence,
            "depth_value_convention": "larger_is_closer",
            "crowd_policy": "reject_iscrowd",
        },
        "counts": {
            "rows": len(rows),
            "images": len(rows),
            "by_label": dict(sorted(counts.items())),
            "by_split": dict(sorted(split_counts.items())),
            "orientation_prompts": 2 * len(rows),
            "train_orientation_prompts": 2 * config.sample_count,
            "validation_orientation_prompts": 0,
            "candidate_by_group": {
                key: len(value) for key, value in candidates.items()
            },
            "depth_accepted": len(accepted_distance),
            "depth_rejected": len(rejected_depth),
        },
        "files": {
            "samples": {"name": samples_path.name, "sha256": _sha256(samples_path)},
            "audit": {"name": audit_path.name, "sha256": _sha256(audit_path)},
            "rejections": {
                "name": "rejections.jsonl",
                "sha256": _sha256(config.output_dir / "rejections.jsonl"),
            },
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

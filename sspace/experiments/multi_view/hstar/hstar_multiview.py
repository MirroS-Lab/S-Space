"""Strict frozen H* HOS panorama multi-view adapter (EVAL-13)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from sspace.experiments.multi_view.common.schema import (
    GT_POSITION,
    MultiViewSample,
)
from sspace.data.hstar.pairs import (
    DATASET_ID,
    DEPTH_CLOSE_FOV_DEGREES,
    DEPTH_FAR_FOV_DEGREES,
    HORIZONTAL_FOV_DEGREES,
    HORIZONTAL_VIEW_OFFSET_DEGREES,
    LABEL_QUOTAS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PUBLISHED_PAIRS_SHA256,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    VERTICAL_VIEW_OFFSET_DEGREES,
    target_name_from_task,
)
from sspace.run_records import canonical_fingerprint, file_sha256


LABEL_AXIS_SIGN = {
    "left": ("horizontal", -1),
    "right": ("horizontal", 1),
    "up": ("vertical", -1),
    "down": ("vertical", 1),
    "far": ("distance", -1),
    "close": ("distance", 1),
}
ROW_FIELDS = {
    "annotation_index",
    "area_ratio_b_over_a",
    "axis",
    "bbox_a_xyxy",
    "bbox_b_xyxy",
    "center_a_xy",
    "center_b_xy",
    "center_displacement",
    "image_a_path",
    "image_a_sha256",
    "image_b_path",
    "image_b_sha256",
    "label",
    "metadata_path",
    "object_name",
    "output_size",
    "prompt",
    "sample_id",
    "scene_id",
    "source_image_name",
    "source_image_sha256",
    "source_sample_id",
    "source_task",
    "target_pitch_degrees",
    "target_pitch_range",
    "target_yaw_degrees",
    "target_yaw_range",
    "view_a_fov_degrees",
    "view_a_pitch_degrees",
    "view_a_yaw_degrees",
    "view_b_fov_degrees",
    "view_b_pitch_degrees",
    "view_b_yaw_degrees",
    "view_overlap_fraction",
}


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H* JSON root must be an object: {path}")
    return value


def _resolved_file(dataset_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("H* relative path must be a non-empty string")
    root = dataset_dir.resolve()
    path = (dataset_dir / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("H* relative path escapes the frozen dataset")
    return path


def _pair(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"H* {field} must contain two values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"H* {field} is non-finite")
    return result


def _box(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"H* {field} must contain four values")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or not (
        0.0 <= box[0] < box[2] <= OUTPUT_WIDTH
        and 0.0 <= box[1] < box[3] <= OUTPUT_HEIGHT
    ):
        raise ValueError(f"H* {field} is invalid")
    return box


def _validate_image(path: Path, digest: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
            raise ValueError(f"H* image must be 1920x1080 RGB: {path}")
    if file_sha256(path) != digest:
        raise ValueError(f"H* image checksum changed: {path}")


def _validated_review_ids(
    manifest_path: Path,
    manifest: dict[str, Any],
    checksums: dict[str, Any],
) -> tuple[str, ...] | None:
    """Reuse the published audit only for identical rows and image hashes."""
    if (
        manifest.get("dataset_fingerprint") == PUBLISHED_PAIRS_SHA256
        and checksums.get("pairs.jsonl") == PUBLISHED_PAIRS_SHA256
        and file_sha256(manifest_path.parent / "pairs.jsonl") == PUBLISHED_PAIRS_SHA256
    ):
        return None
    review = manifest.get("review")
    if not isinstance(review, dict) or (
        review.get("status") != "passed"
        or not str(review.get("reviewed_by") or "").strip()
        or review.get("artifact_fingerprint") != manifest.get("dataset_fingerprint")
        or review.get("samples_per_label") != 20
        or len(review.get("sample_ids", ())) != 120
        or review.get("review_html_sha256") != checksums["review.html"]
    ):
        raise ValueError("H* visual review has not passed")
    return tuple(str(sample_id) for sample_id in review["sample_ids"])


def load_hstar_multiview(dataset_dir: Path) -> tuple[MultiViewSample, ...]:
    """Load and fully validate the frozen PRE-06 HOS-600 dataset.

    The adapter maps target-center X/Y and log normalized target-box area to
    H/V/D. Thus positive coordinate changes mean right, below, and close.

    Args:
        dataset_dir: Frozen HOS-600 directory.

    Returns:
        Exactly 600 samples in canonical ``pairs.jsonl`` order.

    Raises:
        FileNotFoundError: A required dataset file is absent.
        ValueError: Provenance, review, schema, geometry, quota, or checksum
            evidence differs from PRE-06.

    Side effects:
        Decodes and hashes all 1,200 images; writes nothing.
    """
    manifest_path = dataset_dir / "manifest.json"
    pairs_path = dataset_dir / "pairs.jsonl"
    review_path = dataset_dir / "review.html"
    csv_path = dataset_dir / "manifest.csv"
    checksums = _json_object(dataset_dir / "checksums.json")
    expected_files = {"manifest.json", "pairs.jsonl", "review.html", "manifest.csv"}
    if set(checksums) != expected_files:
        raise ValueError("H* checksums.json fields differ from PRE-06")
    for path in (manifest_path, pairs_path, review_path, csv_path):
        if file_sha256(path) != checksums[path.name]:
            raise ValueError(f"H* top-level checksum changed: {path.name}")
    manifest = _json_object(manifest_path)
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("pipeline_step") != "PRE-06"
        or manifest.get("sample_count") != 600
        or manifest.get("label_counts") != LABEL_QUOTAS
        or manifest.get("dataset_fingerprint") != checksums["pairs.jsonl"]
        or manifest.get("depth_labels") != "apparent_scale_only"
    ):
        raise ValueError("H* manifest identity or counts differ from PRE-06")
    if manifest.get("source") != {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "archive": "hos_train.zip",
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
    }:
        raise ValueError("H* source provenance differs from PRE-06")
    if manifest.get("projection") != {
        "horizontal_fov_degrees": HORIZONTAL_FOV_DEGREES,
        "depth_close_fov_degrees": DEPTH_CLOSE_FOV_DEGREES,
        "depth_far_fov_degrees": DEPTH_FAR_FOV_DEGREES,
        "horizontal_view_offset_degrees": HORIZONTAL_VIEW_OFFSET_DEGREES,
        "vertical_view_offset_degrees": VERTICAL_VIEW_OFFSET_DEGREES,
        "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "crosshair": False,
    }:
        raise ValueError("H* projection protocol differs from PRE-06")
    review_ids = _validated_review_ids(manifest_path, manifest, checksums)

    rows = [
        json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 600 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("H* pairs.jsonl must contain exactly 600 objects")
    samples: list[MultiViewSample] = []
    labels: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    targets: set[tuple[str, int]] = set()
    sample_labels: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) != ROW_FIELDS:
            raise ValueError("H* row fields differ from PRE-06")
        sample_id = f"pair_{index:06d}"
        if row["sample_id"] != sample_id:
            raise ValueError("H* sample order or identity is not canonical")
        label = str(row["label"])
        if label not in LABEL_AXIS_SIGN:
            raise ValueError(f"Unknown H* label {label!r}")
        axis, sign = LABEL_AXIS_SIGN[label]
        if row["axis"] != axis:
            raise ValueError("H* label and axis differ")
        scene = str(row["scene_id"])
        annotation_index = int(row["annotation_index"])
        target_key = (scene, annotation_index)
        if target_key in targets:
            raise ValueError("H* reuses an annotated target")
        targets.add(target_key)
        scenes[scene] += 1
        labels[label] += 1
        sample_labels[sample_id] = label
        object_name = str(row["object_name"])
        if (
            target_name_from_task(str(row["source_task"])) != object_name
            or row["prompt"] != f"In View 1, where is the {object_name}?"
            or row["output_size"] != [OUTPUT_WIDTH, OUTPUT_HEIGHT]
        ):
            raise ValueError("H* target name, prompt, or output size changed")
        center_a = _pair(row["center_a_xy"], "center_a_xy")
        center_b = _pair(row["center_b_xy"], "center_b_xy")
        box_a = _box(row["bbox_a_xyxy"], "bbox_a_xyxy")
        box_b = _box(row["bbox_b_xyxy"], "bbox_b_xyxy")
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        area_ratio = area_b / area_a
        if not math.isclose(
            area_ratio, float(row["area_ratio_b_over_a"]), abs_tol=1e-9
        ):
            raise ValueError("H* target area ratio is inconsistent")
        gt_a = (
            center_a[0] / OUTPUT_WIDTH,
            center_a[1] / OUTPUT_HEIGHT,
            math.log(area_a / (OUTPUT_WIDTH * OUTPUT_HEIGHT)),
        )
        gt_b = (
            center_b[0] / OUTPUT_WIDTH,
            center_b[1] / OUTPUT_HEIGHT,
            math.log(area_b / (OUTPUT_WIDTH * OUTPUT_HEIGHT)),
        )
        gt_delta = tuple(b - a for a, b in zip(gt_a, gt_b, strict=True))
        if sign * gt_delta[GT_POSITION[axis]] <= 0.0:
            raise ValueError("H* endpoint sign differs from rendered geometry")
        metadata_path = _resolved_file(dataset_dir, row["metadata_path"])
        metadata = _json_object(metadata_path)
        if metadata != {
            key: value for key, value in row.items() if key != "metadata_path"
        }:
            raise ValueError("H* row and per-sample metadata differ")
        image_a = _resolved_file(dataset_dir, row["image_a_path"])
        image_b = _resolved_file(dataset_dir, row["image_b_path"])
        image_a_sha = str(row["image_a_sha256"])
        image_b_sha = str(row["image_b_sha256"])
        _validate_image(image_a, image_a_sha)
        _validate_image(image_b, image_b_sha)
        sample = MultiViewSample(
            sample_id=sample_id,
            scene=scene,
            object_id=annotation_index,
            object_name=object_name,
            axis=axis,
            split="hos_train",
            view_a_observation_id=f"{row['source_sample_id']}:a",
            view_b_observation_id=f"{row['source_sample_id']}:b",
            image_a_path=image_a,
            image_b_path=image_b,
            image_a_sha256=image_a_sha,
            image_b_sha256=image_b_sha,
            gt_a_hvd=gt_a,
            gt_b_hvd=gt_b,
            gt_delta_hvd=gt_delta,
            visibility_a=area_a / (OUTPUT_WIDTH * OUTPUT_HEIGHT),
            visibility_b=area_b / (OUTPUT_WIDTH * OUTPUT_HEIGHT),
            grounding_mode="name_only",
            target_label=label,
        )
        sample.validate()
        samples.append(sample)
    if labels != Counter(LABEL_QUOTAS) or max(scenes.values()) > 2:
        raise ValueError("H* class quotas or panorama diversity changed")
    if review_ids is not None:
        if (
            len(set(review_ids)) != 120
            or not set(review_ids).issubset(sample_labels)
            or Counter(sample_labels[sample_id] for sample_id in review_ids)
            != Counter({label: 20 for label in LABEL_QUOTAS})
        ):
            raise ValueError("H* review sample identities differ from rows")
    return tuple(samples)


def hstar_multiview_fingerprint(samples: tuple[MultiViewSample, ...]) -> str:
    """Bind ordered HOS-600 identities, images, labels, and H/V/D geometry."""
    if len(samples) != 600:
        raise ValueError("H* fingerprint requires exactly 600 samples")
    return canonical_fingerprint(
        {
            "protocol": "hstar_hos600_causal_order_swap_v1",
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "scene": sample.scene,
                    "object_id": sample.object_id,
                    "object_name": sample.object_name,
                    "axis": sample.axis,
                    "target_label": sample.endpoint_label,
                    "view_a": sample.view_a_observation_id,
                    "view_b": sample.view_b_observation_id,
                    "image_a_sha256": sample.image_a_sha256,
                    "image_b_sha256": sample.image_b_sha256,
                    "gt_a_hvd": sample.gt_a_hvd,
                    "gt_b_hvd": sample.gt_b_hvd,
                    "gt_delta_hvd": sample.gt_delta_hvd,
                }
                for sample in samples
            ],
        }
    )

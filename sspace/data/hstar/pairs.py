"""Construct overlapping H* HOS panorama view pairs (PRE-06)."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from sspace.run_records import file_sha256

from .panorama import equirectangular_to_perspective, project_spherical_points


DATASET_ID = "hstar_hos_rl_panorama_multiview600_v1"
PUBLISHED_PAIRS_SHA256 = (
    "1ecf508862d180783769dc06f15c26d56d6e8daa17ba4c8a06cbd784d478682f"
)
SOURCE_REPOSITORY = "humanoid-vstar/hvs_rl"
SOURCE_REVISION = "030e612b7d140b659c024464ab50508379c2dcfa"
SOURCE_ARCHIVE_SHA256 = (
    "34a9a4c24695983b2b654e22d1bb1599061615f10e7bc20f7672cd4e0a21de94"
)
LABEL_QUOTAS = {label: 100 for label in ("left", "right", "up", "down", "close", "far")}
HORIZONTAL_FOV_DEGREES = 100.0
HORIZONTAL_VIEW_OFFSET_DEGREES = 28.0
VERTICAL_VIEW_OFFSET_DEGREES = 18.0
DEPTH_CLOSE_FOV_DEGREES = 72.0
DEPTH_FAR_FOV_DEGREES = 110.0
MIN_DEPTH_AREA_RATIO = 2.0
MIN_DEPTH_CENTER_DISPLACEMENT = 0.02
MAX_DEPTH_CENTER_DISPLACEMENT = 0.12
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
MIN_CENTER_DISPLACEMENT = 0.30
FRAME_MARGIN = 0.025
MAX_TARGET_WORDS = 16

_COMMAND_PREFIX = re.compile(
    r"^(?:look\s+for(?:\s+and\s+use)?|locate|search\s+for|"
    r"find(?:\s+and\s+buy)?|buy|grab)\s+",
    flags=re.IGNORECASE,
)
_ARTICLE_PREFIX = re.compile(r"^(?:a|an|the|some)\s+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class HOSPairCandidate:
    """One validated same-target panorama pair before image serialization."""

    scene_id: str
    annotation_index: int
    source_image: Path
    source_task: str
    object_name: str
    label: str
    axis: str
    target_yaw_degrees: float
    target_pitch_degrees: float
    target_yaw_range: tuple[float, float]
    target_pitch_range: tuple[float, float]
    view_a_yaw_degrees: float
    view_a_pitch_degrees: float
    view_a_fov_degrees: float
    view_b_yaw_degrees: float
    view_b_pitch_degrees: float
    view_b_fov_degrees: float
    center_a_xy: tuple[float, float]
    center_b_xy: tuple[float, float]
    bbox_a_xyxy: tuple[float, float, float, float]
    bbox_b_xyxy: tuple[float, float, float, float]
    center_displacement: float
    area_ratio_b_over_a: float
    view_overlap_fraction: float
    quality: float

    @property
    def sample_id(self) -> str:
        """Return a stable identity before presentation numbering."""
        return f"hos-{self.scene_id}-{self.annotation_index}-{self.label}"


def target_name_from_task(task: str) -> str | None:
    """Extract a natural target phrase from one official HOS instruction."""
    clean = " ".join(task.strip().rstrip(".?!").split())
    target = _COMMAND_PREFIX.sub("", clean, count=1)
    if target == clean:
        return None
    target = _ARTICLE_PREFIX.sub("", target, count=1).strip()
    if not target or len(target.split()) > MAX_TARGET_WORDS:
        return None
    return target


def _yaw_center_and_width(bounds: Iterable[float]) -> tuple[float, float]:
    start, end = (float(value) % 360.0 for value in bounds)
    width = (end - start) % 360.0
    if not 0.0 < width < 180.0:
        raise ValueError("HOS target yaw interval must be in (0,180) degrees")
    return (start + width / 2.0) % 360.0, width


def _target_projection(
    yaw_range: tuple[float, float],
    pitch_range: tuple[float, float],
    *,
    view_yaw: float,
    view_pitch: float,
    horizontal_fov: float,
) -> tuple[tuple[float, float], tuple[float, float, float, float]] | None:
    center_yaw, width = _yaw_center_and_width(yaw_range)
    center_pitch = (pitch_range[0] + pitch_range[1]) / 2.0
    yaw_samples = (center_yaw + np.linspace(-width / 2.0, width / 2.0, 7)) % 360.0
    pitch_samples = np.linspace(pitch_range[0], pitch_range[1], 7)
    yaw_grid, pitch_grid = np.meshgrid(yaw_samples, pitch_samples)
    projected = project_spherical_points(
        yaw_grid,
        pitch_grid,
        view_yaw_degrees=view_yaw,
        view_pitch_degrees=view_pitch,
        horizontal_fov_degrees=horizontal_fov,
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
    )
    center = project_spherical_points(
        np.asarray(center_yaw),
        np.asarray(center_pitch),
        view_yaw_degrees=view_yaw,
        view_pitch_degrees=view_pitch,
        horizontal_fov_degrees=horizontal_fov,
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
    )
    if not np.isfinite(projected).all() or not np.isfinite(center).all():
        return None
    minimum = projected.reshape(-1, 2).min(axis=0)
    maximum = projected.reshape(-1, 2).max(axis=0)
    margin = np.asarray((FRAME_MARGIN * OUTPUT_WIDTH, FRAME_MARGIN * OUTPUT_HEIGHT))
    if np.any(minimum < margin) or np.any(
        maximum > np.asarray((OUTPUT_WIDTH, OUTPUT_HEIGHT)) - margin
    ):
        return None
    bbox = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    return (float(center[0]), float(center[1])), bbox


def _ordered_candidate(
    *,
    scene_id: str,
    annotation_index: int,
    source_image: Path,
    task: str,
    object_name: str,
    yaw_range: tuple[float, float],
    pitch_range: tuple[float, float],
    axis: str,
    view_a: tuple[float, float, float],
    view_b: tuple[float, float, float],
) -> HOSPairCandidate | None:
    projection_a = _target_projection(
        yaw_range,
        pitch_range,
        view_yaw=view_a[0],
        view_pitch=view_a[1],
        horizontal_fov=view_a[2],
    )
    projection_b = _target_projection(
        yaw_range,
        pitch_range,
        view_yaw=view_b[0],
        view_pitch=view_b[1],
        horizontal_fov=view_b[2],
    )
    if projection_a is None or projection_b is None:
        return None
    center_a, bbox_a = projection_a
    center_b, bbox_b = projection_b
    delta = (
        (center_b[0] - center_a[0]) / OUTPUT_WIDTH,
        (center_b[1] - center_a[1]) / OUTPUT_HEIGHT,
    )
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    area_ratio = area_b / area_a
    if axis == "distance":
        center_displacement = float(np.linalg.norm(delta))
        if not (
            MIN_DEPTH_CENTER_DISPLACEMENT
            <= center_displacement
            <= MAX_DEPTH_CENTER_DISPLACEMENT
        ):
            return None
        if max(area_ratio, 1.0 / area_ratio) < MIN_DEPTH_AREA_RATIO:
            return None
        label = "close" if area_ratio > 1.0 else "far"
        component = math.log(area_ratio)
        overlap = min(view_a[2], view_b[2]) / max(view_a[2], view_b[2])
    elif axis == "horizontal":
        component = delta[0]
        if abs(component) < MIN_CENTER_DISPLACEMENT or abs(delta[1]) > 0.08:
            return None
        center_displacement = abs(component)
        label = "right" if component > 0.0 else "left"
        angular_separation = 2.0 * HORIZONTAL_VIEW_OFFSET_DEGREES
        view_span = HORIZONTAL_FOV_DEGREES
        overlap = 1.0 - angular_separation / float(view_span)
    else:
        component = delta[1]
        if abs(component) < MIN_CENTER_DISPLACEMENT or abs(delta[0]) > 0.08:
            return None
        center_displacement = abs(component)
        label = "down" if component > 0.0 else "up"
        angular_separation = 2.0 * VERTICAL_VIEW_OFFSET_DEGREES
        view_span = np.degrees(
            2.0
            * np.arctan(
                OUTPUT_HEIGHT
                / OUTPUT_WIDTH
                * np.tan(np.radians(HORIZONTAL_FOV_DEGREES) / 2.0)
            )
        )
        overlap = 1.0 - angular_separation / float(view_span)
    target_side = min(
        (bbox_a[2] - bbox_a[0]) / OUTPUT_WIDTH,
        (bbox_a[3] - bbox_a[1]) / OUTPUT_HEIGHT,
        (bbox_b[2] - bbox_b[0]) / OUTPUT_WIDTH,
        (bbox_b[3] - bbox_b[1]) / OUTPUT_HEIGHT,
    )
    if target_side < 0.008:
        return None
    target_yaw, _ = _yaw_center_and_width(yaw_range)
    target_pitch = (pitch_range[0] + pitch_range[1]) / 2.0
    quality = abs(component) + min(target_side, 0.15) + overlap
    return HOSPairCandidate(
        scene_id=scene_id,
        annotation_index=annotation_index,
        source_image=source_image,
        source_task=task,
        object_name=object_name,
        label=label,
        axis=axis,
        target_yaw_degrees=target_yaw,
        target_pitch_degrees=target_pitch,
        target_yaw_range=yaw_range,
        target_pitch_range=pitch_range,
        view_a_yaw_degrees=view_a[0] % 360.0,
        view_a_pitch_degrees=view_a[1],
        view_a_fov_degrees=view_a[2],
        view_b_yaw_degrees=view_b[0] % 360.0,
        view_b_pitch_degrees=view_b[1],
        view_b_fov_degrees=view_b[2],
        center_a_xy=center_a,
        center_b_xy=center_b,
        bbox_a_xyxy=bbox_a,
        bbox_b_xyxy=bbox_b,
        center_displacement=center_displacement,
        area_ratio_b_over_a=area_ratio,
        view_overlap_fraction=overlap,
        quality=quality,
    )


def enumerate_hos_candidates(source_dir: Path) -> list[HOSPairCandidate]:
    """Enumerate PRE-06 candidates from the official extracted HOS directory.

    Only exact 2:1 panoramas are accepted. Each target produces both
    presentation orders for horizontal, vertical, and scale-depth changes.

    Raises:
        FileNotFoundError: ``source_dir`` does not exist.
        ValueError: A scene has an invalid official annotation structure.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    candidates: list[HOSPairCandidate] = []
    for scene in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not scene.is_dir():
            continue
        annotation_path = scene / "annotation.json"
        images = sorted((*scene.glob("*.jpg"), *scene.glob("*.png")))
        if not annotation_path.exists() or len(images) != 1:
            raise ValueError(f"Invalid HOS scene structure: {scene}")
        with Image.open(images[0]) as image:
            if image.width != 2 * image.height:
                continue
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotations, list):
            raise ValueError(f"HOS annotation must be a list: {annotation_path}")
        for annotation_index, row in enumerate(annotations):
            if not {"task", "yaw", "pitch"} <= set(row):
                raise ValueError(f"HOS target fields are missing: {annotation_path}")
            task = str(row["task"])
            object_name = target_name_from_task(task)
            if object_name is None:
                continue
            yaw_range = tuple(float(value) for value in row["yaw"])
            pitch_range = tuple(float(value) for value in row["pitch"])
            if len(yaw_range) != 2 or len(pitch_range) != 2:
                raise ValueError(f"HOS target interval is invalid: {annotation_path}")
            target_yaw, yaw_width = _yaw_center_and_width(yaw_range)
            target_pitch = sum(pitch_range) / 2.0
            pitch_height = pitch_range[1] - pitch_range[0]
            if not (
                1.5 <= yaw_width <= 55.0
                and 1.5 <= pitch_height <= 55.0
                and -60.0 <= target_pitch <= 60.0
            ):
                continue
            canonical_views = {
                "horizontal": (
                    (
                        target_yaw + HORIZONTAL_VIEW_OFFSET_DEGREES,
                        target_pitch,
                        HORIZONTAL_FOV_DEGREES,
                    ),
                    (
                        target_yaw - HORIZONTAL_VIEW_OFFSET_DEGREES,
                        target_pitch,
                        HORIZONTAL_FOV_DEGREES,
                    ),
                ),
                "vertical": (
                    (
                        target_yaw,
                        target_pitch + VERTICAL_VIEW_OFFSET_DEGREES,
                        HORIZONTAL_FOV_DEGREES,
                    ),
                    (
                        target_yaw,
                        target_pitch - VERTICAL_VIEW_OFFSET_DEGREES,
                        HORIZONTAL_FOV_DEGREES,
                    ),
                ),
            }
            digest = hashlib.sha256(
                f"{scene.name}:{annotation_index}".encode()
            ).digest()
            base_yaw = (-1.0 if digest[0] & 1 else 1.0) * (
                2.0 + digest[1] / 255.0 * 3.0
            )
            base_pitch = (-1.0 if digest[2] & 1 else 1.0) * (
                1.0 + digest[3] / 255.0 * 2.0
            )
            shift_yaw = (-1.0 if digest[4] & 1 else 1.0) * (
                4.0 + digest[5] / 255.0 * 4.0
            )
            shift_pitch = (-1.0 if digest[6] & 1 else 1.0) * (
                2.0 + digest[7] / 255.0 * 3.0
            )
            close_view = (
                target_yaw + base_yaw,
                target_pitch + base_pitch,
                DEPTH_CLOSE_FOV_DEGREES,
            )
            far_view = (
                close_view[0] + shift_yaw,
                close_view[1] + shift_pitch,
                DEPTH_FAR_FOV_DEGREES,
            )
            canonical_views["distance"] = (close_view, far_view)
            for axis, (first, second) in canonical_views.items():
                for view_a, view_b in ((first, second), (second, first)):
                    candidate = _ordered_candidate(
                        scene_id=scene.name,
                        annotation_index=annotation_index,
                        source_image=images[0],
                        task=task,
                        object_name=object_name,
                        yaw_range=yaw_range,
                        pitch_range=pitch_range,
                        axis=axis,
                        view_a=view_a,
                        view_b=view_b,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def select_balanced_hos_pairs(
    candidates: Iterable[HOSPairCandidate],
) -> list[HOSPairCandidate]:
    """Select 600 balanced rows without reusing one annotated target."""
    ordered = sorted(
        candidates,
        key=lambda row: (
            -row.quality,
            hashlib.sha256(row.sample_id.encode()).hexdigest(),
        ),
    )
    selected: list[HOSPairCandidate] = []
    counts: Counter[str] = Counter()
    used_targets: set[tuple[str, int]] = set()
    scene_counts: Counter[str] = Counter()
    for row in ordered:
        target_key = (row.scene_id, row.annotation_index)
        if (
            target_key in used_targets
            or scene_counts[row.scene_id] >= 2
            or counts[row.label] >= LABEL_QUOTAS[row.label]
        ):
            continue
        selected.append(row)
        used_targets.add(target_key)
        scene_counts[row.scene_id] += 1
        counts[row.label] += 1
        if counts == Counter(LABEL_QUOTAS):
            break
    if counts != Counter(LABEL_QUOTAS):
        raise ValueError(f"Insufficient balanced HOS candidates: {dict(counts)}")
    return sorted(selected, key=lambda row: row.sample_id)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_review(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    chosen = []
    for label in LABEL_QUOTAS:
        chosen.extend([row for row in rows if row["label"] == label][:20])
    cards = []
    for row in chosen:
        cards.append(
            "<article><h3>"
            + html.escape(f"{row['sample_id']} · {row['label']} · {row['object_name']}")
            + "</h3><div><img src='"
            + html.escape(row["image_a_path"])
            + "'><img src='"
            + html.escape(row["image_b_path"])
            + "'></div><pre>"
            + html.escape(_canonical_json(row))
            + "</pre></article>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>H* HOS PRE-06 review</title>"
        "<style>body{font-family:sans-serif}img{width:46%;margin:1%}"
        "article{border-bottom:1px solid #aaa}pre{white-space:pre-wrap}</style>"
        + "".join(cards)
    )
    (output_dir / "review.html").write_text(document, encoding="utf-8")
    return [str(row["sample_id"]) for row in chosen]


def finalize_hos_dataset(
    output_dir: Path,
    *,
    source_archive_sha256: str,
) -> dict[str, Any]:
    """Verify rendered PRE-06 rows and publish review metadata and checksums.

    Args:
        output_dir: Directory containing complete ``pairs.jsonl``, CSV, metadata,
            and rendered images.
        source_archive_sha256: Actual verified SHA-256 of ``hos_train.zip``.

    Returns:
        The published manifest dictionary.

    Raises:
        FileNotFoundError: A required index, metadata file, or image is absent.
        ValueError: Rows, quotas, identities, or image hashes are invalid.

    Side effects:
        Writes ``review.html``, ``manifest.json``, and ``checksums.json``.
    """
    if source_archive_sha256 != SOURCE_ARCHIVE_SHA256:
        raise ValueError("H* source archive differs from the pinned SHA-256")
    pairs_path = output_dir / "pairs.jsonl"
    csv_path = output_dir / "manifest.csv"
    if not pairs_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("PRE-06 rendered indexes are incomplete")
    pairs_text = pairs_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in pairs_text.splitlines()]
    if Counter(row["label"] for row in records) != Counter(LABEL_QUOTAS):
        raise ValueError("Rendered PRE-06 rows do not match frozen quotas")
    if max(Counter(str(row["scene_id"]) for row in records).values()) > 2:
        raise ValueError("Rendered PRE-06 exceeds the per-panorama limit")
    target_keys = {
        (str(row["scene_id"]), int(row["annotation_index"])) for row in records
    }
    if len(target_keys) != len(records):
        raise ValueError("Rendered PRE-06 reuses an annotated target")
    expected_ids = [f"pair_{index:06d}" for index in range(1, 601)]
    if [row["sample_id"] for row in records] != expected_ids:
        raise ValueError("Rendered PRE-06 sample IDs are not canonical")
    for row in records:
        metadata_path = output_dir / str(row["metadata_path"])
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        for side in ("a", "b"):
            image_path = output_dir / str(row[f"image_{side}_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if file_sha256(image_path) != row[f"image_{side}_sha256"]:
                raise ValueError(f"Rendered PRE-06 image changed: {image_path}")
    review_ids = _write_review(output_dir, records)
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "pipeline_step": "PRE-06",
        "sample_count": len(records),
        "label_counts": dict(Counter(row["label"] for row in records)),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "archive": "hos_train.zip",
            "archive_sha256": source_archive_sha256,
        },
        "projection": {
            "horizontal_fov_degrees": HORIZONTAL_FOV_DEGREES,
            "depth_close_fov_degrees": DEPTH_CLOSE_FOV_DEGREES,
            "depth_far_fov_degrees": DEPTH_FAR_FOV_DEGREES,
            "horizontal_view_offset_degrees": HORIZONTAL_VIEW_OFFSET_DEGREES,
            "vertical_view_offset_degrees": VERTICAL_VIEW_OFFSET_DEGREES,
            "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "crosshair": False,
        },
        "depth_labels": "apparent_scale_only",
        "dataset_fingerprint": hashlib.sha256(pairs_text.encode()).hexdigest(),
        "review": {
            "status": "pending",
            "reviewed_by": None,
            "artifact_fingerprint": None,
            "samples_per_label": 20,
            "sample_ids": review_ids,
            "review_html_sha256": file_sha256(output_dir / "review.html"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    checksums = {
        path.name: file_sha256(path)
        for path in (manifest_path, pairs_path, csv_path, output_dir / "review.html")
    }
    (output_dir / "checksums.json").write_text(
        _canonical_json(checksums) + "\n", encoding="utf-8"
    )
    return manifest


def finalize_hos_review(
    dataset_dir: Path, reviewed_by: str, expected_dataset_fingerprint: str
) -> dict[str, Any]:
    """Mark an unchanged H* artifact reviewed against its exact fingerprint."""
    if not reviewed_by.strip():
        raise ValueError("H* review requires a reviewer")
    manifest_path = dataset_dir / "manifest.json"
    pairs_path = dataset_dir / "pairs.jsonl"
    review_path = dataset_dir / "review.html"
    csv_path = dataset_dir / "manifest.csv"
    checksums_path = dataset_dir / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for path in (manifest_path, pairs_path, review_path, csv_path):
        if file_sha256(path) != checksums.get(path.name):
            raise ValueError(f"H* checksum changed before review: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = file_sha256(pairs_path)
    if (
        expected_dataset_fingerprint != fingerprint
        or manifest.get("dataset_fingerprint") != fingerprint
    ):
        raise ValueError("H* review fingerprint differs from the artifact")
    root = dataset_dir.resolve()
    for row in map(json.loads, pairs_path.read_text(encoding="utf-8").splitlines()):
        for side in ("a", "b"):
            image = (root / str(row[f"image_{side}_path"])).resolve()
            if (
                not image.is_relative_to(root)
                or file_sha256(image) != row[f"image_{side}_sha256"]
            ):
                raise ValueError("H* image changed before review")
    if manifest.get("review", {}).get("status") != "pending":
        raise ValueError("H* review can finalize only a pending artifact")
    manifest["review"].update(
        status="passed",
        reviewed_by=reviewed_by,
        artifact_fingerprint=fingerprint,
    )
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    checksums["manifest.json"] = file_sha256(manifest_path)
    checksums_path.write_text(_canonical_json(checksums) + "\n", encoding="utf-8")
    return manifest


def write_hos_dataset(
    selected: Iterable[HOSPairCandidate],
    output_dir: Path,
    *,
    source_archive: Path,
) -> dict[str, Any]:
    """Render and serialize the immutable 600-row PRE-06 dataset.

    Raises:
        FileExistsError: ``output_dir`` is non-empty.
        ValueError: Quotas, review state, source images, or geometry are invalid.

    Side effects:
        Reads official HOS panoramas and writes images, metadata, indexes,
        checksums, a review page, and a manifest below ``output_dir``.
    """
    from .acquisition import verify_hstar_archive

    source_archive_sha256 = verify_hstar_archive(source_archive, SOURCE_ARCHIVE_SHA256)
    rows = list(selected)
    if Counter(row.label for row in rows) != Counter(LABEL_QUOTAS):
        raise ValueError("Selected HOS rows do not match frozen quotas")
    if max(Counter(row.scene_id for row in rows).values()) > 2:
        raise ValueError("PRE-06 permits at most two targets per HOS panorama")
    if len({(row.scene_id, row.annotation_index) for row in rows}) != len(rows):
        raise ValueError("PRE-06 cannot reuse one annotated HOS target")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows, start=1):
        with Image.open(candidate.source_image) as source:
            source_rgb = np.asarray(source.convert("RGB"))
        pair_id = f"pair_{index:06d}"
        image_paths = []
        image_hashes = []
        for side, yaw, pitch, field_of_view in (
            (
                "a",
                candidate.view_a_yaw_degrees,
                candidate.view_a_pitch_degrees,
                candidate.view_a_fov_degrees,
            ),
            (
                "b",
                candidate.view_b_yaw_degrees,
                candidate.view_b_pitch_degrees,
                candidate.view_b_fov_degrees,
            ),
        ):
            rendered = equirectangular_to_perspective(
                source_rgb,
                horizontal_fov_degrees=field_of_view,
                yaw_degrees=yaw,
                pitch_degrees=pitch,
                width=OUTPUT_WIDTH,
                height=OUTPUT_HEIGHT,
            )
            relative = Path("images") / f"{pair_id}_{side}.jpg"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered).save(destination, quality=95, subsampling=0)
            image_paths.append(relative.as_posix())
            image_hashes.append(file_sha256(destination))
        record = {
            **{
                key: value
                for key, value in asdict(candidate).items()
                if key not in {"source_image", "quality"}
            },
            "sample_id": pair_id,
            "source_sample_id": candidate.sample_id,
            "source_image_name": candidate.source_image.name,
            "source_image_sha256": file_sha256(candidate.source_image),
            "image_a_path": image_paths[0],
            "image_b_path": image_paths[1],
            "image_a_sha256": image_hashes[0],
            "image_b_sha256": image_hashes[1],
            "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "prompt": f"In View 1, where is the {candidate.object_name}?",
        }
        metadata_path = output_dir / "meta" / f"{pair_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(_canonical_json(record) + "\n", encoding="utf-8")
        record["metadata_path"] = metadata_path.relative_to(output_dir).as_posix()
        records.append(record)

    pairs_text = "".join(_canonical_json(row) + "\n" for row in records)
    pairs_path = output_dir / "pairs.jsonl"
    pairs_path.write_text(pairs_text, encoding="utf-8")
    csv_path = output_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "scene_id",
                "annotation_index",
                "object_name",
                "label",
                "axis",
                "image_a_path",
                "image_b_path",
                "metadata_path",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in records
        )
    return finalize_hos_dataset(output_dir, source_archive_sha256=source_archive_sha256)

"""Acquire official COCO images required by the independent validation set."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .annotations import (
    load_coco_instances,
    scan_pair_candidates,
    scan_validation_vertical_candidates,
)
from .config import CocoValidationConfig
from .validation_dataset import TRAINING_DATASET_ID
from .schema import CocoImage, PairCandidate
from .validation import validate_preprocessed_coco

COCO_TRAIN2017_IMAGE_URL = "https://images.cocodataset.org/train2017"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _take_without_local_check(
    candidates: Sequence[PairCandidate], count: int, used: set[int]
) -> list[PairCandidate]:
    selected = []
    for candidate in candidates:
        image_id = candidate.image.image_id
        if image_id in used:
            continue
        selected.append(candidate)
        used.add(image_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} unique candidates, found {len(selected)}")


def _allocate_required_candidates(
    candidates: Mapping[str, Sequence[PairCandidate]],
    vertical_candidates: Sequence[PairCandidate],
    parent_image_ids: set[int],
    samples_per_endpoint: int,
    depth_candidate_count: int,
) -> tuple[tuple[str, PairCandidate], ...]:
    """Allocate COCO-1800 V/H/D pools before checking image presence."""
    used = set(parent_image_ids)
    selected = []
    per_group = 2 * samples_per_endpoint
    for group, values, count in (
        ("vertical", vertical_candidates, per_group),
        ("horizontal", candidates["horizontal"], per_group),
        ("distance", candidates["distance"], depth_candidate_count),
    ):
        selected.extend(
            (group, candidate)
            for candidate in _take_without_local_check(values, count, used)
        )
    return tuple(selected)


def required_validation_images(
    config: CocoValidationConfig,
) -> tuple[tuple[str, CocoImage], ...]:
    """Return the exact V/H/D image pools needed before depth filtering.

    Args:
        config: Validated frozen COCO-1800 construction configuration.

    Returns:
        Ordered ``(group, image)`` pairs: 600 V, 600 H, then 1,400 D.

    Raises:
        ValueError: Parent identity, config values, or candidate counts differ.
        FileNotFoundError: A required immutable config input is absent.

    Side effects:
        Reads the parent samples and official COCO annotation file; it does not
        inspect image presence, infer depth, assign labels, or write files.
    """
    config.validate(require_output_absent=False)
    parent_manifest = validate_preprocessed_coco(config.training_dataset)
    if (
        parent_manifest.get("dataset_id") != TRAINING_DATASET_ID
        or parent_manifest.get("construction", {}).get("split") != "train"
        or parent_manifest.get("counts", {}).get("rows") != 6000
    ):
        raise ValueError("COCO-1800 acquisition requires COCO-6000 training data")
    parent_samples = (
        config.training_dataset / parent_manifest["files"]["samples"]["name"]
    )
    parent_image_ids = {
        int(json.loads(line)["image_id"])
        for line in parent_samples.read_text(encoding="utf-8").splitlines()
    }
    images, objects = load_coco_instances(config.annotation_path)
    candidates = scan_pair_candidates(images, objects)
    vertical = scan_validation_vertical_candidates(images, objects)
    allocated = _allocate_required_candidates(
        candidates,
        vertical,
        parent_image_ids,
        config.samples_per_endpoint,
        config.depth_candidate_count,
    )
    return tuple((group, candidate.image) for group, candidate in allocated)


def _validate_image(path: Path, expected: CocoImage) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        if image.size != (expected.width, expected.height):
            raise ValueError(
                f"COCO image {expected.image_id} size {image.size} != "
                f"{(expected.width, expected.height)}"
            )


def _acquire_coco_images(
    required: Sequence[tuple[str, CocoImage]],
    image_root: Path,
    temporary_directory: Path,
    workers: int,
    expected_sha256_by_image: Mapping[int, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Download and verify one explicitly allocated COCO image pool.

    Args:
        required: Ordered ``(group, image)`` pairs with unique image IDs.
        image_root: Destination for official COCO image files.
        temporary_directory: Staging directory on the same filesystem.
        workers: Positive number of concurrent downloads.

    Returns:
        Ordered per-image records with source URL, byte count, and SHA-256.

    Raises:
        ValueError: Inputs, HTTP metadata, image dimensions, or filesystem
            placement differ from the declared acquisition protocol.

    Side effects:
        Verifies existing images and atomically commits missing official files.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    if len({image.image_id for _, image in required}) != len(required):
        raise ValueError("COCO acquisition image IDs must be unique")
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    if temporary_directory.stat().st_dev != image_root.stat().st_dev:
        raise ValueError("Temporary directory and COCO image root must share a device")

    def acquire(item: tuple[str, CocoImage]) -> dict[str, Any]:
        group, image = item
        destination = image_root / image.file_name
        url = f"{COCO_TRAIN2017_IMAGE_URL}/{image.file_name}"
        if destination.exists():
            status = "existing_verified"
        else:
            temporary = temporary_directory / f"{image.file_name}.part"
            with urllib.request.urlopen(url, timeout=120) as response:
                if response.status != 200:
                    raise ValueError(f"COCO download returned HTTP {response.status}")
                content_length = response.headers.get("Content-Length")
                if content_length is None or int(content_length) <= 0:
                    raise ValueError("COCO download lacks a positive Content-Length")
                expected_bytes = int(content_length)
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if temporary.stat().st_size != expected_bytes:
                raise ValueError(
                    f"COCO download size {temporary.stat().st_size} != "
                    f"Content-Length {expected_bytes}"
                )
            _validate_image(temporary, image)
            try:
                os.link(temporary, destination)
                status = "downloaded_official"
            except FileExistsError:
                status = "concurrent_existing_verified"
            temporary.unlink()
        _validate_image(destination, image)
        digest = _sha256(destination)
        expected_digest = (expected_sha256_by_image or {}).get(image.image_id)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"COCO image SHA-256 changed for {image.image_id}")
        return {
            "group": group,
            "image_id": image.image_id,
            "file_name": image.file_name,
            "url": url,
            "status": status,
            "size": destination.stat().st_size,
            "sha256": digest,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = []
        for index, record in enumerate(executor.map(acquire, required), start=1):
            records.append(record)
            if index == 1 or index % 100 == 0 or index == len(required):
                print(f"COCO images {index}/{len(required)}", flush=True)
        return tuple(records)


def acquire_validation_images(
    config: CocoValidationConfig,
    temporary_directory: Path,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    """Download missing COCO-1800 images from the official COCO URL (PRE-00).

    Existing images are verified and never overwritten. A missing image is
    downloaded to the declared project temporary directory, verified against
    its annotation dimensions, then committed as a hard link so an existing
    destination cannot be replaced.

    Args:
        config: Validated COCO-1800 construction configuration.
        temporary_directory: Project-owned staging directory on the same
            filesystem as the image cache.
        workers: Positive number of concurrent verification/download workers.

    Returns:
        One ordered provenance record per required image, including group,
        image ID, official URL, status, byte count, and SHA-256.

    Raises:
        ValueError: Worker count, filesystem, HTTP response, image dimensions,
            candidate counts, or frozen protocol values differ.
        FileNotFoundError: A required immutable input or committed image is absent.

    Side effects:
        Verifies existing images. Missing images are staged, fsynced, verified,
        hard-linked without replacement into the COCO cache, then unstaged.
    """
    required = required_validation_images(config)
    records = _acquire_coco_images(
        required, config.image_root, temporary_directory, workers
    )
    expected = 4 * config.samples_per_endpoint + config.depth_candidate_count
    if len(records) != expected:
        raise ValueError("COCO-1800 acquisition record count differs")
    return records

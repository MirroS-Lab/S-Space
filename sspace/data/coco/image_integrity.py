"""Decode and fingerprint immutable COCO source images."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


IMAGE_VALIDATION_WORKERS = 16


def resolve_image_root(dataset_dir: Path, source: Mapping[str, Any]) -> Path:
    """Resolve portable dataset-relative roots and legacy absolute manifests."""
    value = source.get("image_root")
    if not isinstance(value, str) or not value:
        raise ValueError("COCO manifest image_root must be a non-empty string")
    if source.get("image_root_base") == "dataset_dir":
        return (dataset_dir / value).resolve()
    return Path(value).resolve()


def _decode_and_hash(path: Path) -> str:
    """Return SHA-256 only after a complete strict pixel decode."""
    if not path.is_file():
        raise FileNotFoundError(path)
    encoded = path.read_bytes()
    with Image.open(BytesIO(encoded)) as image:
        image.load()
    return hashlib.sha256(encoded).hexdigest()


def selected_image_fingerprint(
    rows: Sequence[Mapping[str, Any]], image_root: Path
) -> str:
    """Strictly decode and fingerprint selected dataset images (PRE-06).

    Args:
        rows: Non-empty records containing unique ``image_id`` and
            ``image_path`` fields.
        image_root: Directory containing the declared image files.

    Returns:
        SHA-256 of sorted ``image_id:file_sha256`` records.

    Raises:
        ValueError: Rows are empty, duplicated, or lack required fields.
        FileNotFoundError: A selected image is absent.
        OSError: An image cannot be decoded completely in strict mode.

    Side effects:
        Reads source images with a fixed-size worker pool; writes nothing.
    """
    ordered = sorted(rows, key=lambda value: int(value["image_id"]))
    if not ordered:
        raise ValueError("Selected image fingerprint requires at least one row")
    image_ids = [int(row["image_id"]) for row in ordered]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Selected image fingerprint requires unique image IDs")

    def record(row: Mapping[str, Any]) -> str:
        image_id = int(row["image_id"])
        root = image_root.resolve()
        image_path = (root / str(row["image_path"])).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("COCO image path escapes image_root")
        return f"{image_id}:{_decode_and_hash(image_path)}\n"

    digest = hashlib.sha256()
    with ThreadPoolExecutor(
        max_workers=min(IMAGE_VALIDATION_WORKERS, len(ordered))
    ) as executor:
        for value in executor.map(record, ordered):
            digest.update(value.encode())
    return digest.hexdigest()

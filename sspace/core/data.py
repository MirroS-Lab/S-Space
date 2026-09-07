"""Load validated construction rows used by formal S-Space training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sspace.data.coco.image_integrity import resolve_image_root
from sspace.data.coco.validation import validate_preprocessed_coco


@dataclass(frozen=True)
class ConstructionSample:
    """One frozen image/object relation from the construction dataset."""

    sample_id: str
    image_id: int
    image_path: Path
    group: str
    label: str
    split: str
    query: str
    reference: str


def load_construction_split(
    directory: Path, split: str
) -> tuple[dict, list[ConstructionSample]]:
    """Validate a frozen dataset and return one declared split (TRN-02).

    Args:
        directory: PRE-06 output directory.
        split: Exact split name, either ``train`` or ``validation``.

    Returns:
        The validated manifest and ordered construction samples.

    Raises:
        ValueError: The split is unknown or contains no rows.
        FileNotFoundError: A declared dataset or image file is absent.

    Side effects:
        Reads and validates the complete preprocessing artifact.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported construction split {split!r}")
    manifest = validate_preprocessed_coco(directory)
    sample_path = directory / manifest["files"]["samples"]["name"]
    image_root = resolve_image_root(directory, manifest["source"])
    rows = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["split"] != split:
            continue
        image_path = image_root / value["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            ConstructionSample(
                sample_id=value["sample_id"],
                image_id=int(value["image_id"]),
                image_path=image_path,
                group=value["group"],
                label=value["label"],
                split=value["split"],
                query=value["query"]["name"],
                reference=value["reference"]["name"],
            )
        )
    if not rows:
        raise ValueError(f"Construction split {split!r} is empty")
    return manifest, rows

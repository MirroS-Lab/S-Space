"""Parse official COCO instances and select geometric pair candidates."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .schema import CocoImage, CocoObject, PairCandidate


GROUPS = ("horizontal", "vertical", "distance")


def load_coco_instances(
    path: Path,
) -> tuple[dict[int, CocoImage], dict[int, list[CocoObject]]]:
    """Load usable COCO objects while retaining native polygons (PRE-01).

    COCO bounding boxes use ``[x, y, width, height]``. This function converts
    them to ``[x0, y0, x1, y1]``. Crowd RLE annotations, repeated categories,
    and small or dominant boxes are excluded by the declared construction
    protocol.

    Args:
        path: Official ``instances_train2017.json`` file.

    Returns:
        Image metadata and usable instances keyed by image ID.

    Raises:
        ValueError: Categories are inconsistent or a retained segmentation is
            not an official polygon list.

    Side effects:
        Reads the annotation file.
    """
    dataset = json.loads(path.read_text(encoding="utf-8"))
    category_names = {int(row["id"]): str(row["name"]) for row in dataset["categories"]}
    images = {
        int(row["id"]): CocoImage(
            int(row["id"]),
            str(row["file_name"]),
            int(row["width"]),
            int(row["height"]),
        )
        for row in dataset["images"]
    }
    raw_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        raw_by_image[int(annotation["image_id"])].append(annotation)

    objects: dict[int, list[CocoObject]] = {}
    for image_id, rows in raw_by_image.items():
        image = images[image_id]
        counts = Counter(
            int(row["category_id"]) for row in rows if not bool(row["iscrowd"])
        )
        selected: list[CocoObject] = []
        for row in rows:
            category_id = int(row["category_id"])
            if bool(row["iscrowd"]) or counts[category_id] != 1:
                continue
            x, y, width, height = map(float, row["bbox"])
            area_fraction = width * height / (image.width * image.height)
            width_fraction = width / image.width
            height_fraction = height / image.height
            if (
                area_fraction < 0.008
                or area_fraction > 0.70
                or width_fraction < 0.055
                or height_fraction < 0.055
            ):
                continue
            segmentation = row["segmentation"]
            if not isinstance(segmentation, list):
                raise ValueError(
                    f"Non-crowd annotation {row['id']} does not contain polygons"
                )
            polygons = tuple(tuple(map(float, polygon)) for polygon in segmentation)
            if not polygons or any(
                len(polygon) < 6 or len(polygon) % 2 for polygon in polygons
            ):
                raise ValueError(f"Invalid polygons for annotation {row['id']}")
            selected.append(
                CocoObject(
                    annotation_id=int(row["id"]),
                    image_id=image_id,
                    name=category_names[category_id],
                    bbox=(x, y, x + width, y + height),
                    segmentation=polygons,
                    center_x=(x + width / 2.0) / image.width,
                    center_y=(y + height / 2.0) / image.height,
                    area_fraction=area_fraction,
                    width_fraction=width_fraction,
                    height_fraction=height_fraction,
                )
            )
        if len(selected) >= 2:
            objects[image_id] = selected
    return images, objects


def _iou(first: CocoObject, second: CocoObject) -> float:
    ax0, ay0, ax1, ay1 = first.bbox
    bx0, by0, bx1, by1 = second.bbox
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union


def _score_pair(
    image: CocoImage,
    first: CocoObject,
    second: CocoObject,
    group: str,
    vertical_cross_gap_max: float,
) -> PairCandidate | None:
    if _iou(first, second) > 0.08:
        return None
    dx = abs(first.center_x - second.center_x)
    dy = abs(first.center_y - second.center_y)
    horizontal_boundary = dx - (first.width_fraction + second.width_fraction) / 2.0
    vertical_boundary = dy - (first.height_fraction + second.height_fraction) / 2.0
    area_ratio = first.area_fraction / second.area_fraction
    symmetric_ratio = max(area_ratio, 1.0 / area_ratio)
    visibility = math.sqrt(min(first.area_fraction, second.area_fraction))
    if group == "horizontal":
        if dx < 0.26 or dy > 0.14 or symmetric_ratio > 2.5 or horizontal_boundary <= 0:
            return None
        score, axis_gap, cross_gap, boundary = (
            dx - 0.8 * dy + 0.15 * visibility,
            dx,
            dy,
            horizontal_boundary,
        )
    elif group == "vertical":
        if (
            dy < 0.26
            or dx > vertical_cross_gap_max
            or symmetric_ratio > 2.5
            or vertical_boundary <= 0
        ):
            return None
        score, axis_gap, cross_gap, boundary = (
            dy - 0.8 * dx + 0.15 * visibility,
            dy,
            dx,
            vertical_boundary,
        )
    elif group == "distance":
        center_gap = math.hypot(dx, dy)
        if center_gap < 0.12 or symmetric_ratio > 4.0:
            return None
        score = visibility + 0.08 * center_gap - 0.025 * abs(math.log(area_ratio))
        axis_gap, cross_gap, boundary = center_gap, abs(math.log(area_ratio)), 0.0
    else:
        raise ValueError(f"Unknown group {group!r}")
    return PairCandidate(
        image, group, first, second, score, axis_gap, cross_gap, boundary, area_ratio
    )


def scan_pair_candidates(
    images: dict[int, CocoImage],
    objects: dict[int, list[CocoObject]],
) -> dict[str, list[PairCandidate]]:
    """Return the highest-scoring candidate per image and axis (PRE-02).

    Candidate rules reproduce the audited COCO construction geometry. The
    function is deterministic and has no side effects.
    """
    output = {group: [] for group in GROUPS}
    for image_id, instances in objects.items():
        for group in GROUPS:
            candidates = [
                value
                for first, second in itertools.combinations(instances, 2)
                if (value := _score_pair(images[image_id], first, second, group, 0.14))
                is not None
            ]
            if candidates:
                output[group].append(max(candidates, key=lambda value: value.score))
    for candidates in output.values():
        candidates.sort(key=lambda value: (-value.score, value.image.image_id))
    return output


def scan_validation_vertical_candidates(
    images: dict[int, CocoImage],
    objects: dict[int, list[CocoObject]],
) -> list[PairCandidate]:
    """Return COCO-1800 V candidates with cross-gap bound set to 0.20.

    Every other geometric rule and ranking equation is identical to PRE-02.
    The result contains at most one highest-scoring pair per image and is
    sorted by descending score, then image ID.
    """
    output = []
    for image_id, instances in objects.items():
        candidates = [
            value
            for first, second in itertools.combinations(instances, 2)
            if (value := _score_pair(images[image_id], first, second, "vertical", 0.20))
            is not None
        ]
        if candidates:
            output.append(max(candidates, key=lambda value: value.score))
    output.sort(key=lambda value: (-value.score, value.image.image_id))
    return output


def take_unique(
    candidates: list[PairCandidate],
    count: int,
    used_image_ids: set[int],
    image_root: Path,
) -> list[PairCandidate]:
    """Take the first candidates with image IDs unused by earlier axes.

    The selected image file must exist. A missing file stops the run instead of
    causing selection of a lower-ranked replacement candidate.
    """
    selected = []
    for candidate in candidates:
        image_id = candidate.image.image_id
        if image_id in used_image_ids:
            continue
        image_path = image_root / candidate.image.file_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        selected.append(candidate)
        used_image_ids.add(image_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} unique local candidates, found {len(selected)}")

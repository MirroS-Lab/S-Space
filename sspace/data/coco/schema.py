"""Define typed records shared by COCO preprocessing stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CocoImage:
    """Identify one official COCO image."""

    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoObject:
    """Store one non-crowd COCO instance and its native polygon mask source."""

    annotation_id: int
    image_id: int
    name: str
    bbox: tuple[float, float, float, float]
    segmentation: tuple[tuple[float, ...], ...]
    center_x: float
    center_y: float
    area_fraction: float
    width_fraction: float
    height_fraction: float


@dataclass(frozen=True)
class PairCandidate:
    """Represent the best object pair for one image and spatial axis."""

    image: CocoImage
    group: str
    first: CocoObject
    second: CocoObject
    score: float
    axis_gap: float
    cross_gap: float
    boundary_gap: float
    area_ratio: float


@dataclass(frozen=True)
class DepthMetric:
    """Store mask-averaged relative depth for one candidate pair."""

    first_mean: float
    second_mean: float
    normalized_gap: float
    confidence: float
    near_annotation_id: int
    far_annotation_id: int

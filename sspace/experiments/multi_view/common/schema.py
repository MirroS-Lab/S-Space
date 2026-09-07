"""Define validated H* multi-view samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from sspace.core.prompts.templates import AXIS_ORDER


GT_POSITION = {"horizontal": 0, "vertical": 1, "distance": 2}
ENDPOINT_AXIS_SIGN = {
    "left": ("horizontal", -1),
    "right": ("horizontal", 1),
    "up": ("vertical", -1),
    "down": ("vertical", 1),
    "front": ("distance", -1),
    "back": ("distance", 1),
    "far": ("distance", -1),
    "close": ("distance", 1),
}


def balanced_axis_subset(
    samples: tuple[MultiViewSample, ...], limit: int | None
) -> tuple[MultiViewSample, ...]:
    """Return a source-ordered bounded subset containing every H/V/D axis."""
    if limit is None or limit >= len(samples):
        return samples
    if limit < len(AXIS_ORDER):
        raise ValueError(f"Multi-view smoke limit must be at least {len(AXIS_ORDER)}")
    buckets = {
        axis: [index for index, sample in enumerate(samples) if sample.axis == axis]
        for axis in AXIS_ORDER
    }
    if any(not indices for indices in buckets.values()):
        raise ValueError("Multi-view dataset does not contain every H/V/D axis")
    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        for axis in AXIS_ORDER:
            indices = buckets[axis]
            if depth < len(indices):
                selected.append(indices[depth])
                if len(selected) == limit:
                    break
        depth += 1
    return tuple(samples[index] for index in sorted(selected))


@dataclass(frozen=True)
class MultiViewSample:
    """One same-object image pair with full source-coordinate evidence.

    H/V/D coordinates use positive right, below, and close. ``gt_delta_hvd``
    is always ``view_b - view_a``. The evaluator places A then B in the AB
    prompt and B then A in BA, so its projected state difference must use the
    same B-minus-A direction.
    """

    sample_id: str
    scene: str
    object_id: int
    object_name: str
    axis: str
    split: str
    view_a_observation_id: str
    view_b_observation_id: str
    image_a_path: Path
    image_b_path: Path
    image_a_sha256: str
    image_b_sha256: str
    gt_a_hvd: tuple[float, float, float]
    gt_b_hvd: tuple[float, float, float]
    gt_delta_hvd: tuple[float, float, float]
    visibility_a: float
    visibility_b: float
    grounding_mode: str = "unspecified"
    target_label: str | None = None

    def validate(self) -> None:
        """Reject invalid identities, coordinates, images, or target axes."""
        if (
            not self.sample_id
            or not self.scene
            or not self.object_name
            or self.axis not in AXIS_ORDER
            or not self.split
            or self.view_a_observation_id == self.view_b_observation_id
        ):
            raise ValueError("Multi-view sample identity is invalid")
        values = (*self.gt_a_hvd, *self.gt_b_hvd, *self.gt_delta_hvd)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Multi-view sample coordinates are not finite")
        expected = tuple(
            b - a for a, b in zip(self.gt_a_hvd, self.gt_b_hvd, strict=True)
        )
        if any(
            abs(value - target) > 1e-9
            for value, target in zip(self.gt_delta_hvd, expected, strict=True)
        ):
            raise ValueError("Multi-view coordinate delta is inconsistent")
        if self.gt_delta_hvd[self.axis_index] == 0.0:
            raise ValueError("Multi-view target-axis change cannot be zero")
        endpoint = self.endpoint_label
        endpoint_axis, endpoint_sign = ENDPOINT_AXIS_SIGN[endpoint]
        if endpoint_axis != self.axis or endpoint_sign != int(
            math.copysign(1, self.gt_delta_hvd[self.axis_index])
        ):
            raise ValueError("Multi-view endpoint label differs from ground truth")
        if min(self.visibility_a, self.visibility_b) < 0.0:
            raise ValueError("Multi-view visibility cannot be negative")
        if self.grounding_mode not in {
            "unspecified",
            "name_only",
            "bbox_assisted",
        }:
            raise ValueError("Multi-view grounding mode is invalid")
        for path, digest in (
            (self.image_a_path, self.image_a_sha256),
            (self.image_b_path, self.image_b_sha256),
        ):
            if (
                not path.is_file()
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Multi-view image identity is invalid")

    @property
    def axis_index(self) -> int:
        """Return the H/V/D position for this pair's target axis."""
        return GT_POSITION[self.axis]

    @property
    def endpoint_label(self) -> str:
        """Return the explicit endpoint, preserving benchmark terminology."""
        if self.target_label is not None:
            if self.target_label not in ENDPOINT_AXIS_SIGN:
                raise ValueError("Multi-view target label is invalid")
            return self.target_label
        sign = int(math.copysign(1, self.gt_delta_hvd[self.axis_index]))
        defaults = {
            ("horizontal", -1): "left",
            ("horizontal", 1): "right",
            ("vertical", -1): "up",
            ("vertical", 1): "down",
            ("distance", -1): "front",
            ("distance", 1): "back",
        }
        return defaults[(self.axis, sign)]

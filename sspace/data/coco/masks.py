"""Decode official COCO segmentations with COCO mask semantics."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as coco_mask

from .schema import CocoObject


def instance_mask(instance: CocoObject, width: int, height: int) -> np.ndarray:
    """Rasterize one COCO instance as a boolean image-space mask (PRE-03).

    Args:
        instance: A non-crowd object with official polygon segmentation.
        width: Original image width.
        height: Original image height.

    Returns:
        Boolean array with shape ``[height, width]``.

    Raises:
        ValueError: The official polygons produce an empty mask.

    Side effects:
        None.
    """
    polygons = [list(polygon) for polygon in instance.segmentation]
    encoded = coco_mask.frPyObjects(polygons, height, width)
    merged = coco_mask.merge(encoded)
    mask = np.asarray(coco_mask.decode(merged), dtype=bool)
    if mask.shape != (height, width):
        raise ValueError(
            f"Annotation {instance.annotation_id} decoded to {mask.shape}, "
            f"expected {(height, width)}"
        )
    if not mask.any():
        raise ValueError(f"Annotation {instance.annotation_id} produces an empty mask")
    return mask

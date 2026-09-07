"""Compute mask-averaged Depth Anything pseudo-labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .masks import instance_mask
from .schema import DepthMetric, PairCandidate


def mask_mean_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    """Return arithmetic mean depth over native instance pixels (PRE-04).

    The function implements ``d(o)=sum_p mask[p]D[p]/sum_p mask[p]``. It does
    not crop to the bounding box and does not include background pixels.
    """
    if depth.shape != mask.shape:
        raise ValueError(f"Depth shape {depth.shape} != mask shape {mask.shape}")
    values = depth[mask]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Instance mask contains no valid depth values")
    return float(values.mean())


def group_processed_images(
    processed: list[tuple[int, torch.Tensor]],
) -> dict[tuple[int, int], list[tuple[int, torch.Tensor]]]:
    """Group processor outputs by spatial tensor shape (PRE-04).

    The DPT image processor preserves enough aspect-ratio information that
    different COCO images can produce different ``[H,W]`` tensors. Only equal
    shapes can be concatenated into one model batch. This grouping changes
    execution efficiency only; it does not resize, pad, reject, or otherwise
    alter an image.

    Args:
        processed: ``(batch_position, pixel_values)`` records, where each
            tensor has shape ``[1,3,H,W]``.

    Returns:
        Records grouped by the exact ``(H,W)`` processor output shape.

    Raises:
        ValueError: A processor output is not ``[1,3,H,W]``.

    Side effects:
        None.
    """
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for position, pixels in processed:
        if pixels.ndim != 4 or pixels.shape[:2] != (1, 3):
            raise ValueError(
                f"Expected processor output [1,3,H,W], got {tuple(pixels.shape)}"
            )
        groups.setdefault(tuple(pixels.shape[-2:]), []).append((position, pixels))
    return groups


def infer_mask_depth(
    candidates: list[PairCandidate],
    image_root: Path,
    model_path: Path,
    device: str,
    batch_size: int,
) -> dict[int, DepthMetric]:
    """Infer one mask-based relative-depth metric per candidate (PRE-04).

    Depth Anything V2 predicts relative inverse depth, where larger values are
    treated as closer. For each instance ``o``, this function computes
    ``d(o) = mean(D[p] for p in native_coco_mask(o))``. Confidence is
    ``abs(d(first)-d(second)) / (q95(D)-q05(D))``.

    Raises:
        ValueError: A prediction, image, mask, or batch violates the protocol.

    Side effects:
        Loads the local depth model and runs GPU inference.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False
    )
    model = (
        AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True)
        .to(device)
        .eval()
    )
    output: dict[int, DepthMetric] = {}
    try:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            images = []
            for candidate in batch:
                with Image.open(image_root / candidate.image.file_name) as source:
                    image = source.convert("RGB")
                if image.size != (candidate.image.width, candidate.image.height):
                    raise ValueError(
                        f"Image size mismatch for {candidate.image.image_id}"
                    )
                images.append(image)
            processed = [
                (index, processor(images=image, return_tensors="pt")["pixel_values"])
                for index, image in enumerate(images)
            ]
            predictions: dict[int, torch.Tensor] = {}
            for shape_group in group_processed_images(processed).values():
                inputs = torch.cat([pixels for _, pixels in shape_group]).to(device)
                with torch.inference_mode():
                    predicted = model(pixel_values=inputs).predicted_depth.float()
                for local_index, (batch_index, _) in enumerate(shape_group):
                    predictions[batch_index] = predicted[local_index].cpu()
            if set(predictions) != set(range(len(batch))):
                raise ValueError(
                    "Depth inference did not return one prediction per image"
                )
            for index, (candidate, image) in enumerate(zip(batch, images, strict=True)):
                depth = torch.nn.functional.interpolate(
                    predictions[index][None, None],
                    size=(candidate.image.height, candidate.image.width),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0].numpy()
                finite = depth[np.isfinite(depth)]
                if finite.size != depth.size:
                    raise ValueError(
                        f"Non-finite depth for image {candidate.image.image_id}"
                    )
                robust_range = float(
                    np.quantile(finite, 0.95) - np.quantile(finite, 0.05)
                )
                if robust_range <= 0:
                    raise ValueError(
                        f"Degenerate depth for image {candidate.image.image_id}"
                    )
                first = mask_mean_depth(
                    depth,
                    instance_mask(
                        candidate.first, candidate.image.width, candidate.image.height
                    ),
                )
                second = mask_mean_depth(
                    depth,
                    instance_mask(
                        candidate.second, candidate.image.width, candidate.image.height
                    ),
                )
                gap = (first - second) / robust_range
                near, far = (
                    (candidate.first.annotation_id, candidate.second.annotation_id)
                    if gap > 0
                    else (candidate.second.annotation_id, candidate.first.annotation_id)
                )
                output[candidate.image.image_id] = DepthMetric(
                    first,
                    second,
                    gap,
                    abs(gap),
                    near,
                    far,
                )
                image.close()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output

"""Build whole-object masks and Depth Anything pseudo ground truth."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from sspace.run_records import (
    atomic_json,
    canonical_fingerprint,
    file_sha256,
)


def _load_rows(selection_path: Path, boxes_path: Path) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    by_id = {int(row["showcase_id"]): row for row in boxes["rows"]}
    selection_ids = [int(row["showcase_id"]) for row in selection]
    if not selection_ids or len(set(selection_ids)) != len(selection_ids):
        raise ValueError("Pseudo-GT selection requires unique non-empty IDs")
    if set(by_id) != set(selection_ids):
        raise ValueError("Object-box IDs differ from the selected rows")
    rows = []
    for selected in selection:
        row = dict(selected)
        row.update(by_id[int(row["showcase_id"])])
        box = np.asarray(row["box"], dtype=np.float64)
        if box.shape != (4,) or not np.all((0.0 <= box) & (box <= 1.0)):
            raise ValueError(f"Invalid normalized box for {row['showcase_id']}")
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValueError(f"Degenerate box for {row['showcase_id']}")
        rows.append(row)
    return rows


def _binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if not mask.any():
        raise ValueError(f"Empty mask: {path}")
    return mask


def _pixel_box(normalized: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = normalized
    return [x0 * width, y0 * height, x1 * width, y1 * height]


def _sam_masks(
    rows: list[dict[str, Any]],
    assets_dir: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(model_path, local_files_only=True)
    model = SamModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    mask_dir = output_dir / "object_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, start=1):
            image = Image.open(assets_dir / row["image"]).convert("RGB")
            width, height = image.size
            box = _pixel_box(row["box"], width, height)
            inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
            original_sizes = inputs["original_sizes"]
            reshaped_sizes = inputs["reshaped_input_sizes"]
            model_inputs = {
                name: value.to(device)
                for name, value in inputs.items()
                if name not in {"original_sizes", "reshaped_input_sizes"}
            }
            with torch.inference_mode():
                outputs = model(**model_inputs)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), original_sizes, reshaped_sizes
            )[0][0]
            best = int(outputs.iou_scores[0, 0].argmax().item())
            object_mask = np.asarray(masks[best], dtype=bool)
            part_mask = _binary_mask(assets_dir / row["mask"])
            if object_mask.shape != part_mask.shape:
                raise ValueError(f"SAM mask shape differs for {row['showcase_id']}")
            object_mask |= part_mask
            body_pixels = int((object_mask & ~part_mask).sum())
            if body_pixels < 100:
                rejected.append(
                    {
                        "showcase_id": int(row["showcase_id"]),
                        "reason": "object_body_mask_below_100_pixels",
                        "object_body_pixels": body_pixels,
                    }
                )
            else:
                mask_path = mask_dir / f"{row['showcase_id']:02d}_object_mask.png"
                Image.fromarray(object_mask.astype(np.uint8) * 255).save(mask_path)
                accepted.append(row)
            print(
                f"SAM {position}/{len(rows)} accepted={len(accepted)} "
                f"rejected={len(rejected)}",
                flush=True,
            )
            if len(accepted) == target_count:
                break
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(accepted) != target_count:
        raise ValueError(f"Only {len(accepted)} valid SAM masks; need {target_count}")
    return accepted, rejected


def _predict_depth_pair(
    image: Image.Image, processor: Any, model: Any, device: str
) -> tuple[np.ndarray, np.ndarray]:
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    pixels = processor(images=[image, flipped], return_tensors="pt")["pixel_values"].to(
        device
    )
    with torch.inference_mode():
        predicted = model(pixel_values=pixels).predicted_depth.float()
    resized = (
        torch.nn.functional.interpolate(
            predicted[:, None],
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[:, 0]
        .cpu()
        .numpy()
    )
    return resized[0], np.flip(resized[1], axis=1).copy()


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise ValueError("Cannot compute the centroid of an empty mask")
    return float(columns.mean()), float(rows.mean())


def _signed_label(value: float, dead_zone: float, negative: str, positive: str) -> str:
    if abs(value) < dead_zone:
        return "ambiguous"
    return positive if value > 0 else negative


def _overlay(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    tint = np.empty_like(base)
    tint[:] = color
    base[mask] = 0.55 * base[mask] + 0.45 * tint[mask]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _depth_image(depth: np.ndarray) -> Image.Image:
    low, high = np.quantile(depth, [0.05, 0.95])
    scaled = np.clip((depth - low) / (high - low), 0.0, 1.0)
    rgb = np.stack((scaled, 1.0 - np.abs(2.0 * scaled - 1.0), 1.0 - scaled), axis=-1)
    return Image.fromarray((255.0 * rgb).astype(np.uint8))


def _review_panel(
    image: Image.Image,
    object_mask: np.ndarray,
    part_mask: np.ndarray,
    depth: np.ndarray,
    box: list[float],
    title: str,
) -> Image.Image:
    panels = [
        image.copy(),
        _overlay(image, object_mask, (0, 220, 80)),
        _overlay(_depth_image(depth), part_mask, (255, 0, 0)),
    ]
    draw = ImageDraw.Draw(panels[0])
    draw.rectangle(box, outline=(255, 220, 0), width=max(2, image.width // 400))
    target_height = 280
    resized = []
    for panel in panels:
        panel.thumbnail((420, target_height), Image.Resampling.LANCZOS)
        resized.append(panel)
    canvas = Image.new("RGB", (sum(panel.width for panel in resized), 314), "white")
    offset = 0
    for panel in resized:
        canvas.paste(panel, (offset, 34))
        offset += panel.width
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    return canvas


def _depth_records(
    rows: list[dict[str, Any]],
    assets_dir: Path,
    depth_model_path: Path,
    output_dir: Path,
    device: str,
    hv_dead_zone: float,
    depth_dead_zone: float,
) -> list[dict[str, Any]]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(
        depth_model_path, local_files_only=True, use_fast=False
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        depth_model_path, local_files_only=True
    ).to(device).eval()
    depth_dir = output_dir / "depth_maps"
    panel_dir = output_dir / "review_panels"
    depth_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for position, row in enumerate(rows, start=1):
            image = Image.open(assets_dir / row["image"]).convert("RGB")
            part_mask = _binary_mask(assets_dir / row["mask"])
            object_mask_path = (
                output_dir / "object_masks" / f"{row['showcase_id']:02d}_object_mask.png"
            )
            object_mask = _binary_mask(object_mask_path)
            body_mask = object_mask & ~part_mask
            if body_mask.sum() < 100:
                raise ValueError(f"Object body mask is too small for {row['showcase_id']}")
            original_depth, flipped_depth = _predict_depth_pair(
                image, processor, model, device
            )
            depth = (original_depth + flipped_depth) / 2.0
            finite = depth[np.isfinite(depth)]
            robust_range = float(np.quantile(finite, 0.95) - np.quantile(finite, 0.05))
            if finite.size != depth.size or robust_range <= 0:
                raise ValueError(f"Invalid depth map for {row['showcase_id']}")
            part_mean = float(depth[part_mask].mean())
            body_mean = float(depth[body_mask].mean())
            gap = (part_mean - body_mean) / robust_range
            original_range = float(
                np.quantile(original_depth, 0.95) - np.quantile(original_depth, 0.05)
            )
            flipped_range = float(
                np.quantile(flipped_depth, 0.95) - np.quantile(flipped_depth, 0.05)
            )
            original_gap = float(
                (original_depth[part_mask].mean() - original_depth[body_mask].mean())
                / original_range
            )
            flipped_gap = float(
                (flipped_depth[part_mask].mean() - flipped_depth[body_mask].mean())
                / flipped_range
            )
            flip_consistent = bool(original_gap * flipped_gap > 0)
            part_x, part_y = _centroid(part_mask)
            object_x, object_y = _centroid(object_mask)
            object_rows, object_columns = np.nonzero(object_mask)
            object_width = float(object_columns.max() - object_columns.min() + 1)
            object_height = float(object_rows.max() - object_rows.min() + 1)
            delta_x = (part_x - object_x) / object_width
            delta_y = (part_y - object_y) / object_height
            depth_label = _signed_label(gap, depth_dead_zone, "far", "close")
            if not flip_consistent:
                depth_label = "ambiguous"
            depth_path = depth_dir / f"{row['showcase_id']:02d}_inverse_depth.npy"
            np.save(depth_path, depth.astype(np.float32))
            panel = _review_panel(
                image,
                object_mask,
                part_mask,
                depth,
                _pixel_box(row["box"], image.width, image.height),
                f"#{row['showcase_id']:02d} {row['object']}/{row['part']}  "
                f"H={delta_x:+.3f} V={delta_y:+.3f} D={gap:+.3f}",
            )
            panel.save(panel_dir / f"{row['showcase_id']:02d}_review.jpg", quality=92)
            records.append(
                {
                    "showcase_id": row["showcase_id"],
                    "object": row["object"],
                    "part": row["part"],
                    "score_candidate": bool(row["score"]),
                    "exclusion_reason": row.get("reason"),
                    "horizontal_delta": delta_x,
                    "vertical_delta": delta_y,
                    "horizontal_label": _signed_label(
                        delta_x, hv_dead_zone, "left", "right"
                    ),
                    "vertical_label": _signed_label(
                        delta_y, hv_dead_zone, "above", "below"
                    ),
                    "part_inverse_depth_mean": part_mean,
                    "object_body_inverse_depth_mean": body_mean,
                    "depth_gap_normalized": gap,
                    "depth_gap_original": original_gap,
                    "depth_gap_flipped": flipped_gap,
                    "depth_flip_consistent": flip_consistent,
                    "depth_label": depth_label,
                    "object_mask_sha256": file_sha256(object_mask_path),
                    "depth_map": depth_path.relative_to(output_dir).as_posix(),
                }
            )
            print(f"depth {position}/{len(rows)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def prepare_ground_truth(
    selection_path: Path,
    boxes_path: Path,
    assets_dir: Path,
    sam_model_path: Path,
    depth_model_path: Path,
    output_dir: Path,
    device: str,
    target_count: int,
    hv_dead_zone: float,
    depth_dead_zone: float,
    sam_weights_sha256: str,
    depth_weights_sha256: str,
    expected_retained_sha256: str,
    expected_values_sha256: str,
) -> None:
    """Reproduce the historical masks and portable pseudo-GT record."""
    if file_sha256(sam_model_path / "model.safetensors") != sam_weights_sha256:
        raise ValueError("SlimSAM weights differ from the declared checkpoint")
    if file_sha256(depth_model_path / "model.safetensors") != depth_weights_sha256:
        raise ValueError("Depth Anything weights differ from the declared checkpoint")
    rows = _load_rows(selection_path, boxes_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    retained, rejected = _sam_masks(
        rows, assets_dir, sam_model_path, output_dir, device, target_count
    )
    atomic_json(output_dir / "retained_selection.json", retained)
    if file_sha256(output_dir / "retained_selection.json") != expected_retained_sha256:
        raise ValueError("SlimSAM retained selection differs from the historical run")
    atomic_json(
        output_dir / "sam_filter_report.json",
        {
            "candidate_count": len(rows),
            "retained_count": len(retained),
            "rejected_count": len(rejected),
            "rejected": rejected,
        },
    )
    records = _depth_records(
        retained,
        assets_dir,
        depth_model_path,
        output_dir,
        device,
        hv_dead_zone,
        depth_dead_zone,
    )
    ground_truth = {"review_status": "pending", "rows": records}
    core = {
        "rows": [
            {name: value for name, value in row.items() if name != "depth_map"}
            for row in records
        ],
    }
    if canonical_fingerprint(core) != expected_values_sha256:
        raise ValueError("Pseudo-GT values differ from the historical run")
    atomic_json(output_dir / "pseudo_ground_truth.json", ground_truth)
    atomic_json(
        output_dir / "resolved_config.json",
        {
            "protocol": "sam_box_object_mask_depth_anything_v2_mean_pseudo_gt",
            "selection_sha256": file_sha256(selection_path),
            "boxes_sha256": file_sha256(boxes_path),
            "sam_weights_sha256": sam_weights_sha256,
            "depth_weights_sha256": depth_weights_sha256,
            "horizontal_vertical_dead_zone": hv_dead_zone,
            "depth_dead_zone": depth_dead_zone,
            "depth_value_semantics": "larger_inverse_depth_is_closer",
            "region_statistic": "arithmetic_mean",
            "object_reference": "slimsam_object_mask_minus_instructpart_part_mask",
            "review_status": "pending",
            "ground_truth_values_sha256": expected_values_sha256,
        },
    )
    print("complete pseudo-GT; visual review remains required", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--sam-model", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--hv-dead-zone", type=float, required=True)
    parser.add_argument("--depth-dead-zone", type=float, required=True)
    parser.add_argument("--sam-weights-sha256", required=True)
    parser.add_argument("--depth-weights-sha256", required=True)
    parser.add_argument("--expected-retained-sha256", required=True)
    parser.add_argument("--expected-values-sha256", required=True)
    args = parser.parse_args()
    prepare_ground_truth(
        args.selection.resolve(),
        args.boxes.resolve(),
        args.assets_dir.resolve(),
        args.sam_model.resolve(),
        args.depth_model.resolve(),
        args.output_dir.resolve(),
        args.device,
        args.target_count,
        args.hv_dead_zone,
        args.depth_dead_zone,
        args.sam_weights_sha256,
        args.depth_weights_sha256,
        args.expected_retained_sha256,
        args.expected_values_sha256,
    )


if __name__ == "__main__":
    main()

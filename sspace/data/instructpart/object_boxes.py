"""Detect whole-object boxes for the frozen InstructPart candidates."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sspace.run_records import atomic_json, file_sha256


def _part_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if not mask.any():
        raise ValueError(f"Empty part mask: {path}")
    return mask


def _coverage(mask: np.ndarray, box: np.ndarray) -> float:
    height, width = mask.shape
    x0, y0, x1, y1 = box
    left = max(0, min(width, int(np.floor(x0))))
    top = max(0, min(height, int(np.floor(y0))))
    right = max(0, min(width, int(np.ceil(x1))))
    bottom = max(0, min(height, int(np.ceil(y1))))
    if left >= right or top >= bottom:
        return 0.0
    return float(mask[top:bottom, left:right].sum() / mask.sum())


def _normalized_box(box: np.ndarray, width: int, height: int) -> list[float]:
    scale = np.asarray([width, height, width, height], dtype=np.float64)
    return [float(value) for value in np.clip(box / scale, 0.0, 1.0)]


def _detect(
    row: dict[str, Any],
    assets_dir: Path,
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
    minimum_coverage: float,
) -> tuple[dict[str, Any] | None, str | None]:
    image = Image.open(assets_dir / row["image"]).convert("RGB")
    mask = _part_mask(assets_dir / row["mask"])
    inputs = processor(images=image, text=f"{row['object']}.", return_tensors="pt")
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    candidates = []
    for box, score in zip(boxes, scores, strict=True):
        coverage = _coverage(mask, box)
        if coverage >= minimum_coverage:
            candidates.append((float(score), coverage, box))
    if not candidates:
        reason = (
            "no_object_detection"
            if not len(boxes)
            else "part_mask_not_contained_by_detection"
        )
        return None, reason
    score, coverage, box = max(candidates, key=lambda item: (item[0], item[1]))
    return (
        {
            "showcase_id": int(row["showcase_id"]),
            "box": _normalized_box(box, image.width, image.height),
            "score": True,
            "reason": None,
            "detector_score": score,
            "part_mask_coverage": coverage,
        },
        None,
    )


def detect_boxes(
    selection_path: Path,
    assets_dir: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    box_threshold: float,
    text_threshold: float,
    minimum_coverage: float,
    minimum_count: int,
    expected_weights_sha256: str,
    expected_selection_sha256: str,
    expected_boxes_sha256: str,
) -> None:
    """Run the historical Grounding DINO filter and seal its outputs."""
    if file_sha256(model_path / "model.safetensors") != expected_weights_sha256:
        raise ValueError("Grounding DINO weights differ from the declared checkpoint")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    rows = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(rows) < minimum_count:
        raise ValueError("Candidate selection is smaller than the retained count")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    accepted: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, start=1):
            if str(row["object"]).strip().lower() == str(row["part"]).strip().lower():
                detected, reason = None, "object_and_part_are_identical"
            else:
                detected, reason = _detect(
                    row,
                    assets_dir,
                    processor,
                    model,
                    device,
                    box_threshold,
                    text_threshold,
                    minimum_coverage,
                )
            if detected is None:
                rejected.append(
                    {
                        "showcase_id": int(row["showcase_id"]),
                        "mirror_question_id": int(row["mirror_question_id"]),
                        "reason": reason,
                    }
                )
            else:
                accepted.append(row)
                boxes.append(detected)
            print(
                f"detect {position}/{len(rows)} accepted={len(accepted)} "
                f"rejected={len(rejected)}",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(accepted) < minimum_count:
        raise ValueError(f"Only {len(accepted)} reliable rows; need {minimum_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "selection.json", accepted)
    atomic_json(
        output_dir / "object_boxes.json",
        {
            "detector": "IDEA-Research/grounding-dino-tiny",
            "model_weights_sha256": expected_weights_sha256,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "minimum_part_mask_coverage": minimum_coverage,
            "rows": boxes,
        },
    )
    atomic_json(
        output_dir / "filter_report.json",
        {
            "candidate_count": len(rows),
            "examined_count": len(rows),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejected": rejected,
        },
    )
    if file_sha256(output_dir / "selection.json") != expected_selection_sha256:
        raise ValueError("Grounding DINO retained a different historical subset")
    if file_sha256(output_dir / "object_boxes.json") != expected_boxes_sha256:
        raise ValueError("Grounding DINO boxes differ from the historical run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, required=True)
    parser.add_argument("--text-threshold", type=float, required=True)
    parser.add_argument("--minimum-coverage", type=float, required=True)
    parser.add_argument("--minimum-count", type=int, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-boxes-sha256", required=True)
    args = parser.parse_args()
    detect_boxes(
        args.selection.resolve(),
        args.assets_dir.resolve(),
        args.model_path.resolve(),
        args.output_dir.resolve(),
        args.device,
        args.box_threshold,
        args.text_threshold,
        args.minimum_coverage,
        args.minimum_count,
        args.expected_weights_sha256,
        args.expected_selection_sha256,
        args.expected_boxes_sha256,
    )


if __name__ == "__main__":
    main()

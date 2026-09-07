#!/usr/bin/env python3
"""Shared multi-model inference core for the uploaded-image coordinate WebUI.

Grounding DINO, SAM, validation, projection, and the persisted WebUI payload
remain in the HTTP process.  The selected vision-language model lives in one
persistent JSON-lines worker.  Switching models first disposes that worker, so
the service never keeps two coordinate checkpoints resident on the GPU.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image, ImageOps

from .models import (
    DEFAULT_MODEL_ID,
    MODEL_SPECS,
    ModelSpec,
    checkpoint_path_for,
    load_model_axes_by_layer,
)
from sspace.core.models.molmo import _validate_snapshot_path

MIN_OBJECTS = 2
MAX_OBJECTS = 8
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_BOX_THRESHOLD = 0.4
GROUNDING_TEXT_THRESHOLD = 0.3

SAM_MODEL_ID = "facebook/sam-vit-base"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
GROUNDING_MODEL_PATH = PROJECT_ROOT / ".cache/assets/models/grounding_dino"
SAM_MODEL_PATH = PROJECT_ROOT / ".cache/assets/models/sam_vit_base"


class InferenceStageError(RuntimeError):
    """A safe, structured pipeline error which an HTTP layer may expose."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        *,
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.status_code = int(status_code)


@dataclass(frozen=True)
class ObjectMention:
    label: str
    char_span: tuple[int, int]


@dataclass(frozen=True)
class GroundingCandidate:
    phrase: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class GroundedObject:
    object_id: str
    name: str
    char_span: tuple[int, int]
    bbox_xyxy: tuple[float, float, float, float]
    grounding_score: float


@dataclass(frozen=True)
class SamPrediction:
    mask: np.ndarray
    iou_score: float


@dataclass(frozen=True)
class MentionToken:
    position: int
    piece: str
    model_tokens: tuple[str, ...]
    char_span: tuple[int, int]


@dataclass(frozen=True)
class CoordinateExtraction:
    hidden_states: Mapping[str, Any]
    mentions: Mapping[str, MentionToken]


@dataclass(frozen=True)
class InferenceResult:
    payload: dict[str, Any]
    masks: Mapping[str, Image.Image]
    warnings: list[str]


class GroundingBackend(Protocol):
    def predict(
        self,
        image: Image.Image,
        labels: Sequence[str],
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> Sequence[GroundingCandidate]: ...


class SegmentationBackend(Protocol):
    def predict(
        self, image: Image.Image, boxes: Sequence[Sequence[float]]
    ) -> Sequence[SamPrediction]: ...


def normalise_upload_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and return a detached RGB image."""

    if not isinstance(image, Image.Image):
        raise InferenceStageError(
            "input", "invalid_image", "The upload is not a decoded image."
        )
    try:
        result = ImageOps.exif_transpose(image).convert("RGB")
        result.load()
    except Exception as exc:
        raise InferenceStageError(
            "input", "invalid_image", "The uploaded image cannot be decoded."
        ) from exc
    if result.width < 1 or result.height < 1:
        raise InferenceStageError(
            "input", "invalid_image", "The uploaded image has no pixels."
        )
    return result


def validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise InferenceStageError("input", "invalid_prompt", "Prompt must be text.")
    value = prompt.strip()
    if not value:
        raise InferenceStageError("input", "invalid_prompt", "Prompt cannot be empty.")
    if len(value) > 4096:
        raise InferenceStageError(
            "input", "invalid_prompt", "Prompt is longer than 4096 characters."
        )
    return value


def align_phrase_to_prompt(
    prompt: str,
    phrase: str,
    *,
    case_sensitive: bool = False,
    require_unique: bool = True,
) -> tuple[int, int]:
    """Return the half-open span for an exact phrase in the original prompt."""

    needle = str(phrase).strip().strip(".,!?;:").strip()
    if not needle:
        raise ValueError("Grounding phrase is empty")
    flags = 0 if case_sensitive else re.IGNORECASE
    matches = list(re.finditer(re.escape(needle), prompt, flags))
    if not matches:
        raise ValueError(f"Phrase {needle!r} is not present in the prompt")
    if require_unique and len(matches) != 1:
        raise ValueError(f"Phrase {needle!r} occurs {len(matches)} times in the prompt")
    match = matches[0]
    return match.start(), match.end()


def _coerce_mention(value: Any, prompt: str) -> ObjectMention:
    if isinstance(value, ObjectMention):
        mention = value
    elif isinstance(value, Mapping):
        label = value.get("label", value.get("name"))
        span = value.get("char_span")
        if span is None and isinstance(label, str):
            span = align_phrase_to_prompt(prompt, label, case_sensitive=True)
        mention = ObjectMention(str(label), tuple(int(x) for x in span))
    elif isinstance(value, str):
        mention = ObjectMention(
            value, align_phrase_to_prompt(prompt, value, case_sensitive=True)
        )
    else:
        raise ValueError(f"Unsupported mention value: {type(value)!r}")
    return mention


def validate_mentions(
    prompt: str,
    values: Sequence[Any],
    *,
    minimum: int = MIN_OBJECTS,
    maximum: int = MAX_OBJECTS,
) -> list[ObjectMention]:
    """Require exact, unique, non-overlapping prompt substrings."""

    try:
        mentions = [_coerce_mention(value, prompt) for value in values]
    except (TypeError, ValueError) as exc:
        raise InferenceStageError(
            "mention_extraction",
            "phrase_alignment_failed",
            "Object phrases could not be aligned exactly to the prompt.",
        ) from exc
    if not minimum <= len(mentions) <= maximum:
        raise InferenceStageError(
            "mention_extraction",
            "insufficient_objects" if len(mentions) < minimum else "too_many_objects",
            f"Expected {minimum} to {maximum} distinct visible objects, found {len(mentions)}.",
        )
    labels: set[str] = set()
    checked: list[ObjectMention] = []
    for mention in mentions:
        start, end = mention.char_span
        if not mention.label or not 0 <= start < end <= len(prompt):
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                "An object phrase has an invalid prompt span.",
            )
        if prompt[start:end] != mention.label:
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                f"Object phrase {mention.label!r} is not the exact text at its prompt span.",
            )
        if prompt.count(mention.label) != 1:
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                f"Object phrase {mention.label!r} must occur exactly once in the prompt.",
            )
        folded = mention.label.casefold()
        if folded in labels:
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                "Object phrases must be unique.",
            )
        labels.add(folded)
        checked.append(mention)
    checked.sort(key=lambda item: item.char_span)
    for first, second in zip(checked, checked[1:]):
        if first.char_span[1] > second.char_span[0]:
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                "Object phrase spans overlap.",
            )
    return checked


def parse_mention_json(text: str, prompt: str) -> list[ObjectMention]:
    """Parse the first JSON array in a model response and validate it strictly."""

    start = text.find("[")
    if start < 0:
        raise InferenceStageError(
            "mention_extraction",
            "invalid_mention_response",
            "Object extraction did not return a JSON array.",
        )
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise InferenceStageError(
            "mention_extraction",
            "invalid_mention_response",
            "Object extraction returned invalid JSON.",
        ) from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InferenceStageError(
            "mention_extraction",
            "invalid_mention_response",
            "Object extraction must return a JSON string array.",
        )
    raw: list[ObjectMention] = []
    for label in value:
        try:
            span = align_phrase_to_prompt(
                prompt, label, case_sensitive=True, require_unique=True
            )
        except ValueError as exc:
            raise InferenceStageError(
                "mention_extraction",
                "phrase_alignment_failed",
                "Every extracted object must be one exact, uniquely occurring prompt substring.",
            ) from exc
        raw.append(ObjectMention(label=label, char_span=span))
    return validate_mentions(prompt, raw)


def _normalise_candidate_phrase(value: str) -> str:
    return str(value).strip().strip(".,!?;:").strip().casefold()


def _candidate_mention(
    candidate: GroundingCandidate, mentions: Sequence[ObjectMention]
) -> ObjectMention | None:
    phrase = _normalise_candidate_phrase(candidate.phrase)
    exact = [item for item in mentions if item.label.casefold() == phrase]
    if len(exact) == 1:
        return exact[0]
    contained = [
        item
        for item in mentions
        if phrase
        and (phrase in item.label.casefold() or item.label.casefold() in phrase)
    ]
    return contained[0] if len(contained) == 1 else None


def _coerce_candidate(value: Any) -> GroundingCandidate:
    if isinstance(value, GroundingCandidate):
        return value
    if isinstance(value, Mapping):
        phrase = value.get("phrase", value.get("label", value.get("text_label", "")))
        box = value.get("bbox_xyxy", value.get("box", value.get("boxes")))
        return GroundingCandidate(
            phrase=str(phrase),
            bbox_xyxy=tuple(float(x) for x in box),
            score=float(value.get("score", 0.0)),
        )
    raise TypeError(f"Unsupported grounding candidate: {type(value)!r}")


def select_grounded_objects(
    prompt: str,
    mentions: Sequence[ObjectMention],
    candidates: Sequence[Any],
    *,
    width: int,
    height: int,
    minimum: int = MIN_OBJECTS,
    maximum: int = MAX_OBJECTS,
) -> tuple[list[GroundedObject], list[str]]:
    """Keep the highest-scoring valid box for every requested prompt span."""

    del prompt  # Spans were already validated against it by validate_mentions.
    warnings: list[str] = []
    best: dict[tuple[int, int], tuple[ObjectMention, GroundingCandidate]] = {}
    for raw in candidates:
        try:
            candidate = _coerce_candidate(raw)
            mention = _candidate_mention(candidate, mentions)
            if mention is None:
                warnings.append(
                    f"Ignored unaligned grounding phrase {candidate.phrase!r}."
                )
                continue
            values = [float(item) for item in candidate.bbox_xyxy]
            if (
                len(values) != 4
                or not math.isfinite(candidate.score)
                or not all(map(math.isfinite, values))
            ):
                raise ValueError("non-finite box or score")
            x0, y0, x1, y1 = values
            box = (
                max(0.0, min(float(width), x0)),
                max(0.0, min(float(height), y0)),
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("degenerate box")
            candidate = GroundingCandidate(
                candidate.phrase, box, float(candidate.score)
            )
        except (TypeError, ValueError):
            warnings.append("Ignored an invalid grounding box.")
            continue
        existing = best.get(mention.char_span)
        if existing is None or candidate.score > existing[1].score:
            best[mention.char_span] = (mention, candidate)

    chosen = list(best.values())
    if len(chosen) > maximum:
        chosen = sorted(chosen, key=lambda item: item[1].score, reverse=True)[:maximum]
        warnings.append(
            f"Grounding found more than {maximum} objects; kept the highest-scoring {maximum}."
        )
    chosen.sort(key=lambda item: item[0].char_span)
    if len(chosen) < minimum:
        raise InferenceStageError(
            "grounding",
            "insufficient_objects",
            f"Grounding found {len(chosen)} usable objects; at least {minimum} are required.",
        )
    missing = [item.label for item in mentions if item.char_span not in best]
    if missing:
        warnings.append("No valid grounding box for: " + ", ".join(missing) + ".")
    objects = [
        GroundedObject(
            object_id=f"object_{index + 1:02d}",
            name=mention.label,
            char_span=mention.char_span,
            bbox_xyxy=candidate.bbox_xyxy,
            grounding_score=candidate.score,
        )
        for index, (mention, candidate) in enumerate(chosen)
    ]
    return objects, warnings


def select_sam_masks(
    masks: Any,
    scores: Any,
    boxes: Sequence[Sequence[float]],
    *,
    minimum_pixels: int = 16,
) -> list[SamPrediction]:
    """Reuse the offline highest-IoU selection and bbox intersection exactly."""

    mask_array = np.asarray(masks, dtype=bool)
    score_array = np.asarray(scores, dtype=np.float32)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, None]
    if mask_array.ndim != 4 or score_array.ndim != 2:
        raise ValueError(
            f"Invalid SAM outputs: masks={mask_array.shape}, scores={score_array.shape}"
        )
    if mask_array.shape[:2] != score_array.shape or mask_array.shape[0] != len(boxes):
        raise ValueError("SAM mask/score/box count mismatch")
    selected: list[SamPrediction] = []
    height, width = mask_array.shape[-2:]
    for index, box in enumerate(boxes):
        choice = int(np.argmax(score_array[index]))
        mask = mask_array[index, choice].copy()
        x0, y0, x1, y1 = map(int, map(round, box))
        box_mask = np.zeros_like(mask, dtype=bool)
        box_mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
        mask &= box_mask
        pixels = int(mask.sum())
        if pixels < minimum_pixels:
            raise InferenceStageError(
                "segmentation",
                "mask_too_small",
                f"SAM mask {index + 1} is too small ({pixels} pixels).",
            )
        selected.append(
            SamPrediction(mask=mask, iou_score=float(score_array[index, choice]))
        )
    return selected


def mask_to_rgba(mask: Any) -> Image.Image:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {array.shape}")
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    rgba[array, :3] = np.asarray([102, 223, 201], dtype=np.uint8)
    rgba[array, 3] = 145
    return Image.fromarray(rgba, mode="RGBA")


def normalise_axes(values: Any) -> np.ndarray:
    axes = np.asarray(values, dtype=np.float32)
    if axes.ndim != 2 or axes.shape[0] != 3 or not np.isfinite(axes).all():
        raise ValueError(f"Invalid axes {axes.shape}")
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("Axis norms must be positive")
    return axes / norms


def _as_float_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def project_hidden_states(
    hidden_states: Mapping[str, Any], axes: Any
) -> dict[str, dict[str, list[float]]]:
    """Project object states and calculate the two scene-centered variants."""

    unit_axes = normalise_axes(axes)
    raw_by_id: dict[str, np.ndarray] = {}
    state_by_id: dict[str, np.ndarray] = {}
    for object_id, value in hidden_states.items():
        vector = _as_float_vector(value)
        if (
            vector.ndim != 1
            or vector.shape[0] != unit_axes.shape[1]
            or not np.isfinite(vector).all()
        ):
            raise ValueError(f"Invalid hidden state for {object_id!r}: {vector.shape}")
        raw_by_id[object_id] = unit_axes @ vector
        norm = max(float(np.linalg.norm(vector)), 1e-12)
        state_by_id[object_id] = unit_axes @ (vector / norm)
    if not raw_by_id:
        raise ValueError("No object hidden states")
    order = list(raw_by_id)
    raw_mean = np.stack([raw_by_id[key] for key in order]).mean(axis=0)
    state_mean = np.stack([state_by_id[key] for key in order]).mean(axis=0)
    result: dict[str, dict[str, list[float]]] = {}
    for object_id in order:
        values = {
            "raw": raw_by_id[object_id],
            "scene_centered": raw_by_id[object_id] - raw_mean,
            "state_normalized": state_by_id[object_id],
            "state_normalized_centered": state_by_id[object_id] - state_mean,
        }
        result[object_id] = {
            key: [float(item) for item in array] for key, array in values.items()
        }
    return result


def _pretrained_kwargs(revision: str, cache_dir: Path | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "revision": revision,
        "local_files_only": True,
    }
    if cache_dir is not None:
        values["cache_dir"] = str(cache_dir)
    return values


def _resolved_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class HuggingFaceGroundingBackend:
    """Pinned, local-only Grounding DINO adapter loaded at first use."""

    def __init__(
        self,
        *,
        model_id: str = GROUNDING_MODEL_ID,
        revision: str = GROUNDING_REVISION,
        model_path: Path = GROUNDING_MODEL_PATH,
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.model_path = Path(model_path)
        self.device_name = _resolved_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.processor: Any | None = None
        self.model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self.cache_dir is None:
            source: str | Path = _validate_snapshot_path(
                self.model_path, self.model_id, self.revision
            )
            kwargs = {"local_files_only": True}
        else:
            source = self.model_id
            kwargs = _pretrained_kwargs(self.revision, self.cache_dir)
        self.processor = AutoProcessor.from_pretrained(source, **kwargs)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(source, **kwargs)
            .to(self.device_name)
            .eval()
        )

    def predict(
        self,
        image: Image.Image,
        labels: Sequence[str],
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[GroundingCandidate]:
        import torch

        self._ensure_loaded()
        assert self.processor is not None and self.model is not None
        # A list makes the processor construct canonical ``label. label.`` text
        # while preserving only the requested object phrases, rather than
        # grounding instruction words from the user's natural-language prompt.
        inputs = self.processor(images=image, text=list(labels), return_tensors="pt")
        inputs = inputs.to(self.device_name)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(image.height, image.width)],
        )[0]
        scores = result["scores"].detach().float().cpu().tolist()
        boxes = result["boxes"].detach().float().cpu().tolist()
        phrases = result.get("text_labels", [])
        if not (len(scores) == len(boxes) == len(phrases)):
            raise RuntimeError("Grounding DINO returned inconsistent candidate arrays")
        return [
            GroundingCandidate(
                phrase=str(phrase),
                bbox_xyxy=tuple(float(item) for item in box),
                score=float(score),
            )
            for phrase, box, score in zip(phrases, boxes, scores, strict=True)
        ]


class HuggingFaceSamBackend:
    """Pinned, local-only SAM adapter preserving the offline mask policy."""

    def __init__(
        self,
        *,
        model_id: str = SAM_MODEL_ID,
        revision: str = SAM_REVISION,
        model_path: Path = SAM_MODEL_PATH,
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.model_path = Path(model_path)
        self.device_name = _resolved_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.processor: Any | None = None
        self.model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import SamModel, SamProcessor

        if self.cache_dir is None:
            source: str | Path = _validate_snapshot_path(
                self.model_path, self.model_id, self.revision
            )
            kwargs = {"local_files_only": True}
        else:
            source = self.model_id
            kwargs = _pretrained_kwargs(self.revision, self.cache_dir)
        self.processor = SamProcessor.from_pretrained(source, **kwargs)
        self.model = (
            SamModel.from_pretrained(source, torch_dtype=torch.float32, **kwargs)
            .to(self.device_name)
            .eval()
        )

    def predict(
        self, image: Image.Image, boxes: Sequence[Sequence[float]]
    ) -> list[SamPrediction]:
        import torch

        self._ensure_loaded()
        assert self.processor is not None and self.model is not None
        inputs = self.processor(
            images=image,
            input_boxes=[[[float(item) for item in box] for box in boxes]],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        reshaped_sizes = inputs["reshaped_input_sizes"]
        tensor_inputs = {
            key: value.to(device=self.device_name)
            for key, value in inputs.items()
            if key not in {"original_sizes", "reshaped_input_sizes"}
        }
        with torch.inference_mode():
            outputs = self.model(**tensor_inputs, multimask_output=True)
        masks = (
            self.processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                original_sizes.detach().cpu(),
                reshaped_sizes.detach().cpu(),
            )[0]
            .numpy()
            .astype(bool)
        )
        scores = outputs.iou_scores[0].detach().float().cpu().numpy()
        return select_sam_masks(masks, scores, boxes)


def _coerce_sam_prediction(value: Any) -> SamPrediction:
    if isinstance(value, SamPrediction):
        return value
    if isinstance(value, Mapping):
        return SamPrediction(
            mask=np.asarray(value["mask"], dtype=bool),
            iou_score=float(value.get("iou_score", value.get("score", 0.0))),
        )
    if isinstance(value, Sequence) and len(value) == 2:
        return SamPrediction(np.asarray(value[0], dtype=bool), float(value[1]))
    raise TypeError(f"Unsupported SAM prediction: {type(value)!r}")


def _coerce_mention_token(value: Any) -> MentionToken:
    if isinstance(value, MentionToken):
        return value
    if isinstance(value, Mapping):
        return MentionToken(
            position=int(value["position"]),
            piece=str(value["piece"]),
            model_tokens=tuple(str(item) for item in value["model_tokens"]),
            char_span=tuple(int(item) for item in value["char_span"]),
        )
    raise TypeError(f"Unsupported mention token: {type(value)!r}")


def _coerce_coordinate_extraction(value: Any) -> CoordinateExtraction:
    if isinstance(value, CoordinateExtraction):
        return value
    if isinstance(value, Mapping):
        return CoordinateExtraction(
            hidden_states=value["hidden_states"],
            mentions={
                key: _coerce_mention_token(item)
                for key, item in value["mentions"].items()
            },
        )
    raise TypeError(f"Unsupported coordinate extraction: {type(value)!r}")


def _local_asset_base(value: str) -> str:
    # The standalone CLI writes image.png and masks/ next to result.json.
    if value == ".":
        return "."
    base = str(value).rstrip("/")
    if (
        not base.startswith("/")
        or base.startswith("//")
        or "\\" in base
        or "\0" in base
    ):
        raise ValueError("asset_base must be a local absolute URL path")
    return base


def build_web_payload(
    *,
    image: Image.Image,
    prompt: str,
    run_id: str,
    asset_base: str,
    objects: Sequence[GroundedObject],
    segmentations: Mapping[str, SamPrediction],
    extraction: CoordinateExtraction,
    coordinates: Mapping[str, Mapping[str, Sequence[float]]],
    warnings: Sequence[str],
    mention_source: str,
) -> dict[str, Any]:
    """Build one online sample using the existing frozen WebUI contract."""

    base = _local_asset_base(asset_base)
    if not run_id:
        raise ValueError("run_id cannot be empty")
    by_id = {item.object_id: item for item in objects}
    required = set(by_id)
    if set(segmentations) != required or set(extraction.hidden_states) != required:
        raise ValueError("Object, segmentation, and coordinate IDs do not match")
    if set(extraction.mentions) != required or set(coordinates) != required:
        raise ValueError("Object, mention, and projected-coordinate IDs do not match")

    top_objects: list[dict[str, Any]] = []
    view_objects: dict[str, dict[str, Any]] = {}
    prompt_mentions: list[dict[str, Any]] = []
    for item in objects:
        x0, y0, x1, y1 = item.bbox_xyxy
        center_px = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        center = [center_px[0] / image.width, center_px[1] / image.height]
        mask_url = f"{base}/masks/{item.object_id}.png"
        segmentation = segmentations[item.object_id]
        token = extraction.mentions[item.object_id]
        if token.char_span != item.char_span:
            raise ValueError(f"Token span mismatch for {item.object_id}")
        common = {
            "object_id": item.object_id,
            "name": item.name,
            "bbox_xyxy": [float(value) for value in item.bbox_xyxy],
            "center_px": [float(value) for value in center_px],
            "center_normalized": [float(value) for value in center],
            "grounding_score": float(item.grounding_score),
            "sam_iou_score": float(segmentation.iou_score),
            "mask_url": mask_url,
        }
        top_objects.append(common)
        values = coordinates[item.object_id]
        view_objects[item.name] = {
            "object_id": item.object_id,
            **{
                mode: [float(number) for number in values[mode]]
                for mode in (
                    "raw",
                    "scene_centered",
                    "state_normalized",
                    "state_normalized_centered",
                )
            },
            "bbox_center": common["center_normalized"],
            "bbox_xyxy": common["bbox_xyxy"],
            "grounding_score": common["grounding_score"],
            "sam_iou_score": common["sam_iou_score"],
            "mask_url": mask_url,
        }
        prompt_mentions.append(
            {
                "object": item.name,
                "object_id": item.object_id,
                "position": int(token.position),
                "piece": token.piece,
                "model_tokens": list(token.model_tokens),
                "char_span": list(token.char_span),
            }
        )

    sample = {
        "sample_id": run_id,
        "data_source": "upload",
        "scene_id": run_id,
        "question_id": "uploaded_prompt",
        "width": int(image.width),
        "height": int(image.height),
        "image_url": f"{base}/image.png",
        "objects": top_objects,
        "views": [
            {
                "id": "uploaded_prompt",
                "prompts": [
                    {
                        "sample_id": run_id,
                        "view": "uploaded_prompt",
                        "text": prompt,
                        "mentions": prompt_mentions,
                    }
                ],
                "objects": view_objects,
            }
        ],
    }
    payload: dict[str, Any] = {
        "metadata": {
            "experiment_version": 1,
            "axis_normalization": "unit_l2",
            "axis_orientation": {
                "horizontal": "right",
                "vertical": "below",
                "distance": "close",
            },
            "grounding_model": GROUNDING_MODEL_ID,
            "grounding_revision": GROUNDING_REVISION,
            "grounding_box_threshold": GROUNDING_BOX_THRESHOLD,
            "grounding_text_threshold": GROUNDING_TEXT_THRESHOLD,
            "sam_model": SAM_MODEL_ID,
            "sam_revision": SAM_REVISION,
            "segmentation_overlay": "box-prompted SAM RGBA mask",
            "mention_source": mention_source,
            "n_samples": 1,
            "metrics": [],
            "warnings": list(warnings),
        },
        "samples": [sample],
    }
    # Enforce the persistence boundary: no tensors, numpy scalars, or NaNs.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


DEFAULT_WORKER_SCRIPT = HERE / "worker.py"
MAX_WORKER_LINE_BYTES = 32 * 1024 * 1024
LOGGER = logging.getLogger("object_coordinate_lens.inference")


def worker_python_path(value: Path) -> Path:
    """Make a worker interpreter absolute without dereferencing its venv link."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class ModelWorkerError(RuntimeError):
    """A model subprocess failed its startup or JSON-lines request."""


class PersistentModelWorkerManager:
    """Own at most one persistent model worker and synchronise its RPC stream."""

    def __init__(
        self,
        model_specs: Mapping[str, ModelSpec] = MODEL_SPECS,
        *,
        device: str = "auto",
        cache_dir: Path | None = None,
        worker_python: Path | str | Mapping[str, Path | str] | None = None,
        worker_script: Path | None = None,
        runtime_root: Path | None = None,
        startup_timeout: float = 600.0,
        request_timeout: float = 600.0,
        shutdown_timeout: float = 10.0,
        temp_dir: Path | None = None,
        available_layers_by_model: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.model_specs = dict(model_specs)
        if available_layers_by_model is None:
            self.available_layers_by_model = {
                model_id: tuple(load_model_axes_by_layer(spec)[0])
                for model_id, spec in self.model_specs.items()
            }
        else:
            self.available_layers_by_model = {
                model_id: tuple(int(layer) for layer in layers)
                for model_id, layers in available_layers_by_model.items()
            }
        if set(self.available_layers_by_model) != set(self.model_specs):
            raise ValueError("Worker layer registry does not match the model registry")
        for model_id, layers in self.available_layers_by_model.items():
            if (
                not layers
                or len(set(layers)) != len(layers)
                or self.model_specs[model_id].default_layer not in layers
            ):
                raise ValueError(f"Invalid worker layer registry for {model_id}")
        self.device = str(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.worker_python = worker_python
        self.worker_script = Path(worker_script or DEFAULT_WORKER_SCRIPT).resolve()
        self.runtime_root = Path(runtime_root or PROJECT_ROOT / ".runtime").resolve()
        self.startup_timeout = self._positive_timeout(
            startup_timeout, "startup_timeout"
        )
        self.request_timeout = self._positive_timeout(
            request_timeout, "request_timeout"
        )
        self.shutdown_timeout = self._positive_timeout(
            shutdown_timeout, "shutdown_timeout"
        )
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._active_model_id: str | None = None
        self._read_buffer = bytearray()
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def _positive_timeout(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return result

    def _python_for(self, model_id: str) -> str:
        value = self.worker_python
        if isinstance(value, Mapping):
            value = value.get(model_id)
        return str(Path(value).expanduser()) if value is not None else sys.executable

    def _overlay_for(self, spec: ModelSpec) -> Path | None:
        override = os.environ.get(f"OBJECT_LENS_RUNTIME_PATH_{spec.id.upper()}")
        if override:
            return Path(override).expanduser().resolve()
        if not spec.runtime_overlay:
            return None
        overlay = Path(spec.runtime_overlay).expanduser()
        if overlay.is_absolute():
            return overlay
        candidates: list[Path] = [
            self.runtime_root / overlay,
            self.runtime_root.parent / overlay,
        ]
        candidates.append(PROJECT_ROOT / overlay)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        # Preserve the declared runtime root in the resulting error/path.
        return candidates[0].resolve()

    def _environment_for(self, spec: ModelSpec) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        overlay = self._overlay_for(spec)
        if overlay is not None:
            if not overlay.is_dir():
                raise FileNotFoundError(
                    f"Runtime overlay for {spec.id} does not exist: {overlay}"
                )
            current = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(overlay) if not current else str(overlay) + os.pathsep + current
            )
        return environment

    def _readline(self, timeout: float) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise ModelWorkerError("Model worker is not running")
        deadline = time.monotonic() + timeout
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                if newline > MAX_WORKER_LINE_BYTES:
                    raise ModelWorkerError(
                        "Model worker returned an oversized response"
                    )
                line = bytes(self._read_buffer[:newline])
                del self._read_buffer[: newline + 1]
                return line
            if len(self._read_buffer) > MAX_WORKER_LINE_BYTES:
                raise ModelWorkerError("Model worker returned an oversized response")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the model worker")
            readable, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
            if not readable:
                raise TimeoutError("Timed out waiting for the model worker")
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                code = process.poll()
                raise ModelWorkerError(
                    f"Model worker exited unexpectedly (code {code})"
                )
            self._read_buffer.extend(chunk)

    def _read_json(self, timeout: float) -> dict[str, Any]:
        raw = self._readline(timeout)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelWorkerError("Model worker returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ModelWorkerError("Model worker response must be a JSON object")
        return value

    def _write_json(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ModelWorkerError("Model worker is not running")
        data = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        view = memoryview(data)
        while view:
            written = os.write(process.stdin.fileno(), view)
            if written <= 0:
                raise ModelWorkerError("Could not write to the model worker")
            view = view[written:]

    def _start_locked(self, model_id: str) -> None:
        if self._closed:
            raise ModelWorkerError("Model worker manager is closed")
        if model_id not in self.model_specs:
            raise KeyError(model_id)
        if (
            self._active_model_id == model_id
            and self._process is not None
            and self._process.poll() is None
        ):
            return
        self._stop_locked()
        spec = self.model_specs[model_id]
        command = [
            self._python_for(model_id),
            str(self.worker_script),
            "--model-id",
            model_id,
            "--device",
            self.device,
        ]
        if self.cache_dir is not None:
            command.extend(("--cache-dir", str(self.cache_dir)))
        try:
            self._read_buffer.clear()
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
                cwd=str(PROJECT_ROOT),
                env=self._environment_for(spec),
            )
            ready = self._read_json(self.startup_timeout)
            expected_layers = list(self.available_layers_by_model[model_id])
            if (
                ready.get("ready") is not True
                or ready.get("model_id") != model_id
                or ready.get("default_layer") != spec.default_layer
                or ready.get("available_layers") != expected_layers
            ):
                error = ready.get("error")
                message = error.get("message") if isinstance(error, Mapping) else None
                raise ModelWorkerError(
                    message
                    or f"Worker for {model_id} did not become ready with matching layers"
                )
            self._active_model_id = model_id
        except (ModelWorkerError, TimeoutError):
            self._stop_locked()
            raise
        except Exception as exc:
            self._stop_locked()
            raise ModelWorkerError(f"Could not start worker for {model_id}") from exc

    def _rpc_locked(self, model_id: str, action: str, **values: Any) -> Any:
        self._start_locked(model_id)
        request_id = uuid.uuid4().hex
        try:
            self._write_json({"request_id": request_id, "action": action, **values})
        except (ModelWorkerError, OSError) as exc:
            self._stop_locked()
            if isinstance(exc, ModelWorkerError):
                raise
            raise ModelWorkerError(
                "Could not communicate with the model worker"
            ) from exc
        try:
            response = self._read_json(self.request_timeout)
        except TimeoutError:
            self._stop_locked()
            raise
        except ModelWorkerError:
            self._stop_locked()
            raise
        except (OSError, ValueError) as exc:
            self._stop_locked()
            raise ModelWorkerError(
                "Could not communicate with the model worker"
            ) from exc
        if response.get("request_id") != request_id:
            self._stop_locked()
            raise ModelWorkerError(
                "Model worker response ID does not match the request"
            )
        if response.get("ok") is not True:
            error = response.get("error")
            if isinstance(error, Mapping):
                error_type = str(error.get("type", "ModelWorkerError"))
                message = str(error.get("message", "Model worker request failed"))
                raise ModelWorkerError(f"{error_type}: {message}")
            raise ModelWorkerError("Model worker request failed")
        if "result" not in response:
            raise ModelWorkerError("Model worker response has no result")
        return response["result"]

    def _with_temp_image(
        self, image: Image.Image, function: Callable[[str], Any]
    ) -> Any:
        handle, raw_path = tempfile.mkstemp(suffix=".png", dir=self.temp_dir)
        os.close(handle)
        path = Path(raw_path)
        try:
            image.save(path, format="PNG")
            return function(str(path))
        finally:
            path.unlink(missing_ok=True)

    def generate_mentions(self, model_id: str, image: Image.Image, prompt: str) -> str:
        with self._lock:
            result = self._with_temp_image(
                image,
                lambda image_path: self._rpc_locked(
                    model_id,
                    "generate_mentions",
                    image_path=image_path,
                    prompt=prompt,
                ),
            )
        if not isinstance(result, Mapping) or not isinstance(result.get("answer"), str):
            raise ModelWorkerError("Mention worker result must contain a text answer")
        return result["answer"]

    def extract_coordinates(
        self,
        model_id: str,
        image: Image.Image,
        prompt: str,
        spans: Mapping[str, tuple[int, int]],
        layer: int | None = None,
    ) -> Mapping[str, Any]:
        serialised_spans = {
            key: [int(value[0]), int(value[1])] for key, value in spans.items()
        }
        with self._lock:
            result = self._with_temp_image(
                image,
                lambda image_path: self._rpc_locked(
                    model_id,
                    "extract_coordinates",
                    image_path=image_path,
                    prompt=prompt,
                    spans=serialised_spans,
                    layer=layer,
                ),
            )
        if not isinstance(result, Mapping):
            raise ModelWorkerError("Coordinate worker result must be a JSON object")
        return result

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        self._active_model_id = None
        self._read_buffer.clear()
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    request_id = uuid.uuid4().hex
                    data = (
                        json.dumps(
                            {"request_id": request_id, "action": "shutdown"},
                            allow_nan=False,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    process.stdin.write(data)
                process.wait(timeout=self.shutdown_timeout)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=self.shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        LOGGER.error(
                            "Worker process %s did not exit after SIGKILL", process.pid
                        )
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def model_readiness(self, model_id: str) -> dict[str, Any]:
        """Check the immutable local assets needed before advertising a model."""

        try:
            spec = self.model_specs[model_id]
        except KeyError:
            return {"ready": False, "message": "Model is not registered."}
        checks = (
            (self.worker_script.is_file(), "Model worker script is unavailable."),
            (
                Path(self._python_for(model_id)).expanduser().is_file(),
                "Model worker Python is unavailable.",
            ),
        )
        for ready, message in checks:
            if not ready:
                return {"ready": False, "message": message}
        checkpoint = checkpoint_path_for(spec, self.cache_dir)
        if (
            not checkpoint.is_dir()
            or checkpoint.is_symlink()
            or checkpoint.name != spec.revision
        ):
            return {
                "ready": False,
                "message": "Pinned model checkpoint is unavailable.",
            }
        if spec.runtime_overlay:
            overlay = self._overlay_for(spec)
            if overlay is None or not overlay.is_dir():
                return {
                    "ready": False,
                    "message": "Model runtime overlay is unavailable.",
                }
        return {"ready": True}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_locked()


def _manual_mentions(prompt: str, values: Sequence[Any]) -> list[Any]:
    try:
        mentions = validate_mentions(prompt, values)
        if any(
            item.label != item.label.strip()
            or unicodedata.category(item.label[0]).startswith("P")
            or unicodedata.category(item.label[-1]).startswith("P")
            for item in mentions
        ):
            raise InferenceStageError(
                "mention_selection",
                "invalid_object_phrase",
                "Manual object phrases cannot start or end with whitespace or punctuation.",
            )
    except InferenceStageError as exc:
        raise InferenceStageError(
            "mention_selection", exc.code, exc.message, status_code=exc.status_code
        ) from exc
    return mentions


class OnlineInferencePipeline:
    """One shared segmentation pipeline with registered coordinate models."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        device: str = "auto",
        worker_python: Path | str | Mapping[str, Path | str] | None = None,
        runtime_root: Path | None = None,
        startup_timeout: float = 600.0,
        request_timeout: float = 600.0,
        model_specs: Mapping[str, ModelSpec] = MODEL_SPECS,
        model_manager: Any | None = None,
        grounding_backend: GroundingBackend | None = None,
        sam_backend: SegmentationBackend | None = None,
        box_threshold: float = GROUNDING_BOX_THRESHOLD,
        text_threshold: float = GROUNDING_TEXT_THRESHOLD,
    ) -> None:
        self.model_specs = dict(model_specs)
        if not self.model_specs:
            raise ValueError("At least one model spec is required")
        if DEFAULT_MODEL_ID not in self.model_specs:
            raise ValueError(f"Default model {DEFAULT_MODEL_ID!r} is not registered")
        self.default_model_id = DEFAULT_MODEL_ID
        self.axes_by_model_layer: dict[tuple[str, int], np.ndarray] = {}
        self.axis_metadata_by_model_layer: dict[tuple[str, int], dict[str, Any]] = {}
        self.available_layers_by_model: dict[str, tuple[int, ...]] = {}
        for model_id, spec in self.model_specs.items():
            if model_id != spec.id:
                raise ValueError(f"Registry key {model_id!r} != spec id {spec.id!r}")
            raw_axes, metadata_by_layer = load_model_axes_by_layer(spec)
            layers = tuple(raw_axes)
            if tuple(metadata_by_layer) != layers:
                raise ValueError(f"{model_id} axes and metadata layers do not match")
            self.available_layers_by_model[model_id] = layers
            for layer in layers:
                key = (model_id, layer)
                self.axes_by_model_layer[key] = normalise_axes(raw_axes[layer])
                self.axis_metadata_by_model_layer[key] = metadata_by_layer[layer]

        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        cache_path = Path(cache_dir) if cache_dir is not None else None
        temporary_root = PROJECT_ROOT / ".tmp" / "object-coordinate-lens"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager or PersistentModelWorkerManager(
            self.model_specs,
            device=device,
            cache_dir=cache_path,
            worker_python=worker_python,
            runtime_root=runtime_root,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            temp_dir=temporary_root,
            available_layers_by_model=self.available_layers_by_model,
        )
        self.grounding_backend = grounding_backend or HuggingFaceGroundingBackend(
            device=device, cache_dir=cache_path
        )
        self.sam_backend = sam_backend or HuggingFaceSamBackend(
            device=device, cache_dir=cache_path
        )
        self._closed = False

    def _model_public_config(self, spec: ModelSpec) -> dict[str, Any]:
        metadata = self.axis_metadata_by_model_layer[(spec.id, spec.default_layer)]
        available_layers = self.available_layers_by_model[spec.id]
        readiness_function = getattr(self.model_manager, "model_readiness", None)
        readiness = (
            readiness_function(spec.id)
            if callable(readiness_function)
            else {"ready": True}
        )
        result = {
            "id": spec.id,
            "label": spec.label,
            "repository": spec.repo,
            "revision": spec.revision,
            "source_layer": int(spec.default_layer),
            "default_layer": int(spec.default_layer),
            "available_layers": list(available_layers),
            "layer_count": int(spec.layer_count),
            "hidden_size": int(spec.hidden),
            "axis_artifact_id": metadata["artifact_id"],
            "axis_bundle": Path(spec.axes_path).parent.name,
            "ready": bool(readiness.get("ready")),
        }
        if not result["ready"] and readiness.get("message"):
            result["message"] = str(readiness["message"])
        return result

    @property
    def public_config(self) -> dict[str, Any]:
        models = [self._model_public_config(spec) for spec in self.model_specs.values()]
        return {
            "capabilities": {
                "model_selection": True,
                "layer_selection": True,
                "manual_mentions": True,
                "rerun_saved_uploads": True,
            },
            "models": models,
            "default_model_id": self.default_model_id,
            "grounding": {
                "id": GROUNDING_MODEL_ID,
                "revision": GROUNDING_REVISION,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
            },
            "segmentation": {"id": SAM_MODEL_ID, "revision": SAM_REVISION},
        }

    def _selected_model_layer(
        self, model_id: str | None, layer: int | None
    ) -> tuple[str, ModelSpec, int]:
        requested_id = self.default_model_id if model_id is None else str(model_id)
        if layer is not None and (
            isinstance(layer, bool) or not isinstance(layer, int)
        ):
            raise InferenceStageError(
                "input",
                "invalid_layer",
                "Layer must be an integer.",
                status_code=400,
            )
        selected_id = requested_id
        try:
            spec = self.model_specs[selected_id]
        except KeyError as exc:
            allowed = ", ".join(self.model_specs)
            raise InferenceStageError(
                "input",
                "invalid_model",
                f"Unknown model {requested_id!r}; expected one of: {allowed}.",
                status_code=400,
            ) from exc
        selected_layer = spec.default_layer if layer is None else layer
        selected_layer = int(selected_layer)
        available_layers = self.available_layers_by_model[selected_id]
        if selected_layer not in available_layers:
            allowed = ", ".join(f"L{value}" for value in available_layers)
            raise InferenceStageError(
                "input",
                "invalid_layer",
                f"{spec.label} has no L{selected_layer} basis; expected one of: {allowed}.",
                status_code=400,
            )
        return selected_id, spec, selected_layer

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        run_id: str,
        asset_base: str,
        *,
        manual_mentions: Sequence[Any] | None = None,
        model_id: str | None = None,
        layer: int | None = None,
    ) -> InferenceResult:
        if self._closed:
            raise RuntimeError("Inference pipeline is closed")
        selected_id, spec, selected_layer = self._selected_model_layer(model_id, layer)
        image = normalise_upload_image(image)
        prompt = validate_prompt(prompt)
        if manual_mentions is None:
            try:
                answer = self.model_manager.generate_mentions(
                    selected_id, image, prompt
                )
            except TimeoutError as exc:
                LOGGER.exception("Mention generation timed out for %s", selected_id)
                raise InferenceStageError(
                    "mention_extraction",
                    "model_unavailable",
                    "The selected model is temporarily unavailable.",
                    status_code=503,
                ) from exc
            except ModelWorkerError as exc:
                LOGGER.exception("Mention worker failed for %s", selected_id)
                raise InferenceStageError(
                    "mention_extraction",
                    "model_unavailable",
                    "The selected model is temporarily unavailable.",
                    status_code=503,
                ) from exc
            mentions = parse_mention_json(answer, prompt)
            mention_source = "model"
            mention_model_id: str | None = selected_id
        else:
            mentions = _manual_mentions(prompt, manual_mentions)
            mention_source = "manual"
            mention_model_id = None

        raw_candidates = self.grounding_backend.predict(
            image,
            [item.label for item in mentions],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        objects, warnings = select_grounded_objects(
            prompt, mentions, raw_candidates, width=image.width, height=image.height
        )
        raw_segmentations = self.sam_backend.predict(
            image, [item.bbox_xyxy for item in objects]
        )
        segmentations = [_coerce_sam_prediction(item) for item in raw_segmentations]
        if len(segmentations) != len(objects):
            raise RuntimeError("SAM returned the wrong number of masks")
        by_id: dict[str, SamPrediction] = {}
        mask_images: dict[str, Image.Image] = {}
        for item, segmentation in zip(objects, segmentations, strict=True):
            if segmentation.mask.shape != (image.height, image.width):
                raise RuntimeError(
                    f"SAM mask shape {segmentation.mask.shape} != "
                    f"image shape {(image.height, image.width)}"
                )
            by_id[item.object_id] = segmentation
            mask_images[item.object_id] = mask_to_rgba(segmentation.mask)

        spans = {item.object_id: item.char_span for item in objects}
        try:
            raw_extraction = self.model_manager.extract_coordinates(
                selected_id, image, prompt, spans, selected_layer
            )
            reported_layer = raw_extraction.get("source_layer")
            if (
                isinstance(reported_layer, bool)
                or not isinstance(reported_layer, int)
                or int(reported_layer) != selected_layer
            ):
                raise ModelWorkerError(
                    f"Coordinate worker reported layer {reported_layer!r}, "
                    f"expected L{selected_layer}"
                )
        except TimeoutError as exc:
            LOGGER.exception("Coordinate extraction timed out for %s", selected_id)
            raise InferenceStageError(
                "coordinate_extraction",
                "model_unavailable",
                "The selected model is temporarily unavailable.",
                status_code=503,
            ) from exc
        except ModelWorkerError as exc:
            LOGGER.exception("Coordinate worker failed for %s", selected_id)
            raise InferenceStageError(
                "coordinate_extraction",
                "model_unavailable",
                "The selected model is temporarily unavailable.",
                status_code=503,
            ) from exc
        extraction: CoordinateExtraction = _coerce_coordinate_extraction(raw_extraction)
        selected_key = (selected_id, selected_layer)
        coordinates = project_hidden_states(
            extraction.hidden_states, self.axes_by_model_layer[selected_key]
        )
        payload = build_web_payload(
            image=image,
            prompt=prompt,
            run_id=run_id,
            asset_base=asset_base,
            objects=objects,
            segmentations=by_id,
            extraction=extraction,
            coordinates=coordinates,
            warnings=warnings,
            mention_source=mention_source,
        )
        axis_metadata = self.axis_metadata_by_model_layer[selected_key]
        payload_metadata = payload["metadata"]
        payload_metadata.update(
            {
                "model_id": selected_id,
                "model_label": spec.label,
                "model_path": spec.repo,
                "model_revision": spec.revision,
                "source_layer": selected_layer,
                "axis_artifact_id": axis_metadata["artifact_id"],
                "axis_bundle": Path(spec.axes_path).parent.name,
                "selected_axes": dict(axis_metadata),
                "transformers_version": spec.transformers_version,
                "layer_numbering": "zero_based_post_block",
                "object_token_protocol": "last_overlapping_subtoken",
                "mention_model_id": mention_model_id,
                "grounding_box_threshold": self.box_threshold,
                "grounding_text_threshold": self.text_threshold,
            }
        )
        if axis_metadata.get("artifact_selected_layer") is not None:
            payload_metadata["axis_exported_layer"] = selected_layer
            payload_metadata["axis_artifact_selected_layer"] = int(
                axis_metadata["artifact_selected_layer"]
            )
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return InferenceResult(payload=payload, masks=mask_images, warnings=warnings)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.model_manager, "close", None)
        if close is not None:
            close()

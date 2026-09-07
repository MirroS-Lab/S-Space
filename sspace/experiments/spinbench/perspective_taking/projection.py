"""Construct SpinBench prompts and score rotated projections."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sspace.core.prompts.rendering import RenderedPrompt, render_prompt_pair
from sspace.core.prompts.templates import AXIS_ORDER

from .adapter import (
    VIEW_ROTATIONS,
    SpinBenchSample,
    rotate_coordinates,
    spinbench_target_margin,
)


PROJECTION_PROTOCOL = "selected_layer_margin_rotation_matrix_v1"
PROMPT_STYLE = "baseline"
PROBE_CONTEXTS = {"none", "official_source_premise"}
SCORE_METHODS = ("margin",)


@dataclass(frozen=True)
class SpinBenchProjectionRecord:
    """One traceable EVAL-09 source score, rotation, and A/B prediction."""

    evaluation_id: str
    sample_id: str
    global_sample_index: int
    row_index: int
    premise_mode: str
    transform: str
    source_group: str
    target_view: str
    target_property: str
    object_a: str
    object_b: str
    gold_answer: str
    selected_layer: int
    original_margin: float
    swapped_margin: float
    margin: float
    rotation_matrix_hvd: list[list[float]]
    source_coordinates: dict[str, list[float]]
    target_coordinates: dict[str, list[float]]
    target_margins: dict[str, float]
    predictions: dict[str, str]
    correct: dict[str, bool]
    projection_prompt_original: str
    projection_prompt_swapped: str
    source_premise: str | None
    image_path: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without changing numeric evidence."""
        return asdict(self)


def _prefix_prompt(prompt: RenderedPrompt, prefix: str) -> RenderedPrompt:
    offset = len(prefix)
    return RenderedPrompt(
        text=prefix + prompt.text,
        query_span=(
            prompt.query_span[0] + offset,
            prompt.query_span[1] + offset,
        ),
        reference_span=(
            prompt.reference_span[0] + offset,
            prompt.reference_span[1] + offset,
        ),
    )


def render_spinbench_projection_prompts(
    sample: SpinBenchSample,
    probe_context: str,
) -> tuple[RenderedPrompt, RenderedPrompt]:
    """Render frozen original/swapped source-relation prompts (EVAL-09).

    Args:
        sample: Strict official SpinBench row with A/B bindings.
        probe_context: ``none`` or ``official_source_premise``.

    Returns:
        Original A-as-query and swapped B-as-query prompts with exact spans.

    Raises:
        ValueError: Context conflicts with the sample's official premise mode.

    Side effects:
        None.
    """
    if probe_context not in PROBE_CONTEXTS:
        raise ValueError(f"Unknown SpinBench probe context {probe_context!r}")
    prompts = render_prompt_pair(
        PROMPT_STYLE,
        sample.source_group,
        sample.object_a,
        sample.object_b,
    )
    if probe_context == "none":
        if sample.premise_mode != "without":
            raise ValueError("No-context projection requires without-premise rows")
        return prompts
    if sample.premise_mode != "with" or sample.source_premise is None:
        raise ValueError("Official premise context requires with-premise rows")
    prefix = sample.source_premise + "\n"
    return tuple(_prefix_prompt(prompt, prefix) for prompt in prompts)  # type: ignore[return-value]


def source_coordinates(source_margin: float, source_group: str) -> np.ndarray:
    """Place one observed source relation in sparse H/V/D coordinates.

    SpinBench source prompts measure only horizontal (``lr``) or distance
    (``fn``). Unobserved coordinates remain zero; no answer rule fills them.
    """
    margin = float(source_margin)
    if not np.isfinite(margin):
        raise ValueError("SpinBench source margin must be finite")
    if source_group not in {"horizontal", "distance"}:
        raise ValueError("SpinBench source group must be horizontal or distance")
    coordinates = np.zeros(3, dtype=np.float64)
    coordinates[AXIS_ORDER.index(source_group)] = margin
    return coordinates


def _rotation_prediction(
    source_margin: float,
    sample: SpinBenchSample,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    source = source_coordinates(source_margin, sample.source_group)
    target = rotate_coordinates(source, sample.target_view)
    target_margin = spinbench_target_margin(target, sample.target_property)
    return source, target, target_margin, "A" if target_margin > 0 else "B"


def score_spinbench_projection(
    samples: Sequence[SpinBenchSample],
    margins: np.ndarray,
    prompt_pairs: Sequence[Sequence[RenderedPrompt]],
    selected_layer: int,
    evaluation_id: str,
) -> tuple[SpinBenchProjectionRecord, ...]:
    """Score EVAL-09 using only explicit H/V/D rotation matrices.

    The directed original margin is placed in a sparse H/V/D
    vector, multiplied by ``R_view``, and read on the target property axis.
    No transform-to-answer lookup table is used.

    Returns:
        Ordered immutable records for every official row.

    Raises:
        ValueError: Shapes, finite values, prompt order, layer, or ties fail.

    Side effects:
        None.
    """
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (len(samples), 2) or not np.isfinite(values).all():
        raise ValueError("SpinBench margins must be finite [N,2]")
    if len(prompt_pairs) != len(samples) or selected_layer < 0:
        raise ValueError("SpinBench prompts or selected layer are invalid")
    if not evaluation_id.strip():
        raise ValueError("SpinBench evaluation ID must be non-empty")
    records = []
    for index, (sample, pair) in enumerate(zip(samples, prompt_pairs, strict=True)):
        prompts = tuple(pair)
        if len(prompts) != 2:
            raise ValueError("SpinBench scoring needs two projection prompts")
        original, swapped = (float(value) for value in values[index])
        source_values = {"margin": original}
        projected = {
            method: _rotation_prediction(margin, sample)
            for method, margin in source_values.items()
        }
        predictions = {method: result[3] for method, result in projected.items()}
        records.append(
            SpinBenchProjectionRecord(
                evaluation_id=evaluation_id,
                sample_id=sample.sample_id,
                global_sample_index=index,
                row_index=sample.row_index,
                premise_mode=sample.premise_mode,
                transform=sample.transform,
                source_group=sample.source_group,
                target_view=sample.target_view,
                target_property=sample.target_property,
                object_a=sample.object_a,
                object_b=sample.object_b,
                gold_answer=sample.answer,
                selected_layer=selected_layer,
                original_margin=original,
                swapped_margin=swapped,
                margin=original,
                rotation_matrix_hvd=VIEW_ROTATIONS[sample.target_view].tolist(),
                source_coordinates={
                    method: result[0].tolist() for method, result in projected.items()
                },
                target_coordinates={
                    method: result[1].tolist() for method, result in projected.items()
                },
                target_margins={
                    method: result[2] for method, result in projected.items()
                },
                predictions=predictions,
                correct={
                    method: prediction == sample.answer
                    for method, prediction in predictions.items()
                },
                projection_prompt_original=prompts[0].text,
                projection_prompt_swapped=prompts[1].text,
                source_premise=sample.source_premise,
                image_path=str(sample.image_path),
                metadata=dict(sample.metadata),
            )
        )
    return tuple(records)


def summarize_spinbench_projection(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize margin-based EVAL-09 scoring by semantic group."""
    if not records:
        raise ValueError("Cannot summarize empty SpinBench projection records")
    identities = {
        (
            str(record["evaluation_id"]),
            str(record["premise_mode"]),
            int(record["selected_layer"]),
        )
        for record in records
    }
    indices = [int(record["global_sample_index"]) for record in records]
    if len(identities) != 1 or len(indices) != len(set(indices)):
        raise ValueError("SpinBench projection summary mixes runs or sample indices")
    buckets: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {method: [0, 0] for method in SCORE_METHODS}
    )
    for record in records:
        correct = record["correct"]
        if set(correct) != set(SCORE_METHODS):
            raise ValueError("SpinBench record has an unknown score-method set")
        for key in (
            "all",
            f"transform:{record['transform']}",
            f"source_group:{record['source_group']}",
            f"target_view:{record['target_view']}",
            f"target_property:{record['target_property']}",
        ):
            for method in SCORE_METHODS:
                buckets[key][method][0] += int(bool(correct[method]))
                buckets[key][method][1] += 1

    def score(key: str) -> dict[str, dict[str, int | float]]:
        return {
            method: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for method, values in buckets[key].items()
        }

    def grouped(prefix: str) -> dict[str, Any]:
        return {
            key.removeprefix(prefix): score(key)
            for key in sorted(buckets)
            if key.startswith(prefix)
        }

    identity = next(iter(identities))
    return {
        "evaluation_id": identity[0],
        "premise_mode": identity[1],
        "selected_layer": identity[2],
        "sample_count": len(records),
        "projection_protocol": PROJECTION_PROTOCOL,
        "overall": score("all"),
        "by_transform": grouped("transform:"),
        "by_source_group": grouped("source_group:"),
        "by_target_view": grouped("target_view:"),
        "by_target_property": grouped("target_property:"),
    }

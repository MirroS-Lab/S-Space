"""Compute layer-matched pairwise margin diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from sspace.core.prompts.templates import AXIS_ORDER

from .decision import classify_pairwise_margin
from .schema import ENDPOINTS, PairwiseEvaluationSample


@dataclass(frozen=True)
class LayerwisePairwiseMetric:
    """Store accuracy and class-balance diagnostics for one post-block layer."""

    layer_id: int
    correct: int
    total: int
    overall_accuracy: float
    axis_accuracy: tuple[float, float, float]
    worst_axis_accuracy: float
    positive_prediction_rate: tuple[float, float, float]

    def to_dict(self) -> dict:
        """Return a JSON-compatible record without changing metric values."""
        return asdict(self)


def score_layerwise_margins(
    samples: Sequence[PairwiseEvaluationSample],
    original_margins: np.ndarray,
    layer_ids: Sequence[int],
) -> tuple[LayerwisePairwiseMetric, ...]:
    """Classify every layer by the sign of its original pairwise margin.

    For sample ``i`` in axis group ``g`` and layer ``l``, this function
    predicts the positive endpoint iff

    ``v[l,g]^T(h_query[i,l] - h_reference[i,l]) > 0``.

    Positive endpoints are right, below, and close. Ground-truth labels enter
    only after margin extraction, when this function computes accuracy.

    Args:
        samples: Ordered pairwise benchmark rows.
        original_margins: Finite layer-matched scores ``[N,L]``.
        layer_ids: Unique increasing post-block IDs ``[L]``.

    Returns:
        One metric per layer. Axis tuples follow H/V/D order.

    Raises:
        ValueError: Shapes, layers, benchmark groups, labels, or values fail.

    Side effects:
        None.

    Example:
        ``metrics = score_layerwise_margins(samples, margins[:, 0], layers)``
    """
    layers = tuple(int(value) for value in layer_ids)
    margins = np.asarray(original_margins, dtype=np.float64)
    if margins.shape != (len(samples), len(layers)):
        raise ValueError("Original margins must have shape [N,L]")
    if not layers or layers != tuple(sorted(set(layers))) or layers[0] < 0:
        raise ValueError("Layer IDs must be unique, increasing, and nonnegative")
    if not np.isfinite(margins).all():
        raise ValueError("Original margins must be finite")
    for sample in samples:
        sample.validate()

    group_array = np.asarray([sample.group for sample in samples])
    label_array = np.asarray([sample.gold_endpoint for sample in samples])
    metrics = []
    for layer_position, layer_id in enumerate(layers):
        correct = np.empty(len(samples), dtype=bool)
        positive = margins[:, layer_position] > 0.0
        for row, sample in enumerate(samples):
            prediction = classify_pairwise_margin(
                sample.group, margins[row, layer_position]
            )
            correct[row] = prediction != "tie" and prediction == sample.gold_endpoint
        axis_accuracy = []
        positive_rate = []
        for group in AXIS_ORDER:
            selected = group_array == group
            if not selected.any():
                raise ValueError(f"Benchmark has no {group} samples")
            allowed = set(ENDPOINTS[group])
            if not set(label_array[selected]).issubset(allowed):
                raise ValueError(f"Benchmark contains invalid {group} labels")
            axis_accuracy.append(float(correct[selected].mean()))
            positive_rate.append(float(positive[selected].mean()))
        metrics.append(
            LayerwisePairwiseMetric(
                layer_id=layer_id,
                correct=int(correct.sum()),
                total=len(samples),
                overall_accuracy=float(correct.mean()),
                axis_accuracy=tuple(axis_accuracy),
                worst_axis_accuracy=min(axis_accuracy),
                positive_prediction_rate=tuple(positive_rate),
            )
        )
    return tuple(metrics)


def diagnostic_best_layer(metrics: Sequence[LayerwisePairwiseMetric]) -> int:
    """Choose the diagnostic maximum by overall, worst axis, then shallower ID."""
    if not metrics:
        raise ValueError("Layerwise diagnostic metrics cannot be empty")
    if len({metric.layer_id for metric in metrics}) != len(metrics):
        raise ValueError("Layerwise diagnostic layer IDs must be unique")
    return max(
        metrics,
        key=lambda metric: (
            metric.overall_accuracy,
            metric.worst_axis_accuracy,
            -metric.layer_id,
        ),
    ).layer_id

"""Score validation readouts and select the default S-Space layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from sspace.core.prompts.templates import AXIS_ORDER, PROMPT_STYLE_ORDER


POSITIVE_ENDPOINT = {
    "horizontal": "right",
    "vertical": "below",
    "distance": "close",
}
DECISION_PROTOCOL = "margin_sign_v1"
PAIRWISE_SCORE = "v^T(h_query-h_reference)"
SELECTION_METRIC = "margin_sign_overall_accuracy"
SELECTION_RULE = (
    "overall_accuracy",
    "worst_axis_accuracy",
    "shallower_layer_id",
)


@dataclass(frozen=True)
class LayerMetrics:
    """Validation metrics for one post-block layer.

    ``axis_accuracy`` follows horizontal, vertical, distance order. Metrics are
    computed per construction sample after averaging its five prompt-style
    margins, so prompt variants do not count as independent examples.
    """

    layer_id: int
    correct: int
    total: int
    overall_accuracy: float
    axis_accuracy: tuple[float, float, float]
    worst_axis_accuracy: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable record without changing its values."""
        return asdict(self)


def validation_metrics(
    original_margins: np.ndarray,
    groups: Sequence[str],
    endpoint_labels: Sequence[str],
    sample_ids: Sequence[str],
    layer_ids: Sequence[int],
) -> tuple[LayerMetrics, ...]:
    """Compute frozen per-layer validation accuracy (TRN-04).

    Each row is one ``sample × prompt style`` task. For sample ``s``, layer
    ``l``, and its axis ``g``, this function averages
    ``m_original[s,t,l]`` across prompt styles ``t``. A positive
    mean predicts right, below, or close. Endpoint labels are read only here;
    they never affect axis construction.

    Args:
        original_margins: Finite original-prompt margins ``[N, L]``.
        groups: Length-``N`` axis names.
        endpoint_labels: Length-``N`` validation labels.
        sample_ids: Length-``N`` IDs; repeated rows are prompt styles.
        layer_ids: Length-``L`` post-block layer IDs.

    Returns:
        One :class:`LayerMetrics` per layer, in ``layer_ids`` order.

    Raises:
        ValueError: Shapes, labels, grouping, style balance, or finiteness fail.

    Side effects:
        None.
    """
    margins = np.asarray(original_margins, dtype=np.float64)
    layers = tuple(int(value) for value in layer_ids)
    row_count = margins.shape[0] if margins.ndim == 2 else -1
    if margins.ndim != 2 or row_count <= 0:
        raise ValueError("Validation margins must have shape [N,L]")
    if (
        len(layers) != margins.shape[1]
        or layers != tuple(sorted(set(layers)))
        or layers[0] < 0
    ):
        raise ValueError(
            "Layer IDs must match margin columns and be unique and increasing"
        )
    if any(
        len(values) != row_count for values in (groups, endpoint_labels, sample_ids)
    ):
        raise ValueError("Validation row metadata lengths do not match margins")
    if not np.isfinite(margins).all():
        raise ValueError("Validation margins must be finite")

    group_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    rows_by_sample: dict[str, list[int]] = {}
    for row, sample_id in enumerate(sample_ids):
        rows_by_sample.setdefault(sample_id, []).append(row)

    sample_scores, sample_groups, sample_labels = [], [], []
    style_count: int | None = None
    for sample_id, rows in rows_by_sample.items():
        row_groups = {groups[row] for row in rows}
        row_labels = {endpoint_labels[row] for row in rows}
        if len(row_groups) != 1 or len(row_labels) != 1:
            raise ValueError(f"Validation sample {sample_id!r} changes group or label")
        if style_count is None:
            style_count = len(rows)
        elif len(rows) != style_count:
            raise ValueError("Validation samples have unequal prompt-style counts")
        group = next(iter(row_groups))
        if group not in group_index:
            raise ValueError(f"Unknown validation group {group!r}")
        label = next(iter(row_labels))
        endpoints = {
            "horizontal": {"left", "right"},
            "vertical": {"above", "below"},
            "distance": {"far", "close"},
        }[group]
        if label not in endpoints:
            raise ValueError(f"Label {label!r} is invalid for {group}")
        sample_scores.append(margins[rows].mean(axis=0))
        sample_groups.append(group)
        sample_labels.append(label)

    if style_count != len(PROMPT_STYLE_ORDER):
        raise ValueError(
            f"Validation samples require exactly {len(PROMPT_STYLE_ORDER)} prompt styles"
        )
    if set(sample_groups) != set(AXIS_ORDER):
        raise ValueError("Validation samples must cover H/V/D exactly")

    scores = np.stack(sample_scores)
    metrics = []
    for layer_position, layer_id in enumerate(layers):
        nonzero = scores[:, layer_position] != 0.0
        correct = nonzero & np.asarray(
            [
                (scores[row, layer_position] > 0)
                == (sample_labels[row] == POSITIVE_ENDPOINT[sample_groups[row]])
                for row in range(len(sample_groups))
            ],
            dtype=bool,
        )
        axis_accuracy = tuple(
            float(correct[np.asarray(sample_groups) == group].mean())
            for group in AXIS_ORDER
        )
        metrics.append(
            LayerMetrics(
                layer_id=int(layer_id),
                correct=int(correct.sum()),
                total=int(correct.size),
                overall_accuracy=float(correct.mean()),
                axis_accuracy=axis_accuracy,
                worst_axis_accuracy=min(axis_accuracy),
            )
        )
    return tuple(metrics)


def select_default_layer(metrics: Sequence[LayerMetrics]) -> int:
    """Select by overall accuracy, worst-axis accuracy, then shallower ID.

    Args:
        metrics: Non-empty metrics with unique layer IDs.

    Returns:
        The selected post-block layer ID.

    Raises:
        ValueError: Metrics are empty, duplicated, or non-finite.

    Side effects:
        None.
    """
    if not metrics:
        raise ValueError("Layer selection requires validation metrics")
    if len({value.layer_id for value in metrics}) != len(metrics):
        raise ValueError("Validation layer IDs must be unique")
    values = [
        number
        for metric in metrics
        for number in (metric.overall_accuracy, metric.worst_axis_accuracy)
    ]
    if not np.isfinite(values).all():
        raise ValueError("Validation metrics must be finite")
    return max(
        metrics,
        key=lambda value: (
            value.overall_accuracy,
            value.worst_axis_accuracy,
            -value.layer_id,
        ),
    ).layer_id


def build_validation_evidence(
    dataset_id: str,
    dataset_fingerprint: str,
    metrics: Sequence[LayerMetrics],
    train_image_ids: Sequence[int],
    validation_image_ids: Sequence[int],
) -> dict:
    """Build complete disjoint margin-sign layer-selection evidence (TRN-04).

    Args:
        dataset_id: Exact validation dataset identifier.
        dataset_fingerprint: Lowercase SHA-256 validation dataset fingerprint.
        metrics: Per-layer zero-boundary validation metrics.
        train_image_ids: Selected COCO construction image IDs.
        validation_image_ids: Selected COCO readout image IDs.

    Returns:
        The complete ``construction.validation`` manifest record.

    Raises:
        ValueError: Dataset identity, sample counts, image uniqueness,
            disjointness, or layer metrics are invalid.

    Side effects:
        None.
    """
    if not dataset_id:
        raise ValueError("Validation dataset ID must be non-empty")
    if len(dataset_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_fingerprint
    ):
        raise ValueError("Validation dataset fingerprint must be lowercase SHA-256")
    metric_values = tuple(metrics)
    selected_layer = select_default_layer(metric_values)
    totals = {metric.total for metric in metric_values}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        raise ValueError("Validation layers must report one positive sample count")
    sample_count = next(iter(totals))
    train_ids = tuple(int(value) for value in train_image_ids)
    validation_ids = tuple(int(value) for value in validation_image_ids)
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Selected construction image IDs must be unique")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Selected validation image IDs must be unique")
    if len(validation_ids) != sample_count:
        raise ValueError("Validation image count differs from layer metrics")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(
            f"Construction and validation selections overlap on {len(overlap)} images"
        )
    per_layer = [metric.to_dict() for metric in metric_values]
    selected_metrics = next(
        row for row in per_layer if int(row["layer_id"]) == selected_layer
    )
    return {
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_count": sample_count,
        "train_image_overlap_count": 0,
        "decision_protocol": DECISION_PROTOCOL,
        "score": PAIRWISE_SCORE,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": list(SELECTION_RULE),
        "selected_layer": selected_layer,
        "selected_metrics": selected_metrics,
        "per_layer": per_layer,
    }

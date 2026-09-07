"""Map pairwise margins to endpoints and metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from sspace.core.artifacts import SSpaceArtifact
from sspace.core.prompts.templates import AXIS_ORDER
from .decision import classify_pairwise_margin
from .schema import PairwiseEvaluationSample, PairwiseScoreRecord


def evaluate_pairwise_scores(
    artifact: SSpaceArtifact,
    samples: Sequence[PairwiseEvaluationSample],
    original_margins: np.ndarray,
    swapped_margins: np.ndarray,
    projection_prompts: Sequence[tuple[str, str]],
    layer: int,
) -> tuple[PairwiseScoreRecord, ...]:
    """Map precomputed pairwise margins to benchmark endpoints (EVAL-03).

    Only the directed original margin is classified. Positive values map to
    right, below, or close; negative values map to the opposite endpoint. An
    exact zero is recorded as a tie and always marked incorrect.

    Args:
        artifact: Validated model-native S-Space artifact.
        samples: Ordered pairwise benchmark rows.
        original_margins: Finite original-prompt scores ``[N]``.
        swapped_margins: Finite swapped-prompt scores ``[N]``.
        projection_prompts: Ordered original/swapped prompt text per sample.
        layer: Exported post-block layer ID.

    Returns:
        One immutable, traceable score record per input sample.

    Raises:
        ValueError: Protocol, shape, sample, or finite-value checks fail.
        KeyError: The artifact does not export ``layer``.

    Side effects:
        None.
    """
    artifact.validate()
    artifact.axes(layer)
    original = np.asarray(original_margins, dtype=np.float64)
    swapped = np.asarray(swapped_margins, dtype=np.float64)
    count = len(samples)
    if original.shape != (count,) or swapped.shape != (count,):
        raise ValueError("Pairwise evaluation margins must have shape [N]")
    if len(projection_prompts) != count:
        raise ValueError("Projection prompts and samples have different lengths")
    if not np.isfinite(original).all() or not np.isfinite(swapped).all():
        raise ValueError("Pairwise evaluation margins must be finite")
    for sample in samples:
        sample.validate()
    records = []
    for index, sample in enumerate(samples):
        margin = float(original[index])
        prediction = classify_pairwise_margin(sample.group, margin)
        records.append(
            PairwiseScoreRecord(
                benchmark=sample.benchmark,
                sample_id=sample.sample_id,
                split=sample.split,
                layer_id=int(layer),
                group=sample.group,
                query=sample.query,
                reference=sample.reference,
                original_margin=float(original[index]),
                swapped_margin=float(swapped[index]),
                margin=margin,
                predicted_endpoint=prediction,
                gold_endpoint=sample.gold_endpoint,
                correct=(prediction != "tie" and prediction == sample.gold_endpoint),
                decision_protocol="margin_sign_v1",
                direct_prompt=sample.direct_prompt,
                projection_prompt_original=projection_prompts[index][0],
                projection_prompt_swapped=projection_prompts[index][1],
                image_sha256=sample.image_sha256,
                metadata=dict(sample.metadata),
            )
        )
    return tuple(records)


def summarize_pairwise_records(
    records: Sequence[PairwiseScoreRecord],
) -> dict[str, object]:
    """Aggregate overall, per-axis, and per-endpoint accuracy from records."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    benchmark = {record.benchmark for record in records}
    split = {record.split for record in records}
    layer = {record.layer_id for record in records}
    protocol = {record.decision_protocol for record in records}
    if any(len(values) != 1 for values in (benchmark, split, layer, protocol)):
        raise ValueError("Summary records mix benchmark, split, layer, or protocol")
    for record in records:
        expected = (
            record.predicted_endpoint != "tie"
            and record.predicted_endpoint == record.gold_endpoint
        )
        if record.correct != expected:
            raise ValueError(
                f"Stored correctness differs from prediction for {record.sample_id}"
            )

    def score(selected: Sequence[PairwiseScoreRecord]) -> dict[str, object]:
        return {
            "correct": sum(record.correct for record in selected),
            "total": len(selected),
            "accuracy": sum(record.correct for record in selected) / len(selected),
        }

    endpoint_counts = Counter(record.gold_endpoint for record in records)
    return {
        "benchmark": next(iter(benchmark)),
        "split": next(iter(split)),
        "layer_id": next(iter(layer)),
        "decision_protocol": next(iter(protocol)),
        "overall": score(records),
        "by_axis": {
            group: score([record for record in records if record.group == group])
            for group in AXIS_ORDER
            if any(record.group == group for record in records)
        },
        "by_endpoint": {
            endpoint: score(
                [record for record in records if record.gold_endpoint == endpoint]
            )
            for endpoint in sorted(endpoint_counts)
        },
    }

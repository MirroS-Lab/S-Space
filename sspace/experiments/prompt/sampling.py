"""Select bounded prompt-control samples without dropping relation endpoints."""

from __future__ import annotations

from collections.abc import Sequence

from sspace.experiments.common.pairwise.schema import (
    ENDPOINTS,
    PairwiseEvaluationSample,
)


def balanced_prompt_subset(
    samples: Sequence[PairwiseEvaluationSample], limit: int | None
) -> tuple[PairwiseEvaluationSample, ...]:
    """Return a source-ordered bounded subset covering all six endpoints."""
    values = tuple(samples)
    if limit is None or limit >= len(values):
        return values
    endpoint_order = tuple(
        endpoint for group in ENDPOINTS.values() for endpoint in group
    )
    if limit < len(endpoint_order):
        raise ValueError(f"Prompt smoke limit must be at least {len(endpoint_order)}")
    buckets = {
        endpoint: [
            index
            for index, sample in enumerate(values)
            if sample.gold_endpoint == endpoint
        ]
        for endpoint in endpoint_order
    }
    if any(not indices for indices in buckets.values()):
        raise ValueError("Prompt dataset does not contain every relation endpoint")
    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for endpoint in endpoint_order:
            indices = buckets[endpoint]
            if depth < len(indices):
                selected.append(indices[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return tuple(values[index] for index in sorted(selected))

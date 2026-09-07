"""Apply the formal zero-boundary rule to directed pairwise margins."""

from __future__ import annotations

import math

from .schema import ENDPOINTS


def classify_pairwise_margin(group: str, margin: float) -> str:
    """Return the endpoint predicted by a finite directed margin.

    Positive margins map to right, below, or close; negative margins map to
    left, above, or far. An exact zero has no direction and returns ``"tie"``.

    Args:
        group: One of horizontal, vertical, or distance.
        margin: Directed score ``v_g^T(h_query-h_reference)``.

    Returns:
        One endpoint name or ``"tie"`` for an exact zero.

    Raises:
        ValueError: The group is unknown or the margin is non-finite.

    Side effects:
        None.
    """
    if group not in ENDPOINTS:
        raise ValueError(f"Unsupported pairwise group {group!r}")
    value = float(margin)
    if not math.isfinite(value):
        raise ValueError("Pairwise margin must be finite")
    negative, positive = ENDPOINTS[group]
    if value > 0.0:
        return positive
    if value < 0.0:
        return negative
    return "tie"

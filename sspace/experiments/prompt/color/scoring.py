"""Define the non-spatial color-prompt control."""

from __future__ import annotations

import hashlib

import numpy as np

from sspace.core.prompts.rendering import RenderedObjectPrompt


COLOR_CONTROL_PROMPT_PROTOCOL = "two_object_color_question_counterbalanced_v1"


def query_mentioned_first(sample_id: str, seed: int) -> bool:
    """Choose mention order deterministically without reading spatial labels.

    Args:
        sample_id: Stable non-empty benchmark sample identifier.
        seed: Frozen dataset-order seed.

    Returns:
        ``True`` when the query object precedes the reference object.

    Raises:
        ValueError: ``sample_id`` is empty or has surrounding whitespace.

    Side effects:
        None.

    Example:
        ``query_mentioned_first("sample-1", 42)``
    """
    if not sample_id or sample_id != sample_id.strip():
        raise ValueError("Color-control sample ID must be non-empty and stripped")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return digest[0] < 128


def render_color_control_prompt(
    query: str,
    reference: str,
    *,
    query_first: bool,
) -> RenderedObjectPrompt:
    """Render one non-spatial color question with identity-aligned spans.

    Args:
        query: Original benchmark query-object name.
        reference: Original benchmark reference-object name.
        query_first: Whether ``query`` appears before ``reference``.

    Returns:
        Exact question and half-open spans keyed by ``query`` and ``reference``.

    Raises:
        ValueError: Object names are empty, padded, or identical.

    Side effects:
        None.

    Example:
        ``render_color_control_prompt("cup", "plate", query_first=True)``
    """
    if (
        not query
        or not reference
        or query != query.strip()
        or reference != reference.strip()
        or query == reference
    ):
        raise ValueError("Color-control object names must be distinct and stripped")
    first_role, second_role = (
        ("query", "reference") if query_first else ("reference", "query")
    )
    values = {"query": query, "reference": reference}
    prefix = "What colors are the "
    middle = " and the "
    first = values[first_role]
    second = values[second_role]
    text = f"{prefix}{first}{middle}{second}?"
    first_start = len(prefix)
    second_start = first_start + len(first) + len(middle)
    return RenderedObjectPrompt(
        text=text,
        object_spans={
            first_role: (first_start, first_start + len(first)),
            second_role: (second_start, second_start + len(second)),
        },
    )


def project_color_control_states(
    query_states: np.ndarray,
    reference_states: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    """Project role-aligned color-prompt states into layerwise H/V/D margins.

    Implements ``m[l,g] = v[l,g]^T(h_query[l] - h_reference[l])``. Mention
    order never changes the role subtraction. No threshold or color-answer
    correctness enters this calculation.

    Args:
        query_states: Query last-subtoken states ``[L,D]``.
        reference_states: Reference last-subtoken states ``[L,D]``.
        axes: Unit axes ``[L,3,D]`` in H/V/D order.

    Returns:
        Finite margins ``[L,3]``.

    Raises:
        ValueError: Shapes differ or axes are not unit normalized.
        FloatingPointError: Inputs or outputs are non-finite.

    Side effects:
        None.

    Example:
        ``margins = project_color_control_states(query, reference, axes)``
    """
    query_values = np.asarray(query_states, dtype=np.float64)
    reference_values = np.asarray(reference_states, dtype=np.float64)
    axis_values = np.asarray(axes, dtype=np.float64)
    if query_values.ndim != 2 or reference_values.shape != query_values.shape:
        raise ValueError("Color-control states must share shape [L,D]")
    if axis_values.shape != (query_values.shape[0], 3, query_values.shape[1]):
        raise ValueError("Color-control axes must have shape [L,3,D]")
    if not (
        np.isfinite(query_values).all()
        and np.isfinite(reference_values).all()
        and np.isfinite(axis_values).all()
    ):
        raise FloatingPointError("Color-control states and axes must be finite")
    norms = np.linalg.norm(axis_values, axis=2)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Color-control projection requires unit H/V/D axes")
    margins = np.einsum("lgd,ld->lg", axis_values, query_values - reference_values)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Color-control margins are non-finite")
    return margins

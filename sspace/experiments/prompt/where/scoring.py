"""Define the single-object absolute-question control."""

from __future__ import annotations

import numpy as np

from sspace.core.prompts.rendering import RenderedObjectPrompt


TARGET_ROLE = "target"
SINGLE_OBJECT_PROMPT_PROTOCOL = "single_object_where_pair_v1"


def render_single_object_where_prompt(object_name: str) -> RenderedObjectPrompt:
    """Render exactly ``Where is the {object}?`` with one aligned object span.

    Args:
        object_name: Non-empty stripped object phrase from a benchmark row.

    Returns:
        The absolute-location question and its target-object character span.

    Raises:
        ValueError: ``object_name`` is empty or contains surrounding whitespace.

    Side effects:
        None.
    """
    if not object_name or object_name != object_name.strip():
        raise ValueError("Single-object name must be non-empty and stripped")
    prefix = "Where is the "
    text = f"{prefix}{object_name}?"
    start = len(prefix)
    return RenderedObjectPrompt(
        text=text,
        object_spans={TARGET_ROLE: (start, start + len(object_name))},
    )


def project_separate_object_states(
    query_states: np.ndarray,
    reference_states: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    """Project separately prompted objects into layerwise H/V/D margins.

    Implements
    ``m[l,g] = v[l,g]^T(h_query_where[l] - h_reference_where[l])``. Each state
    comes from an independent single-object question over the same image. No
    absolute origin, fitted offset, or threshold enters the calculation.

    Args:
        query_states: Query-object last-subtoken states ``[L,D]``.
        reference_states: Reference-object last-subtoken states ``[L,D]``.
        axes: Unit H/V/D axes ``[L,3,D]``.

    Returns:
        Finite layerwise margins ``[L,3]``.

    Raises:
        ValueError: State shapes differ or axes are not unit normalized.
        FloatingPointError: Inputs or projections are non-finite.

    Side effects:
        None.
    """
    query = np.asarray(query_states, dtype=np.float64)
    reference = np.asarray(reference_states, dtype=np.float64)
    basis = np.asarray(axes, dtype=np.float64)
    if query.ndim != 2 or reference.shape != query.shape:
        raise ValueError("Separate object states must share shape [L,D]")
    if basis.shape != (query.shape[0], 3, query.shape[1]):
        raise ValueError("Single-object control axes must have shape [L,3,D]")
    if not (
        np.isfinite(query).all()
        and np.isfinite(reference).all()
        and np.isfinite(basis).all()
    ):
        raise FloatingPointError("Single-object states and axes must be finite")
    if not np.allclose(np.linalg.norm(basis, axis=2), 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Single-object projection requires unit H/V/D axes")
    margins = np.einsum("lgd,ld->lg", basis, query - reference)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Single-object margins are non-finite")
    return margins

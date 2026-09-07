"""Render and score the H* multi-view order control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sspace.core.prompts.rendering import RenderedMultiImagePrompt
from sspace.core.prompts.templates import AXIS_ORDER

from .schema import MultiViewSample


TARGET_ROLE = "target"


def render_multiview_order_prompt(object_name: str) -> RenderedMultiImagePrompt:
    """Render the causal H* question for the object shown in View 1."""
    if not object_name or object_name != object_name.strip():
        raise ValueError("Multi-view object name must be non-empty and stripped")
    prefix = "In View 1, where is the "
    question = f"{prefix}{object_name}?"
    start = len(prefix)
    return RenderedMultiImagePrompt(
        text_segments=("", question),
        object_text_index=1,
        object_spans={TARGET_ROLE: (start, start + len(object_name))},
    )


def score_layerwise_multiview_order(
    samples: Sequence[MultiViewSample],
    ordered_object_states: np.ndarray,
    axes: np.ndarray,
    layer_ids: Sequence[int],
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Score layer-matched H* AB/BA object-state deltas.

    The score is ``v[l,a]^T(h_BA[i,l] - h_AB[i,l])``. Each row selects its
    declared H/V/D axis; ground-truth signs are used only for metrics.
    """
    states = np.asarray(ordered_object_states, dtype=np.float32)
    basis = np.asarray(axes, dtype=np.float32)
    layers = tuple(int(layer) for layer in layer_ids)
    if states.ndim != 4 or states.shape[:2] != (len(samples), 2):
        raise ValueError("Layerwise multi-view states must have shape [N,2,L,D]")
    if basis.shape != (states.shape[2], 3, states.shape[3]):
        raise ValueError("Layerwise multi-view axes must have shape [L,3,D]")
    if len(layers) != states.shape[2] or layers != tuple(sorted(set(layers))):
        raise ValueError("Layerwise multi-view layer IDs are invalid")
    if not np.isfinite(states).all() or not np.isfinite(basis).all():
        raise FloatingPointError("Layerwise multi-view inputs are non-finite")
    if not np.allclose(np.linalg.norm(basis, axis=2), 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Layerwise multi-view axes must be unit normalized")
    for sample in samples:
        sample.validate()

    coordinates = np.einsum("nold,lad->nola", states, basis)
    deltas = coordinates[:, 1] - coordinates[:, 0]
    axis_indices = np.asarray([sample.axis_index for sample in samples])
    target_scores = np.stack(
        [deltas[row, :, axis] for row, axis in enumerate(axis_indices)]
    )
    if not np.isfinite(target_scores).all():
        raise FloatingPointError("Layerwise multi-view scores are non-finite")
    ground_truth_signs = np.asarray(
        [int(np.sign(sample.gt_delta_hvd[sample.axis_index])) for sample in samples]
    )
    if np.any(ground_truth_signs == 0):
        raise ValueError("Layerwise multi-view ground truth cannot be tied")

    predicted_signs = np.sign(target_scores).astype(np.int8)
    correct = predicted_signs == ground_truth_signs[:, None]
    axes_by_row = np.asarray([sample.axis for sample in samples])
    labels_by_row = np.asarray([sample.endpoint_label for sample in samples])
    endpoint_labels = tuple(sorted(set(labels_by_row)))
    grounding_by_row = np.asarray([sample.grounding_mode for sample in samples])
    grounding_modes = tuple(sorted(set(grounding_by_row) - {"unspecified"}))

    def metric(mask: np.ndarray, layer_position: int) -> dict[str, Any]:
        total = int(mask.sum())
        if total == 0:
            raise ValueError("Layerwise multi-view metric bucket is empty")
        selected_correct = correct[mask, layer_position]
        selected_signs = predicted_signs[mask, layer_position]
        return {
            "correct": int(selected_correct.sum()),
            "total": total,
            "accuracy": float(selected_correct.mean()),
            "ties": int((selected_signs == 0).sum()),
        }

    metrics = []
    all_rows = np.ones(len(samples), dtype=bool)
    for layer_position, layer_id in enumerate(layers):
        layer_metrics = {
            "layer_id": layer_id,
            "overall": metric(all_rows, layer_position),
            "by_axis": {
                axis: metric(axes_by_row == axis, layer_position) for axis in AXIS_ORDER
            },
            "by_label": {
                label: metric(labels_by_row == label, layer_position)
                for label in endpoint_labels
            },
        }
        if grounding_modes:
            layer_metrics["by_grounding_mode"] = {
                mode: metric(grounding_by_row == mode, layer_position)
                for mode in grounding_modes
            }
        metrics.append(layer_metrics)
    return target_scores.astype(np.float32), tuple(metrics)

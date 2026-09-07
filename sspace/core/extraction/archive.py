"""Load checkpointed axis statistics used by later training stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_axis_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load validated unit axes ``[L,3,D]`` and explicit layer IDs ``[L]``."""
    with np.load(path, allow_pickle=False) as archive:
        expected = {"axes", "layer_ids", "prompt_counts", "fingerprint"}
        if set(archive.files) != expected:
            raise ValueError(f"Axis archive fields {set(archive.files)} != {expected}")
        axes = archive["axes"].astype(np.float32, copy=True)
        layer_ids = archive["layer_ids"].astype(np.int64, copy=True)
        prompt_counts = archive["prompt_counts"].copy()
        fingerprint = archive["fingerprint"].copy()
    if (
        axes.ndim != 3
        or axes.shape[:2] != (len(layer_ids), 3)
        or axes.shape[2] <= 0
        or not np.isfinite(axes).all()
    ):
        raise ValueError("Axis archive has incompatible shape or non-finite values")
    if (
        layer_ids.ndim != 1
        or layer_ids.size == 0
        or int(layer_ids[0]) < 0
        or not np.array_equal(layer_ids, np.unique(layer_ids))
    ):
        raise ValueError("Axis layer IDs must be one-dimensional and increasing")
    if (
        prompt_counts.shape != (3,)
        or not np.issubdtype(prompt_counts.dtype, np.integer)
        or np.any(prompt_counts <= 0)
    ):
        raise ValueError("Axis prompt counts must be three positive integers")
    fingerprint_text = str(fingerprint.item()) if fingerprint.shape == () else ""
    if len(fingerprint_text) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint_text
    ):
        raise ValueError("Axis fingerprint must be one lowercase SHA-256 value")
    norms = np.linalg.norm(axes.astype(np.float64), axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Every archived layer-axis vector must have unit L2 norm")
    return axes, layer_ids

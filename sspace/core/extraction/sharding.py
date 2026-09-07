"""Assign deterministic sample shards to extraction workers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SHARDING_PROTOCOL = "global_sample_index_mod_world_size_v1"


def shard_sample_indices(
    selected_indices: Sequence[int],
    rank: int,
    world_size: int,
) -> np.ndarray:
    """Return selected global sample indices assigned to one worker (TRN-02D).

    The only assignment rule is ``global_index % world_size == rank``. Input
    order must be strictly increasing so worker order is also global order.
    """
    values = np.asarray(selected_indices, dtype=np.int64)
    if values.ndim != 1 or not np.array_equal(values, np.unique(values)):
        raise ValueError("Selected sample indices must be unique and increasing")
    if values.size and int(values[0]) < 0:
        raise ValueError("Selected sample indices must be nonnegative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Rank must be in [0, world_size)")
    return values[values % world_size == rank]


def expand_global_job_indices(
    sample_indices: Sequence[int], prompt_style_count: int
) -> np.ndarray:
    """Expand sample indices into sample-major, style-minor global job IDs."""
    if prompt_style_count <= 0:
        raise ValueError("prompt_style_count must be positive")
    samples = np.asarray(sample_indices, dtype=np.int64)
    if samples.ndim != 1 or not np.array_equal(samples, np.unique(samples)):
        raise ValueError("Sample indices must be unique and increasing")
    return np.asarray(
        [
            int(sample) * prompt_style_count + style
            for sample in samples
            for style in range(prompt_style_count)
        ],
        dtype=np.int64,
    )

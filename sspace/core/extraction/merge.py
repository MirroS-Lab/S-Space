"""Merge sample-parallel Final-logit shards in strict global order."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256

from ..data import ConstructionSample
from ..prompts.templates import AXIS_ORDER, PROMPT_STYLE_ORDER
from .archive import load_axis_archive
from .sharding import expand_global_job_indices, shard_sample_indices
from .storage import validate_complete_shard


MERGED_FILES = (
    "axes.npz",
    "all_axis_margins.npy",
    "task_margins.npy",
    "global_job_indices.npy",
    "job_groups.npy",
)
MERGED_CHECKSUM_FILES = (
    *MERGED_FILES,
    "gradient_sums.npy",
    "gradient_means.npy",
    "gradient_counts.npy",
)
MINIMUM_MEAN_GRADIENT_L2 = 1e-12


def _validate_completed_merge(directory: Path, run_fingerprint: str) -> dict[str, Any]:
    """Validate every sealed merged file before reuse or downstream loading."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "run_fingerprint",
        "world_size",
        "sample_count",
        "job_count",
        "record_count",
        "model_layer_count",
        "layer_ids",
        "axes_shape",
        "gradient_counts",
        "minimum_mean_gradient_l2",
        "raw_mean_gradient_l2",
        "canonical_sum_order",
        "worker_sum_crosscheck_rtol",
        "worker_sum_crosscheck_atol",
        "checksums_sha256",
    }
    if set(manifest) != expected_fields or manifest.get("status") != "complete":
        raise ValueError("Merged manifest fields or status differ from the schema")
    if (
        manifest["schema_version"] != "2.1.0"
        or manifest["run_fingerprint"] != run_fingerprint
    ):
        raise ValueError("Merged output has a different schema or run fingerprint")
    checksums_path = directory / "checksums.json"
    if file_sha256(checksums_path) != manifest["checksums_sha256"]:
        raise ValueError("Merged checksum manifest hash mismatch")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if set(checksums) != set(MERGED_CHECKSUM_FILES):
        raise ValueError("Merged checksum file set differs from the schema")
    for name in MERGED_CHECKSUM_FILES:
        if file_sha256(directory / name) != checksums[name]:
            raise ValueError(f"Merged checksum mismatch for {name}")

    axes, layer_ids = load_axis_archive(directory / "axes.npz")
    layers = len(layer_ids)
    jobs = int(manifest["job_count"])
    expected_shapes = {
        "all_axis_margins.npy": (jobs, 2, layers, 3),
        "task_margins.npy": (jobs, 2, layers),
        "global_job_indices.npy": (jobs,),
        "job_groups.npy": (jobs,),
        "gradient_sums.npy": (3, int(manifest["model_layer_count"]), axes.shape[-1]),
        "gradient_means.npy": (3, int(manifest["model_layer_count"]), axes.shape[-1]),
        "gradient_counts.npy": (3,),
    }
    for name, shape in expected_shapes.items():
        value = np.load(directory / name, mmap_mode="r")
        if value.shape != shape:
            raise ValueError(f"Merged {name} shape {value.shape} != {shape}")
        if name not in {
            "global_job_indices.npy",
            "job_groups.npy",
            "gradient_counts.npy",
        }:
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Merged {name} contains non-finite values")
    if manifest["layer_ids"] != layer_ids.tolist() or manifest["axes_shape"] != list(
        axes.shape
    ):
        raise ValueError("Merged manifest axis metadata differs from axes.npz")
    means = np.load(directory / "gradient_means.npy", mmap_mode="r")
    raw_norms = np.linalg.norm(means, axis=-1)
    if (
        manifest["minimum_mean_gradient_l2"] != MINIMUM_MEAN_GRADIENT_L2
        or np.asarray(manifest["raw_mean_gradient_l2"]).shape != raw_norms.shape
        or not np.allclose(
            manifest["raw_mean_gradient_l2"], raw_norms, rtol=1e-15, atol=0.0
        )
        or np.any(raw_norms[:, layer_ids] < MINIMUM_MEAN_GRADIENT_L2)
    ):
        raise ValueError("Merged raw gradient norms or degeneracy gate differ")
    return manifest


def _unit_axes(
    gradient_sums: np.ndarray,
    gradient_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Globally average then L2-normalize H/V/D gradients per layer."""
    sums = np.asarray(gradient_sums, dtype=np.float64)
    counts = np.asarray(gradient_counts, dtype=np.int64)
    if sums.ndim != 3 or sums.shape[0] != 3 or counts.shape != (3,):
        raise ValueError("Gradient sums/counts must have shapes [3,L,D] and [3]")
    if np.any(counts <= 0) or not np.isfinite(sums).all():
        raise ValueError("Every axis requires finite gradients and positive counts")
    means = sums / counts[:, None, None]
    norms = np.linalg.norm(means, axis=-1)
    shared_degenerate = np.all(norms < MINIMUM_MEAN_GRADIENT_L2, axis=0)
    partial_degenerate = (
        np.any(norms < MINIMUM_MEAN_GRADIENT_L2, axis=0) & ~shared_degenerate
    )
    if np.any(partial_degenerate):
        layers = np.flatnonzero(partial_degenerate).tolist()
        raise ValueError(f"Only some H/V/D gradients are degenerate at layers {layers}")
    valid_layers = np.flatnonzero(~shared_degenerate).astype(np.int64)
    if valid_layers.size == 0:
        raise ValueError("Every model layer has degenerate H/V/D gradients")
    axes_group_first = means[:, valid_layers, :] / norms[:, valid_layers, None]
    axes = np.transpose(axes_group_first, (1, 0, 2)).astype(np.float32)
    if not np.isfinite(axes).all():
        raise FloatingPointError("Normalized global axes contain non-finite values")
    return axes, valid_layers, means


def _project_object_states(
    object_states: np.ndarray,
    axes: np.ndarray,
    layer_ids: Sequence[int],
) -> np.ndarray:
    """Project query-minus-reference states on layer-matched unit axes.

    The implemented equation is
    ``m[o,l,g] = v[l,g]^T(h_query[o,l] - h_reference[o,l])``.
    Object role order is fixed as query, reference.

    Args:
        object_states: Float states ``[O,L_model,2,D]``.
        axes: Unit H/V/D axes ``[L_valid,3,D]``.
        layer_ids: Model post-block IDs corresponding to ``axes``.

    Returns:
        Finite float32 margins ``[O,L_valid,3]``.

    Raises:
        ValueError: Shapes, role count, or layer IDs are incompatible.
        FloatingPointError: States, axes, or margins contain non-finite values.
    """
    states = np.asarray(object_states, dtype=np.float32)
    basis = np.asarray(axes, dtype=np.float32)
    layers = np.asarray(layer_ids, dtype=np.int64)
    if states.ndim != 4 or states.shape[2] != 2:
        raise ValueError("Object states must have shape [O,L_model,2,D]")
    if basis.ndim != 3 or basis.shape[1] != len(AXIS_ORDER):
        raise ValueError("Axes must have shape [L_valid,3,D]")
    if layers.shape != (basis.shape[0],) or basis.shape[-1] != states.shape[-1]:
        raise ValueError("Axes, layer IDs, and object states are incompatible")
    if (
        layers.size == 0
        or np.any(layers < 0)
        or np.any(layers >= states.shape[1])
        or len(np.unique(layers)) != len(layers)
    ):
        raise ValueError("Layer IDs must be unique supported post-block indices")
    if not np.isfinite(states).all() or not np.isfinite(basis).all():
        raise FloatingPointError("Object states and axes must be finite")
    differences = states[:, layers, 0, :] - states[:, layers, 1, :]
    margins = np.einsum("old,lgd->olg", differences, basis, optimize=True)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Projected object-state margins are non-finite")
    return margins.astype(np.float32, copy=False)


def _load_worker_arrays(
    directory: Path,
    samples: Sequence[ConstructionSample],
    expected_samples: np.ndarray,
    expected_jobs: np.ndarray,
    layer_count: int,
    hidden_size: int,
) -> dict[str, np.ndarray]:
    stored_samples = np.load(directory / "global_sample_indices.npy", mmap_mode="r")
    if stored_samples.dtype != np.int64 or not np.array_equal(
        stored_samples, expected_samples
    ):
        raise ValueError(
            f"Worker global sample IDs differ from assignment: {directory}"
        )
    jobs = np.load(directory / "global_job_indices.npy", mmap_mode="r")
    if jobs.dtype != np.int64 or not np.array_equal(jobs, expected_jobs):
        raise ValueError(f"Worker global job IDs differ from assignment: {directory}")
    arrays = {
        "jobs": jobs,
        "states": np.load(directory / "object_states.npy", mmap_mode="r"),
        "gradients": np.load(directory / "role_gradients.npy", mmap_mode="r"),
        "contrasts": np.load(directory / "logit_contrasts.npy", mmap_mode="r"),
        "sums": np.load(directory / "gradient_sums.npy"),
        "counts": np.load(directory / "gradient_counts.npy"),
    }
    job_count = len(expected_jobs)
    expected = {
        "states": (job_count, 2, layer_count, 2, hidden_size),
        "gradients": (job_count, 2, layer_count, hidden_size),
        "contrasts": (job_count, 2, 3),
        "sums": (3, layer_count, hidden_size),
        "counts": (3,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Worker {name} shape {arrays[name].shape} != {shape}")
        if name not in {"counts"} and not np.isfinite(arrays[name]).all():
            raise FloatingPointError(f"Worker {name} contains non-finite values")
    records = [
        json.loads(line)
        for line in (directory / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(records) != job_count * 2:
        raise ValueError(f"Worker orientation-record count differs: {directory}")
    for local_job, global_job in enumerate(expected_jobs):
        pair = records[local_job * 2 : local_job * 2 + 2]
        if [record.get("orientation") for record in pair] != ["original", "swapped"]:
            raise ValueError("Worker orientation order differs from the protocol")
        if any(
            int(record.get("global_job_index", -1)) != int(global_job)
            for record in pair
        ):
            raise ValueError("Worker record global job ID differs from its array row")
        expected_style = PROMPT_STYLE_ORDER[int(global_job) % len(PROMPT_STYLE_ORDER)]
        if any(record.get("style") != expected_style for record in pair):
            raise ValueError("Worker record prompt style differs from global job ID")
        sample_index = int(global_job) // len(PROMPT_STYLE_ORDER)
        sample = samples[sample_index]
        if any(
            int(record.get("global_sample_index", -1)) != sample_index
            or record.get("sample_id") != sample.sample_id
            or record.get("group") != sample.group
            for record in pair
        ):
            raise ValueError(
                "Worker record sample identity differs from construction data"
            )
    return arrays


def merge_completed_shards(
    output_dir: Path,
    samples: Sequence[ConstructionSample],
    selected_sample_indices: Sequence[int],
    run_config: dict[str, Any],
    world_size: int,
    model_layer_count: int,
    hidden_size: int,
) -> dict[str, Any]:
    """Merge raw gradients in global job order, then derive axes and margins.

    The canonical sum loops over increasing global job IDs and adds original
    then swapped role gradients. This makes the axis independent of worker
    count and assignment. Worker-local normalized axes are never
    inputs to this function.

    Returns:
        Complete merge manifest with layer IDs and checksums.

    Raises:
        ValueError: Shards, assignments, shapes, sums, or group coverage differ.
        FloatingPointError: Any input or derived array is non-finite.

    Side effects:
        Reads every sealed shard and atomically publishes ``merged/``.
    """
    output_dir = Path(output_dir)
    final_dir = output_dir / "merged"
    temporary_dir = output_dir / "merged.incomplete"
    run_fingerprint = canonical_fingerprint(run_config)
    if final_dir.exists():
        return _validate_completed_merge(final_dir, run_fingerprint)
    if temporary_dir.exists():
        raise FileExistsError(
            f"Interrupted merge directory requires explicit inspection: {temporary_dir}"
        )
    temporary_dir.mkdir(parents=True, exist_ok=False)

    selected = np.asarray(selected_sample_indices, dtype=np.int64)
    if selected.size == 0 or int(selected[-1]) >= len(samples):
        raise ValueError("Selected merge sample indices are empty or out of range")
    global_jobs = expand_global_job_indices(selected, len(PROMPT_STYLE_ORDER))
    workers: dict[int, dict[str, np.ndarray]] = {}
    shard_directories = {
        path.name for path in (output_dir / "shards").glob("rank_*") if path.is_dir()
    }
    expected_directories = {f"rank_{rank:03d}" for rank in range(world_size)}
    if shard_directories != expected_directories:
        raise ValueError(
            "Worker shard rank directories do not exactly match world size"
        )
    for rank in range(world_size):
        directory = output_dir / "shards" / f"rank_{rank:03d}"
        validate_complete_shard(directory, run_fingerprint)
        rank_samples = shard_sample_indices(selected, rank, world_size)
        expected_jobs = expand_global_job_indices(rank_samples, len(PROMPT_STYLE_ORDER))
        workers[rank] = _load_worker_arrays(
            directory,
            samples,
            rank_samples,
            expected_jobs,
            model_layer_count,
            hidden_size,
        )

    lookup: dict[int, tuple[int, int]] = {}
    for rank, arrays in workers.items():
        for local_job, global_job in enumerate(arrays["jobs"]):
            key = int(global_job)
            if key in lookup:
                raise ValueError(f"Duplicate global job {key} across worker shards")
            lookup[key] = (rank, local_job)
    if set(lookup) != set(map(int, global_jobs)):
        raise ValueError("Worker shards do not exactly cover selected global jobs")

    group_to_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    job_groups = np.asarray(
        [
            group_to_index[samples[int(job) // len(PROMPT_STYLE_ORDER)].group]
            for job in global_jobs
        ],
        dtype=np.int8,
    )
    gradient_sums = np.zeros((3, model_layer_count, hidden_size), dtype=np.float64)
    gradient_counts = np.zeros(3, dtype=np.int64)
    for job_position, global_job in enumerate(global_jobs):
        rank, local_job = lookup[int(global_job)]
        gradients = np.asarray(workers[rank]["gradients"][local_job], dtype=np.float64)
        group = int(job_groups[job_position])
        gradient_sums[group] += gradients[0]
        gradient_sums[group] += gradients[1]
        gradient_counts[group] += 2

    shard_sums = sum(
        (np.asarray(worker["sums"], dtype=np.float64) for worker in workers.values()),
        start=np.zeros_like(gradient_sums),
    )
    shard_counts = sum(
        (np.asarray(worker["counts"], dtype=np.int64) for worker in workers.values()),
        start=np.zeros_like(gradient_counts),
    )
    if not np.array_equal(gradient_counts, shard_counts):
        raise ValueError("Canonical and worker aggregate gradient counts differ")
    expected_counts = np.asarray(
        [
            sum(samples[int(index)].group == group for index in selected)
            * len(PROMPT_STYLE_ORDER)
            * 2
            for group in AXIS_ORDER
        ],
        dtype=np.int64,
    )
    if not np.array_equal(gradient_counts, expected_counts):
        raise ValueError("Merged gradient counts differ from selected sample groups")
    if run_config.get("run_kind") == "full":
        if len(selected) != 6000 or len(global_jobs) != 30000:
            raise ValueError(
                "Full merge must cover 6,000 samples and 30,000 paired jobs"
            )
        if gradient_counts.tolist() != [20000, 20000, 20000]:
            raise ValueError(
                "Full merge must contain 20,000 orientation gradients per axis"
            )
    if not np.allclose(gradient_sums, shard_sums, rtol=1e-10, atol=1e-12):
        difference = float(np.max(np.abs(gradient_sums - shard_sums)))
        raise ValueError(
            f"Canonical and worker aggregate sums differ; max_abs={difference}"
        )

    axes, layer_ids, gradient_means = _unit_axes(gradient_sums, gradient_counts)
    job_count, valid_layer_count = len(global_jobs), len(layer_ids)
    all_margins = open_memmap(
        temporary_dir / "all_axis_margins.npy",
        mode="w+",
        dtype=np.float32,
        shape=(job_count, 2, valid_layer_count, 3),
    )
    task_margins = open_memmap(
        temporary_dir / "task_margins.npy",
        mode="w+",
        dtype=np.float32,
        shape=(job_count, 2, valid_layer_count),
    )
    for job_position, global_job in enumerate(global_jobs):
        rank, local_job = lookup[int(global_job)]
        states = np.asarray(workers[rank]["states"][local_job], dtype=np.float32)
        margins = _project_object_states(states, axes, layer_ids)
        all_margins[job_position] = margins
        task_margins[job_position] = margins[:, :, int(job_groups[job_position])]
    all_margins.flush()
    task_margins.flush()
    if not np.isfinite(all_margins).all() or not np.isfinite(task_margins).all():
        raise FloatingPointError("Merged all-axis or task margins are non-finite")

    np.save(temporary_dir / "global_job_indices.npy", global_jobs)
    np.save(temporary_dir / "job_groups.npy", job_groups)
    np.savez(
        temporary_dir / "axes.npz",
        axes=axes,
        layer_ids=layer_ids,
        prompt_counts=gradient_counts,
        fingerprint=np.asarray(run_fingerprint),
    )
    np.save(temporary_dir / "gradient_sums.npy", gradient_sums)
    np.save(temporary_dir / "gradient_means.npy", gradient_means)
    np.save(temporary_dir / "gradient_counts.npy", gradient_counts)
    checksums = {
        name: file_sha256(temporary_dir / name) for name in MERGED_CHECKSUM_FILES
    }
    atomic_json(temporary_dir / "checksums.json", checksums)
    manifest = {
        "schema_version": "2.1.0",
        "status": "complete",
        "run_fingerprint": run_fingerprint,
        "world_size": world_size,
        "sample_count": int(selected.size),
        "job_count": job_count,
        "record_count": job_count * 2,
        "model_layer_count": model_layer_count,
        "layer_ids": layer_ids.tolist(),
        "axes_shape": list(axes.shape),
        "gradient_counts": gradient_counts.tolist(),
        "minimum_mean_gradient_l2": MINIMUM_MEAN_GRADIENT_L2,
        "raw_mean_gradient_l2": np.linalg.norm(gradient_means, axis=-1).tolist(),
        "canonical_sum_order": "global_job_then_original_swapped_v1",
        "worker_sum_crosscheck_rtol": 1e-10,
        "worker_sum_crosscheck_atol": 1e-12,
        "checksums_sha256": file_sha256(temporary_dir / "checksums.json"),
    }
    atomic_json(temporary_dir / "manifest.json", manifest)
    os.replace(temporary_dir, final_dir)
    return _validate_completed_merge(final_dir, run_fingerprint)

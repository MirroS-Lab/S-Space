"""Compute checkpointed COCO-1800 readout for every model adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image

from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256

from sspace.core.data import ConstructionSample
from sspace.core.extraction.execution import (
    LayerwisePostBlockStates,
    move_model_inputs,
)
from sspace.core.models.runtime import LoadedModelRuntime
from sspace.core.prompts.rendering import render_prompt_pair
from sspace.core.prompts.templates import AXIS_ORDER, PROMPT_STYLE_ORDER


def _create_margin_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    """Create one new float32 NaN-filled validation margin array."""
    if path.exists():
        raise FileExistsError(path)
    margins = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    margins[:] = np.nan
    margins.flush()
    return margins


def _margin_prefix_sha256(margins: np.ndarray, next_job: int) -> str:
    prefix = np.ascontiguousarray(margins[:next_job], dtype=np.float32)
    return hashlib.sha256(prefix.tobytes(order="C")).hexdigest()


def _file_prefix_sha256(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Validation record prefix is shorter than checkpoint")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_records(
    records: Sequence[dict[str, Any]],
    margins: np.ndarray,
    samples: Sequence[ConstructionSample],
    jobs: Sequence[tuple[int, str]],
) -> None:
    if len(records) != len(jobs) or margins.shape[0] < len(jobs):
        raise ValueError("Validation record count differs")
    for job_index, ((sample_index, style), record) in enumerate(
        zip(jobs, records, strict=True)
    ):
        sample = samples[sample_index]
        if (
            record.get("job_index") != job_index
            or record.get("global_sample_index") != sample_index
            or record.get("sample_id") != sample.sample_id
            or record.get("style") != style
            or record.get("group") != sample.group
            or record.get("endpoint_label") != sample.label
        ):
            raise ValueError(f"Validation record identity differs at job {job_index}")
        record_margins = np.asarray(record.get("margins"), dtype=np.float32)
        if record_margins.shape != margins[job_index].shape or not np.array_equal(
            record_margins, margins[job_index]
        ):
            raise ValueError(f"Validation record margins differ at job {job_index}")


def _validate_completed_validation(
    output_dir: Path,
    fingerprint: str,
    samples: Sequence[ConstructionSample],
    jobs: Sequence[tuple[int, str]],
    layers: tuple[int, ...],
    shape: tuple[int, ...],
) -> dict[str, Any]:
    """Validate a sealed validation readout before reuse or publication."""
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "fingerprint",
        "sample_count",
        "job_count",
        "layer_ids",
        "shape",
        "checksums_sha256",
    }
    if set(manifest) != expected_fields or manifest.get("status") != "complete":
        raise ValueError("Validation manifest fields or status differ from the schema")
    if (
        manifest["schema_version"] != "1.0.0"
        or manifest["fingerprint"] != fingerprint
        or manifest["job_count"] != len(jobs)
        or manifest["sample_count"] != len(jobs) // len(PROMPT_STYLE_ORDER)
        or manifest["layer_ids"] != list(layers)
        or manifest["shape"] != list(shape)
    ):
        raise ValueError("Validation manifest identity or layout differs")
    checksums_path = output_dir / "checksums.json"
    if file_sha256(checksums_path) != manifest["checksums_sha256"]:
        raise ValueError("Validation checksum manifest hash mismatch")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if set(checksums) != {"margins.npy", "records.jsonl"}:
        raise ValueError("Validation checksum file set differs from the schema")
    for name, checksum in checksums.items():
        if file_sha256(output_dir / name) != checksum:
            raise ValueError(f"Validation checksum mismatch for {name}")
    margins = np.load(output_dir / "margins.npy", mmap_mode="r")
    if margins.shape != shape or margins.dtype != np.float32:
        raise ValueError("Completed validation margin layout differs")
    if not np.isfinite(margins).all():
        raise FloatingPointError("Completed validation margins are non-finite")
    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    _validate_records(records, margins, samples, jobs)
    return manifest


def extract_validation_margins(
    runtime: LoadedModelRuntime,
    samples: Sequence[ConstructionSample],
    selected_sample_indices: Sequence[int],
    axes: np.ndarray,
    layer_ids: Sequence[int],
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
) -> dict[str, Any]:
    """Extract original/swapped task margins for disjoint validation (TRN-04D).

    Returns:
        Complete manifest for ``margins.npy[J,2,L_valid]`` and records.

    Side effects:
        Runs forward-only readout on one rank and writes resumable outputs.
    """
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    selected = tuple(int(value) for value in selected_sample_indices)
    if not selected or tuple(sorted(set(selected))) != selected or selected[0] < 0:
        raise ValueError("Validation sample indices must be non-empty and increasing")
    if selected[-1] >= len(samples):
        raise ValueError("Validation sample index exceeds the split")
    axis_values = np.asarray(axes, dtype=np.float32)
    layers = tuple(int(value) for value in layer_ids)
    if axis_values.shape != (len(layers), 3, runtime.spec.hidden_size):
        raise ValueError("Validation axes have an incompatible shape")
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or layers[0] < 0
        or layers[-1] >= runtime.spec.layer_count
    ):
        raise ValueError("Validation layer IDs are not supported and increasing")
    if not np.isfinite(axis_values).all():
        raise FloatingPointError("Validation axes contain non-finite values")
    norms = np.linalg.norm(axis_values.astype(np.float64), axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Validation axes must have unit L2 norm")

    jobs = [
        (sample_index, style)
        for sample_index in selected
        for style in PROMPT_STYLE_ORDER
    ]
    shape = (len(jobs), 2, len(layers))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    margins_path = output_dir / "margins.npy"
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    fingerprint = canonical_fingerprint(run_config)
    if manifest_path.exists():
        return _validate_completed_validation(
            output_dir, fingerprint, samples, jobs, layers, shape
        )
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_checkpoint_fields = {
            "schema_version",
            "fingerprint",
            "next_job",
            "records_offset",
            "shape",
            "margins_prefix_sha256",
            "records_prefix_sha256",
        }
        if set(checkpoint) != expected_checkpoint_fields:
            raise ValueError("Validation checkpoint fields differ from schema 2")
        if checkpoint["schema_version"] != "2.0.0":
            raise ValueError("Validation checkpoint schema differs")
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Validation checkpoint fingerprint mismatch")
        if checkpoint["shape"] != list(shape):
            raise ValueError("Validation checkpoint shape differs")
        margins = np.load(margins_path, mmap_mode="r+")
        if margins.shape != shape or margins.dtype != np.float32:
            raise ValueError("Validation checkpoint margin layout mismatch")
        if isinstance(checkpoint["next_job"], bool) or not isinstance(
            checkpoint["next_job"], int
        ):
            raise ValueError("Validation checkpoint next_job must be an integer")
        next_job = checkpoint["next_job"]
        if not 0 <= next_job <= len(jobs):
            raise ValueError("Validation checkpoint next_job is outside the layout")
        if isinstance(checkpoint["records_offset"], bool) or not isinstance(
            checkpoint["records_offset"], int
        ):
            raise ValueError("Validation checkpoint record offset must be an integer")
        records_offset = checkpoint["records_offset"]
        if not 0 <= records_offset <= records_path.stat().st_size:
            raise ValueError("Validation checkpoint record offset is invalid")
        if not np.isfinite(margins[:next_job]).all():
            raise FloatingPointError("Validation checkpoint margins are non-finite")
        if (
            _margin_prefix_sha256(margins, next_job)
            != checkpoint["margins_prefix_sha256"]
            or _file_prefix_sha256(records_path, records_offset)
            != checkpoint["records_prefix_sha256"]
        ):
            raise ValueError("Validation checkpoint committed prefix checksum differs")
        prefix = records_path.read_bytes()[:records_offset].decode("utf-8")
        committed_records = [json.loads(line) for line in prefix.splitlines()]
        _validate_records(
            committed_records,
            margins,
            samples,
            jobs[:next_job],
        )
        with records_path.open("r+b") as stream:
            stream.truncate(records_offset)
        margins[next_job:] = np.nan
        margins.flush()
    else:
        if margins_path.exists() or records_path.exists():
            raise FileExistsError("Validation files exist without a checkpoint")
        margins = _create_margin_memmap(margins_path, shape)
        records_path.touch(exist_ok=False)
        next_job = 0

    def checkpoint(stream: Any) -> None:
        margins.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "schema_version": "2.0.0",
                "fingerprint": fingerprint,
                "next_job": next_job,
                "records_offset": stream.tell(),
                "shape": list(shape),
                "margins_prefix_sha256": _margin_prefix_sha256(margins, next_job),
                "records_prefix_sha256": _file_prefix_sha256(
                    records_path, stream.tell()
                ),
            },
        )

    group_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    axis_tensor = torch.from_numpy(axis_values)
    with records_path.open("a", encoding="utf-8") as records:
        checkpoint(records)
        with LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor:
            while next_job < len(jobs):
                sample_index, style = jobs[next_job]
                sample = samples[sample_index]
                prompts = render_prompt_pair(
                    style,
                    sample.group,
                    sample.query,
                    sample.reference,
                )
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
                prepared_values = []
                difference_values = []
                batch_size = runtime.spec.orientation_batch_size
                for start in range(0, len(prompts), batch_size):
                    prepared = runtime.prepare(
                        runtime.processor, image, prompts[start : start + batch_size]
                    )
                    prepared_values.append(prepared)
                    difference_values.append(
                        extractor.extract(
                            move_model_inputs(prepared.inputs, runtime.model),
                            prepared.positions,
                        )
                    )
                differences = torch.cat(difference_values, dim=0)
                positions = tuple(
                    position
                    for prepared in prepared_values
                    for position in prepared.positions
                )
                token_pieces = tuple(
                    piece
                    for prepared in prepared_values
                    for piece in prepared.token_pieces
                )
                selected_differences = differences[:, list(layers), :]
                group_axes = axis_tensor[:, group_index[sample.group], :]
                task_margins = torch.einsum(
                    "bld,ld->bl", selected_differences, group_axes
                )
                if (
                    task_margins.shape != (2, len(layers))
                    or not torch.isfinite(task_margins).all()
                ):
                    raise FloatingPointError("Validation task margins are invalid")
                margins[next_job] = task_margins.numpy()
                records.write(
                    json.dumps(
                        {
                            "job_index": next_job,
                            "global_sample_index": sample_index,
                            "sample_id": sample.sample_id,
                            "style": style,
                            "group": sample.group,
                            "endpoint_label": sample.label,
                            "prompts": [prompt.text for prompt in prompts],
                            "positions": positions,
                            "token_pieces": token_pieces,
                            "margins": task_margins.tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_job += 1
                if next_job % checkpoint_every == 0:
                    checkpoint(records)
                    print(f"validation jobs {next_job}/{len(jobs)}", flush=True)
        checkpoint(records)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Completed validation margins are non-finite")
    checksums = {
        name: file_sha256(output_dir / name)
        for name in ("margins.npy", "records.jsonl")
    }
    atomic_json(output_dir / "checksums.json", checksums)
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "fingerprint": fingerprint,
        "sample_count": len(selected),
        "job_count": len(jobs),
        "layer_ids": list(layers),
        "shape": list(shape),
        "checksums_sha256": file_sha256(output_dir / "checksums.json"),
    }
    atomic_json(manifest_path, manifest)
    return _validate_completed_validation(
        output_dir, fingerprint, samples, jobs, layers, shape
    )

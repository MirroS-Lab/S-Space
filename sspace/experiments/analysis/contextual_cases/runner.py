"""Run resumable selected-layer projections for contextual case manifests."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sspace.core.artifacts import SSpaceArtifact
from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256
from sspace.core.extraction.execution import (
    SelectedLayerObjectStates,
    move_model_inputs,
)
from sspace.core.models.runtime import LoadedModelRuntime
from sspace.core.prompts.rendering import (
    RenderedMultiImagePrompt,
    RenderedObjectPrompt,
)

from .config import CaseEvaluationConfig
from .schema import CaseSample


def _open_images(sample: CaseSample) -> list[Image.Image]:
    images = []
    for item in sample.images:
        item.validate()
        image = Image.open(item.path)
        image.load()
        images.append(image.convert("RGB"))
    return images


def _prepare(
    runtime: LoadedModelRuntime, images: Sequence[Image.Image], sample: CaseSample
):
    if len(images) == 1:
        prompt = RenderedObjectPrompt(sample.prompt, sample.object_spans)
        return runtime.prepare_objects(runtime.processor, images[0], (prompt,))
    segments = tuple("" for _ in range(len(images) - 1)) + (sample.prompt,)
    prompt = RenderedMultiImagePrompt(segments, len(segments) - 1, sample.object_spans)
    return runtime.prepare_multi_image_objects(runtime.processor, images, prompt)


def _atomic_state(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _projection_record(
    artifact: SSpaceArtifact,
    sample: CaseSample,
    states: np.ndarray,
    positions: dict[str, int],
    token_pieces: dict[str, str],
    selected_layer: int,
    state_file: str,
    state_sha256: str,
) -> dict[str, Any]:
    role_order = sample.object_order
    coordinates = artifact.project_objects(states, selected_layer)
    role_projections = {
        role: {
            axis: float(coordinates[index, axis_index])
            for axis_index, axis in enumerate(artifact.manifest.axis_order)
        }
        for index, role in enumerate(role_order)
    }
    pairwise = None
    if role_order == ("target", "reference"):
        margin = artifact.project_pairwise(
            states[0][None, :], states[1][None, :], selected_layer
        )[0]
        pairwise = {
            axis: float(margin[index])
            for index, axis in enumerate(artifact.manifest.axis_order)
        }
    return {
        "case_id": sample.case_id,
        "theme": sample.theme,
        "family": sample.family,
        "condition": sample.condition,
        "prompt": sample.prompt,
        "selected_layer": selected_layer,
        "image_paths": [str(image.path) for image in sample.images],
        "image_sha256": [image.sha256 for image in sample.images],
        "object_spans": {
            key: list(value) for key, value in sample.object_spans.items()
        },
        "object_order": list(role_order),
        "positions": positions,
        "token_pieces": token_pieces,
        "role_projections": role_projections,
        "target_minus_reference": pairwise,
        "metadata": dict(sample.metadata),
        "state_file": state_file,
        "state_sha256": state_sha256,
    }


def run_case_evaluation(
    runtime: LoadedModelRuntime,
    artifact: SSpaceArtifact,
    samples: Sequence[CaseSample],
    config: CaseEvaluationConfig,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Project exact named token states and checkpoint each completed case.

    The saved state for case ``i`` has shape ``[O,D]``, where ``O`` is one
    target or target/reference and ``D`` is the model hidden size. Projections
    are ``m_g = v_g^T h``; when a reference exists, the matched readout is
    ``v_g^T(h_target-h_reference)``. H/V/D signs are right/below/close.
    """
    if runtime.spec.key != config.model_adapter:
        raise ValueError("Case runtime differs from the config")
    axes = artifact.axes(config.selected_layer)
    if axes.shape != (3, runtime.spec.hidden_size):
        raise ValueError("Case artifact layer has an incompatible shape")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    fingerprint = canonical_fingerprint(run_config)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Case checkpoint fingerprint mismatch")
        next_index = int(checkpoint["next_sample_index"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
        for orphan in states_dir.glob("*.npy"):
            if int(orphan.stem) >= next_index:
                orphan.unlink()
    else:
        if records_path.exists() or any(states_dir.iterdir()):
            raise FileExistsError("Case outputs exist without a checkpoint")
        records_path.touch(exist_ok=False)
        next_index = 0
    committed_offset = records_path.stat().st_size

    def checkpoint(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample_index": next_index,
                "records_offset": committed_offset,
                "sample_count": len(samples),
            },
        )

    try:
        with records_path.open("a", encoding="utf-8") as stream:
            with SelectedLayerObjectStates(
                runtime.model, runtime.blocks[config.selected_layer]
            ) as extractor:
                while next_index < len(samples):
                    sample = samples[next_index]
                    images = _open_images(sample)
                    try:
                        prepared = _prepare(runtime, images, sample)
                        state = extractor.extract(
                            move_model_inputs(prepared.inputs, runtime.model),
                            prepared.positions,
                            sample.object_order,
                        )[0].numpy()
                    finally:
                        for image in images:
                            image.close()
                    expected = (len(sample.object_order), runtime.spec.hidden_size)
                    if state.shape != expected or not np.isfinite(state).all():
                        raise ValueError(f"Case state {state.shape} != {expected}")
                    state_path = states_dir / f"{next_index:06d}.npy"
                    _atomic_state(state_path, state.astype(np.float32, copy=False))
                    record = _projection_record(
                        artifact,
                        sample,
                        state,
                        dict(prepared.positions[0]),
                        dict(prepared.token_pieces[0]),
                        config.selected_layer,
                        str(state_path.relative_to(output_dir)),
                        file_sha256(state_path),
                    )
                    stream.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    next_index += 1
                    committed_offset = stream.tell()
                    if next_index % config.checkpoint_every == 0:
                        checkpoint(stream)
                        print(
                            f"contextual cases {next_index}/{len(samples)}", flush=True
                        )
            checkpoint(stream)
    except BaseException:
        with records_path.open("r+b") as stream:
            stream.truncate(committed_offset)
        with records_path.open("a", encoding="utf-8") as stream:
            checkpoint(stream)
        raise
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    by_theme = {
        theme: sum(record["theme"] == theme for record in records)
        for theme in sorted({record["theme"] for record in records})
    }
    return {
        "sample_count": len(records),
        "selected_layer": config.selected_layer,
        "artifact_id": artifact.manifest.artifact_id,
        "by_theme": by_theme,
        "families": sorted({record["family"] for record in records}),
    }

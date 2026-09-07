"""Run checkpointed pairwise model projections."""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from numpy.lib.format import open_memmap

from sspace.core.artifacts import SSpaceArtifact
from sspace.run_records import atomic_json, canonical_fingerprint
from sspace.core.extraction.execution import (
    LayerwisePostBlockStates,
    move_model_inputs,
)
from sspace.core.models.runtime import LoadedModelRuntime
from sspace.core.prompts.rendering import RenderedPrompt, render_prompt_pair
from sspace.core.prompts.templates import AXIS_ORDER

from .schema import PairwiseEvaluationSample, open_pairwise_image


def _new_margin_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    """Create one new finite-checkable float32 margin array."""
    if path.exists():
        raise FileExistsError(path)
    margins = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    margins[:] = np.nan
    margins.flush()
    return margins


def _initialize_margin_output(
    margin_path: Path,
    records_path: Path,
    checkpoint_path: Path,
    shape: tuple[int, ...],
    fingerprint: str,
) -> np.memmap:
    """Create margin files with a resumable zero-progress checkpoint."""
    if margin_path.exists() or records_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Evaluation outputs exist without a checkpoint")
    margins = _new_margin_memmap(margin_path, shape)
    records_path.touch(exist_ok=False)
    atomic_json(
        checkpoint_path,
        {
            "fingerprint": fingerprint,
            "next_sample": 0,
            "records_offset": 0,
            "margin_shape": list(shape),
        },
    )
    return margins


def _model_layer_difference(differences: torch.Tensor, layer: int) -> torch.Tensor:
    """Select a post-block state by its actual zero-based model layer ID."""
    if differences.ndim != 3:
        raise ValueError("Layerwise differences must have shape [B,L,D]")
    if not 0 <= int(layer) < differences.shape[1]:
        raise ValueError(f"Model does not expose post-block layer {layer}")
    return differences[:, int(layer), :]


def _orientation_batches(
    prompts: Sequence[RenderedPrompt], orientation_batch_size: int
) -> tuple[tuple[RenderedPrompt, ...], ...]:
    """Partition one original/swapped prompt pair without changing its order.

    Args:
        prompts: Exactly two prompts in original, swapped order.
        orientation_batch_size: Explicit model capability, either one or two.

    Returns:
        One batch of two prompts or two batches of one prompt.

    Raises:
        ValueError: The prompt count or declared batch size is invalid.

    Side effects:
        None.
    """
    values = tuple(prompts)
    if len(values) != 2:
        raise ValueError("Pairwise evaluation requires original and swapped prompts")
    if orientation_batch_size == 2:
        return (values,)
    if orientation_batch_size == 1:
        return ((values[0],), (values[1],))
    raise ValueError("orientation_batch_size must be 1 or 2")


def _extract_prompt_pair_differences(
    runtime: LoadedModelRuntime,
    extractor: LayerwisePostBlockStates,
    image: Image.Image,
    prompts: Sequence[RenderedPrompt],
) -> tuple[
    torch.Tensor,
    tuple[dict[str, int], ...],
    tuple[dict[str, str], ...],
]:
    """Extract both orientations with the model's declared batch capability."""
    differences = []
    positions: list[dict[str, int]] = []
    token_pieces: list[dict[str, str]] = []
    for prompt_batch in _orientation_batches(
        prompts, runtime.spec.orientation_batch_size
    ):
        prepared = runtime.prepare(runtime.processor, image, prompt_batch)
        value = extractor.extract(
            move_model_inputs(prepared.inputs, runtime.model), prepared.positions
        )
        if value.ndim != 3 or value.shape[0] != len(prompt_batch):
            raise ValueError("Layerwise prompt-batch differences are invalid")
        differences.append(value)
        positions.extend(prepared.positions)
        token_pieces.extend(prepared.token_pieces)
    combined = torch.cat(differences, dim=0)
    if combined.shape[0] != 2:
        raise ValueError("Pairwise prompt differences must preserve two orientations")
    return combined, tuple(positions), tuple(token_pieces)


def _extract_prompt_pair_margins(
    runtime: LoadedModelRuntime,
    artifact: SSpaceArtifact,
    samples: Sequence[Any],
    prompt_pairs: Sequence[Sequence[RenderedPrompt]],
    layer: int,
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
    image_loader: Callable[[Any], Image.Image],
) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    """Extract selected-axis margins for explicit original/swapped prompts.

    For sample ``i`` and its declared axis ``g``, the stored values are
    ``v[layer,g]^T(h_query-h_reference)`` for the original and object-swapped
    prompts supplied by the benchmark-specific prompt constructor. Records
    preserve both prompt texts, exact token positions, token pieces, and raw
    margins. This shared model-execution function does not construct prompts or
    inspect benchmark labels (EVAL-01/EVAL-09).

    Args:
        runtime: Loaded model adapter and object-token alignment runtime.
        artifact: Validated S-Space artifact containing unit H/V/D axes.
        samples: Ordered objects exposing ``sample_id`` and ``group``.
        prompt_pairs: One explicit original/swapped ``RenderedPrompt`` pair per
            sample, in the same order.
        layer: Actual zero-based post-block layer ID.
        output_dir: New or resumable raw-margin output directory.
        run_config: Fully resolved configuration bound into the resume hash.
        checkpoint_every: Complete samples between durable checkpoints.
        image_loader: Benchmark-specific validated image decoder.

    Returns:
        Margins ``[N,2]`` and ordered original/swapped prompt-text pairs.

    Raises:
        ValueError: Artifact, sample, checkpoint, token, shape, or finite checks fail.

    Side effects:
        Runs model forwards and writes resumable margin and JSONL records.
    """
    artifact.validate()
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if runtime.spec.orientation_batch_size not in {1, 2}:
        raise ValueError("orientation_batch_size must be 1 or 2")
    if len(prompt_pairs) != len(samples):
        raise ValueError("Samples and prompt pairs must have the same length")
    axes = torch.from_numpy(artifact.axes(layer))
    if axes.shape[-1] != runtime.spec.hidden_size:
        raise ValueError("Artifact axes and runtime hidden size differ")
    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    for sample, prompts in zip(samples, prompt_pairs, strict=True):
        if not str(getattr(sample, "sample_id", "")).strip():
            raise ValueError("Pairwise margin samples need a non-empty sample ID")
        if getattr(sample, "group", None) not in axis_index:
            raise ValueError("Pairwise margin sample has an invalid axis group")
        if len(tuple(prompts)) != 2:
            raise ValueError("Every sample needs original and swapped prompts")

    output_dir.mkdir(parents=True, exist_ok=True)
    margin_path = output_dir / "margins.npy"
    records_path = output_dir / "margin_records.jsonl"
    checkpoint_path = output_dir / "margin_checkpoint.json"
    fingerprint = canonical_fingerprint(run_config)
    shape = (len(samples), 2)
    prompts_by_index: list[tuple[str, str] | None] = [None] * len(samples)

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Evaluation margin checkpoint fingerprint mismatch")
        margins = np.load(margin_path, mmap_mode="r+")
        if margins.shape != shape or margins.dtype != np.float32:
            raise ValueError("Evaluation margin checkpoint shape or dtype mismatch")
        next_sample = int(checkpoint["next_sample"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            prompts_by_index[int(record["sample_index"])] = tuple(record["prompts"])
    else:
        margins = _initialize_margin_output(
            margin_path, records_path, checkpoint_path, shape, fingerprint
        )
        next_sample = 0

    def checkpoint(stream: Any) -> None:
        margins.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample": next_sample,
                "records_offset": stream.tell(),
                "margin_shape": list(shape),
            },
        )

    with records_path.open("a", encoding="utf-8") as records:
        with LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                prompts = tuple(prompt_pairs[next_sample])
                differences, positions, token_pieces = _extract_prompt_pair_differences(
                    runtime,
                    extractor,
                    image_loader(sample),
                    prompts,
                )
                selected = _model_layer_difference(differences, layer)
                axis = axes[axis_index[sample.group]]
                values = torch.einsum("bd,d->b", selected, axis)
                if values.shape != (2,) or not torch.isfinite(values).all():
                    raise ValueError("Benchmark margin must be two finite values")
                margins[next_sample] = values.numpy()
                prompt_text = (prompts[0].text, prompts[1].text)
                prompts_by_index[next_sample] = prompt_text
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "sample_id": sample.sample_id,
                            "group": sample.group,
                            "prompts": prompt_text,
                            "positions": positions,
                            "token_pieces": token_pieces,
                            "margins": values.tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                if next_sample % checkpoint_every == 0:
                    checkpoint(records)
                    print(
                        f"evaluation margins {next_sample}/{len(samples)}", flush=True
                    )
        checkpoint(records)
    if not np.isfinite(margins).all() or any(
        value is None for value in prompts_by_index
    ):
        raise ValueError("Completed evaluation margin output is incomplete")
    return np.asarray(margins), tuple(prompts_by_index)  # type: ignore[arg-type]


def extract_benchmark_margins(
    runtime: LoadedModelRuntime,
    artifact: SSpaceArtifact,
    samples: Sequence[PairwiseEvaluationSample],
    layer: int,
    prompt_style: str,
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    """Extract original/swapped margins for formal pairwise benchmark rows.

    The benchmark direct prompt is never used to compute a projection score.
    This wrapper validates typed pairwise rows, renders the frozen prompt style,
    and delegates model execution to the shared explicit-prompt extractor.
    """
    for sample in samples:
        sample.validate()
    prompt_pairs = tuple(
        render_prompt_pair(
            prompt_style,
            sample.group,
            sample.query,
            sample.reference,
        )
        for sample in samples
    )
    return _extract_prompt_pair_margins(
        runtime=runtime,
        artifact=artifact,
        samples=samples,
        prompt_pairs=prompt_pairs,
        layer=layer,
        output_dir=output_dir,
        run_config=run_config,
        checkpoint_every=checkpoint_every,
        image_loader=open_pairwise_image,
    )


def extract_layerwise_benchmark_margins(
    runtime: LoadedModelRuntime,
    axes: np.ndarray,
    layer_ids: Sequence[int],
    samples: Sequence[PairwiseEvaluationSample],
    prompt_style: str,
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
) -> np.ndarray:
    """Extract a layer-matched pairwise margin for every benchmark row.

    For benchmark row ``i``, orientation ``o``, and exported layer ``l``, this
    diagnostic computes

    ``m[i,o,l] = v[l,g]^T(h_query[i,o,l] - h_reference[i,o,l])``.

    The axis and hidden state always come from the same post-block layer.
    Original and swapped prompts are evaluated together, although margin-sign
    scoring reads only the original margin. This is the model-execution part
    of DIA-02; it never reads ground-truth endpoint labels.

    Args:
        runtime: Loaded formal model runtime containing the pinned processor,
            text blocks, prompt preparation, and object-token alignment.
        axes: Finite unit axes ``[L,3,D]`` in H/V/D order.
        layer_ids: Increasing post-block IDs corresponding to ``axes``.
        samples: Ordered pairwise benchmark rows.
        prompt_style: One frozen projection prompt style.
        output_dir: New or resumable margin-record directory.
        run_config: Fully resolved configuration included in the resume hash.
        checkpoint_every: Complete benchmark rows between checkpoints.
    Returns:
        Finite float32 margins ``[N,2,L]`` in original/swapped order.

    Raises:
        ValueError: Axis, layer, sample, checkpoint, token, or shape checks fail.
        FloatingPointError: A state, margin, or completed output is non-finite.

    Side effects:
        Runs model forwards and writes resumable margins and per-row records.

    Example:
        ``margins = extract_layerwise_benchmark_margins(...); margins[:, 0]``
    """
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if runtime.spec.orientation_batch_size not in {1, 2}:
        raise ValueError("orientation_batch_size must be 1 or 2")
    axis_values = np.asarray(axes, dtype=np.float32)
    layer_values = tuple(int(value) for value in layer_ids)
    if axis_values.ndim != 3 or axis_values.shape[1] != len(AXIS_ORDER):
        raise ValueError("Axes must have shape [L,3,D] in H/V/D order")
    if axis_values.shape[0] != len(layer_values) or not np.isfinite(axis_values).all():
        raise ValueError("Axes and layer IDs are incompatible or non-finite")
    if axis_values.shape[-1] != runtime.spec.hidden_size:
        raise ValueError("Axes and runtime hidden size differ")
    model_layer_count = len(runtime.blocks)
    if (
        not layer_values
        or layer_values != tuple(sorted(set(layer_values)))
        or layer_values[0] < 0
        or layer_values[-1] >= model_layer_count
    ):
        raise ValueError("Layer IDs must be increasing supported post-block layers")
    for sample in samples:
        sample.validate()

    output_dir.mkdir(parents=True, exist_ok=True)
    margin_path = output_dir / "margins.npy"
    records_path = output_dir / "margin_records.jsonl"
    checkpoint_path = output_dir / "margin_checkpoint.json"
    fingerprint = canonical_fingerprint(run_config)
    shape = (len(samples), 2, len(layer_values))

    if checkpoint_path.exists():
        checkpoint_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_state["fingerprint"] != fingerprint:
            raise ValueError("Layerwise margin checkpoint fingerprint mismatch")
        margins = np.load(margin_path, mmap_mode="r+")
        if margins.shape != shape or margins.dtype != np.float32:
            raise ValueError("Layerwise margin checkpoint shape or dtype mismatch")
        next_sample = int(checkpoint_state["next_sample"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint_state["records_offset"]))
    else:
        margins = _initialize_margin_output(
            margin_path, records_path, checkpoint_path, shape, fingerprint
        )
        next_sample = 0

    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    axis_tensor = torch.from_numpy(axis_values)

    def checkpoint(stream: Any) -> None:
        margins.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample": next_sample,
                "records_offset": stream.tell(),
                "margin_shape": list(shape),
            },
        )

    with records_path.open("a", encoding="utf-8") as records:
        with LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                prompts = render_prompt_pair(
                    prompt_style, sample.group, sample.query, sample.reference
                )
                all_differences, positions, token_pieces = (
                    _extract_prompt_pair_differences(
                        runtime,
                        extractor,
                        open_pairwise_image(sample),
                        prompts,
                    )
                )
                differences = all_differences[:, list(layer_values), :]
                group_axes = axis_tensor[:, axis_index[sample.group], :]
                values = torch.einsum("bld,ld->bl", differences, group_axes)
                if (
                    values.shape != (2, len(layer_values))
                    or not torch.isfinite(values).all()
                ):
                    raise ValueError("Layerwise benchmark margins are invalid")
                margins[next_sample] = values.numpy()
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "sample_id": sample.sample_id,
                            "group": sample.group,
                            "query": sample.query,
                            "reference": sample.reference,
                            "gold_endpoint": sample.gold_endpoint,
                            "direct_prompt": sample.direct_prompt,
                            "metadata": sample.metadata,
                            "projection_prompts": [prompt.text for prompt in prompts],
                            "positions": positions,
                            "token_pieces": token_pieces,
                            "margins": values.tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                if next_sample % checkpoint_every == 0:
                    checkpoint(records)
                    print(
                        f"layerwise evaluation margins {next_sample}/{len(samples)}",
                        flush=True,
                    )
        checkpoint(records)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Completed layerwise margins are non-finite")
    return np.asarray(margins)

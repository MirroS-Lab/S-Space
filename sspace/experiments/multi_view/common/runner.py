"""Run the checkpointed H* multi-view evaluation."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sspace.experiments.multi_view.hstar.hstar_config import (
    HStarMultiViewConfig,
)
from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256
from sspace.core.extraction.execution import (
    LayerwiseNamedObjectStates,
    move_model_inputs,
)
from sspace.core.models.runtime import LoadedModelRuntime

from .schema import MultiViewSample
from .scoring import (
    TARGET_ROLE,
    render_multiview_order_prompt,
    score_layerwise_multiview_order,
)


def _new_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _image(path: Path, expected_sha256: str) -> Image.Image:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Multi-view image checksum changed: {path}")
    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def _layerwise_ordered_state(
    runtime: LoadedModelRuntime,
    extractor: LayerwiseNamedObjectStates,
    images: Sequence[Image.Image],
    object_name: str,
) -> tuple[torch.Tensor, dict[str, int], dict[str, str], tuple[str, str]]:
    prompt = render_multiview_order_prompt(object_name)
    prepared = runtime.prepare_multi_image_objects(runtime.processor, images, prompt)
    states = extractor.extract(
        move_model_inputs(prepared.inputs, runtime.model),
        prepared.positions,
        (TARGET_ROLE,),
    )
    expected = (1, len(runtime.blocks), 1, runtime.spec.hidden_size)
    if tuple(states.shape) != expected:
        raise ValueError(
            f"Layerwise multi-view object states {tuple(states.shape)} != {expected}"
        )
    return (
        states[0, :, 0],
        dict(prepared.positions[0]),
        dict(prepared.token_pieces[0]),
        prompt.text_segments,
    )


def run_layerwise_multiview_order_evaluation(
    runtime: LoadedModelRuntime,
    samples: Sequence[MultiViewSample],
    dataset_fingerprint: str,
    axes: np.ndarray,
    layer_ids: Sequence[int],
    config: HStarMultiViewConfig,
    output_dir: Path,
    run_config: dict[str, Any],
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Run resumable all-layer H* AB/BA forwards and score image binding.

    One AB and one BA forward capture every post-block object state. Layer
    ``l`` is scored only with ``axes[l]`` and ``h_BA[l] - h_AB[l]``. The
    runner stores float32 states as ``[N,2,L,D]`` and never mixes layers.
    """
    if runtime.spec.key != config.model_adapter:
        raise ValueError("Multi-view runtime differs from the config")
    expected_count = (
        config.expected_sample_count if sample_limit is None else sample_limit
    )
    if len(samples) != expected_count:
        raise ValueError("Multi-view sample count differs from the config")
    layers = tuple(int(layer) for layer in layer_ids)
    basis = np.asarray(axes, dtype=np.float32)
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or layers[0] < 0
        or layers[-1] >= len(runtime.blocks)
    ):
        raise ValueError(
            "Multi-view artifact layers must be increasing supported post-block IDs"
        )
    if basis.shape != (len(layers), 3, runtime.spec.hidden_size):
        raise ValueError("Layerwise multi-view axes have an invalid shape")
    if not np.isfinite(basis).all():
        raise FloatingPointError("Layerwise multi-view axes are non-finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "layerwise_records.jsonl"
    states_path = output_dir / "all_layer_object_states.npy"
    checkpoint_path = output_dir / "layerwise_checkpoint.json"
    fingerprint = canonical_fingerprint(run_config)
    shape = (len(samples), 2, len(layers), runtime.spec.hidden_size)
    if checkpoint_path.exists():
        checkpoint_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_state["fingerprint"] != fingerprint:
            raise ValueError("Layerwise multi-view checkpoint fingerprint mismatch")
        states = np.load(states_path, mmap_mode="r+")
        if states.shape != shape or states.dtype != np.float32:
            raise ValueError("Layerwise multi-view state checkpoint changed")
        next_index = int(checkpoint_state["next_sample_index"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint_state["records_offset"]))
    else:
        if records_path.exists() or states_path.exists():
            raise FileExistsError("Layerwise outputs exist without a checkpoint")
        states = _new_memmap(states_path, shape)
        records_path.touch(exist_ok=False)
        next_index = 0
    committed_offset = records_path.stat().st_size

    def checkpoint(stream: Any) -> None:
        states.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample_index": next_index,
                "records_offset": committed_offset,
                "states_shape": list(shape),
                "layer_ids": list(layers),
            },
        )

    try:
        with records_path.open("a", encoding="utf-8") as stream:
            with LayerwiseNamedObjectStates(runtime.model, runtime.blocks) as extractor:
                while next_index < len(samples):
                    sample = samples[next_index]
                    image_a = _image(sample.image_a_path, sample.image_a_sha256)
                    image_b = _image(sample.image_b_path, sample.image_b_sha256)
                    order_evidence = []
                    ordered_states = []
                    for name, images in (
                        ("ab", (image_a, image_b)),
                        ("ba", (image_b, image_a)),
                    ):
                        state, positions, pieces, text_segments = (
                            _layerwise_ordered_state(
                                runtime, extractor, images, sample.object_name
                            )
                        )
                        ordered_states.append(state)
                        order_evidence.append(
                            {
                                "order": name,
                                "image_sha256": (
                                    [sample.image_a_sha256, sample.image_b_sha256]
                                    if name == "ab"
                                    else [sample.image_b_sha256, sample.image_a_sha256]
                                ),
                                "text_segments": list(text_segments),
                                "positions": positions,
                                "token_pieces": pieces,
                            }
                        )
                    state_tensor = torch.stack(ordered_states)[:, list(layers), :]
                    states[next_index] = state_tensor.numpy()
                    delta = state_tensor[1] - state_tensor[0]
                    all_scores = torch.einsum(
                        "ld,lad->la", delta, torch.from_numpy(basis)
                    )
                    target_scores = all_scores[:, sample.axis_index]
                    truth_sign = int(np.sign(sample.gt_delta_hvd[sample.axis_index]))
                    predicted_signs = torch.sign(target_scores).to(torch.int8)
                    record = {
                        "evaluation_id": config.evaluation_id,
                        "sample_index": next_index,
                        "sample_id": sample.sample_id,
                        "scene": sample.scene,
                        "object_id": sample.object_id,
                        "object_name": sample.object_name,
                        "axis": sample.axis,
                        "target_label": sample.endpoint_label,
                        "ground_truth_sign": truth_sign,
                        "layer_ids": list(layers),
                        "state_index": next_index,
                        "order_evidence": order_evidence,
                        "target_projection_delta_by_layer": target_scores.tolist(),
                        "prediction_sign_by_layer": predicted_signs.tolist(),
                        "correct_by_layer": (predicted_signs == truth_sign).tolist(),
                    }
                    stream.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    next_index += 1
                    committed_offset = stream.tell()
                    if next_index % config.checkpoint_every == 0:
                        checkpoint(stream)
                        print(
                            f"layerwise multi-view {next_index}/{len(samples)}",
                            flush=True,
                        )
            checkpoint(stream)
    except BaseException:
        with records_path.open("r+b") as stream:
            stream.truncate(committed_offset)
        with records_path.open("a", encoding="utf-8") as stream:
            checkpoint(stream)
        raise

    target_scores, metrics = score_layerwise_multiview_order(
        samples, states, basis, layers
    )
    np.save(output_dir / "target_projection_deltas.npy", target_scores)
    atomic_json(
        output_dir / "layerwise_metrics.json",
        {"layer_ids": list(layers), "metrics": list(metrics)},
    )
    with (output_dir / "layerwise_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        endpoint_labels = tuple(metrics[0]["by_label"])
        grounding_modes = tuple(sorted(metrics[0].get("by_grounding_mode", {})))
        fields = [
            "layer_id",
            "overall_accuracy",
            "horizontal_accuracy",
            "vertical_accuracy",
            "distance_accuracy",
            *(f"{label}_accuracy" for label in endpoint_labels),
            *(f"{mode}_accuracy" for mode in grounding_modes),
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = {
                "layer_id": metric["layer_id"],
                "overall_accuracy": metric["overall"]["accuracy"],
                "horizontal_accuracy": metric["by_axis"]["horizontal"]["accuracy"],
                "vertical_accuracy": metric["by_axis"]["vertical"]["accuracy"],
                "distance_accuracy": metric["by_axis"]["distance"]["accuracy"],
                **{
                    f"{label}_accuracy": metric["by_label"][label]["accuracy"]
                    for label in endpoint_labels
                },
                **{
                    f"{mode}_accuracy": metric["by_grounding_mode"][mode]["accuracy"]
                    for mode in grounding_modes
                },
            }
            writer.writerow(row)
    best = max(
        metrics,
        key=lambda metric: (
            metric["overall"]["accuracy"],
            min(value["accuracy"] for value in metric["by_axis"].values()),
            -metric["layer_id"],
        ),
    )
    summary = {
        "evaluation_id": config.evaluation_id,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_count": len(samples),
        "layer_ids": list(layers),
        "state_shape": list(shape),
        "diagnostic_best_layer": best["layer_id"],
        "diagnostic_best_metrics": best,
        "prompt_protocol": config.prompt_protocol,
        "projection_protocol": config.projection_protocol,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary

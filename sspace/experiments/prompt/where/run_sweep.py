"""Run one model's all-layer ``Where is object?`` prompt control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from numpy.lib.format import open_memmap

from sspace.determinism import configure_deterministic_cuda
from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.experiments.common.pairwise.config import (
    PairwiseEvaluationConfig,
    load_evaluation_config,
)
from sspace.experiments.common.pairwise.decision import classify_pairwise_margin
from sspace.experiments.common.pairwise.layerwise import (
    diagnostic_best_layer,
    score_layerwise_margins,
)
from sspace.experiments.common.pairwise.registry import BenchmarkRegistry
from sspace.experiments.common.pairwise.schema import (
    PairwiseEvaluationSample,
    open_pairwise_image,
    pairwise_dataset_fingerprint,
)
from sspace.experiments.identity import (
    bind_launch_identity,
    experiment_launch_fingerprint,
)
from sspace.experiments.prompt.config import (
    completed_prompt_output,
    load_prompt_sweep_config,
)
from sspace.experiments.prompt.sampling import balanced_prompt_subset
from sspace.experiments.prompt.where.scoring import (
    SINGLE_OBJECT_PROMPT_PROTOCOL,
    TARGET_ROLE,
    project_separate_object_states,
    render_single_object_where_prompt,
)
from sspace.run_records import (
    RunRecorder,
    atomic_text,
    atomic_json,
    canonical_fingerprint,
    file_sha256,
)
from sspace.core.extraction.execution import (
    LayerwiseNamedObjectStates,
    move_model_inputs,
)
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    LoadedModelRuntime,
    configure_image_max_pixels,
    load_model_runtime,
)
from sspace.core.prompts.templates import AXIS_ORDER


def _load_configs(
    paths: Sequence[Path], project_root: Path
) -> tuple[PairwiseEvaluationConfig, ...]:
    """Load fixed pairwise configs and require one shared model and artifact."""
    configs = tuple(load_evaluation_config(path, project_root) for path in paths)
    identity = {
        (
            config.model_adapter,
            config.model_path,
            config.artifact_dir,
            config.device,
        )
        for config in configs
    }
    if len(identity) != 1:
        raise ValueError("One sweep must share one model, artifact, and device")
    if len({config.benchmark for config in configs}) != len(configs):
        raise ValueError("Single-object sweep benchmarks must be unique")
    return configs


def _resolved_config(
    config: PairwiseEvaluationConfig,
    source_config: Path,
    artifact: SSpaceArtifact,
    output_dir: Path,
) -> dict[str, Any]:
    """Bind one control run to immutable model, artifact, data, and prompt."""
    return {
        "analysis": "single_object_where_layer_sweep_v1",
        "prompt_protocol": SINGLE_OBJECT_PROMPT_PROTOCOL,
        "projection_protocol": "layerwise_separate_object_delta_v1",
        "decision_protocol": "margin_sign_v1",
        "source_config": str(source_config.resolve()),
        "source_config_sha256": file_sha256(source_config),
        "artifact_dir": str(config.artifact_dir),
        "artifact_manifest_sha256": file_sha256(config.artifact_dir / "manifest.json"),
        "artifact_id": artifact.manifest.artifact_id,
        "layer_ids": list(artifact.manifest.model.layer_ids),
        "selected_layer": artifact.manifest.model.selected_layer,
        "model_adapter": config.model_adapter,
        "model_path": str(config.model_path),
        "benchmark": config.benchmark,
        "benchmark_source": str(config.benchmark_source),
        "output_dir": str(output_dir),
        "image_max_pixels": config.image_max_pixels,
        "device": config.device,
        "checkpoint_every": config.checkpoint_every,
        "seed": config.seed,
        "condition_order": ["query_where", "reference_where"],
    }


def _new_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        raise FileExistsError(path)
    values = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    values[:] = np.nan
    values.flush()
    return values


def _extract_object_states(
    runtime: LoadedModelRuntime,
    extractor: LayerwiseNamedObjectStates,
    image: Image.Image,
    object_name: str,
    layer_ids: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one independent absolute question and return states ``[L,D]``."""
    prompt = render_single_object_where_prompt(object_name)
    prepared = runtime.prepare_objects(runtime.processor, image, [prompt])
    states = extractor.extract(
        move_model_inputs(prepared.inputs, runtime.model),
        prepared.positions,
        (TARGET_ROLE,),
    )
    expected = (1, len(runtime.blocks), 1, runtime.spec.hidden_size)
    if tuple(states.shape) != expected:
        raise ValueError(f"Single-object states {tuple(states.shape)} != {expected}")
    selected = states[0, list(layer_ids), 0].numpy()
    return selected, {
        "object_name": object_name,
        "prompt": prompt.text,
        "positions": dict(prepared.positions[0]),
        "token_pieces": dict(prepared.token_pieces[0]),
    }


def _extract_all_layers(
    runtime: LoadedModelRuntime,
    samples: Sequence[PairwiseEvaluationSample],
    axes: np.ndarray,
    layer_ids: Sequence[int],
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run two independent prompts per row and save all states and coordinates."""
    layers = tuple(int(value) for value in layer_ids)
    basis = np.asarray(axes, dtype=np.float32)
    if basis.shape != (len(layers), 3, runtime.spec.hidden_size):
        raise ValueError("Single-object axes must have shape [L,3,D]")
    if not np.isfinite(basis).all():
        raise FloatingPointError("Single-object axes must be finite")
    for sample in samples:
        sample.validate()

    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "object_states.npy"
    coordinates_path = output_dir / "object_coordinates.npy"
    margins_path = output_dir / "all_axis_margins.npy"
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    state_shape = (len(samples), 2, len(layers), runtime.spec.hidden_size)
    coordinate_shape = (len(samples), 2, len(layers), len(AXIS_ORDER))
    margin_shape = (len(samples), len(layers), len(AXIS_ORDER))
    fingerprint = canonical_fingerprint(run_config)

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Single-object checkpoint fingerprint mismatch")
        states = np.load(states_path, mmap_mode="r+")
        coordinates = np.load(coordinates_path, mmap_mode="r+")
        margins = np.load(margins_path, mmap_mode="r+")
        if (
            states.shape != state_shape
            or coordinates.shape != coordinate_shape
            or margins.shape != margin_shape
            or any(
                value.dtype != np.float32 for value in (states, coordinates, margins)
            )
        ):
            raise ValueError("Single-object checkpoint arrays are incompatible")
        next_sample = int(checkpoint["next_sample"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
    else:
        if any(
            path.exists()
            for path in (states_path, coordinates_path, margins_path, records_path)
        ):
            raise FileExistsError("Single-object outputs exist without a checkpoint")
        states = _new_memmap(states_path, state_shape)
        coordinates = _new_memmap(coordinates_path, coordinate_shape)
        margins = _new_memmap(margins_path, margin_shape)
        records_path.touch(exist_ok=False)
        next_sample = 0

    def checkpoint(stream: Any) -> None:
        for values in (states, coordinates, margins):
            values.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample": next_sample,
                "records_offset": stream.tell(),
                "state_shape": list(state_shape),
                "coordinate_shape": list(coordinate_shape),
                "margin_shape": list(margin_shape),
                "condition_order": ["query_where", "reference_where"],
            },
        )

    with records_path.open("a", encoding="utf-8") as records:
        checkpoint(records)
        with LayerwiseNamedObjectStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                image = open_pairwise_image(sample)
                query_states, query_evidence = _extract_object_states(
                    runtime, extractor, image, sample.query, layers
                )
                reference_states, reference_evidence = _extract_object_states(
                    runtime, extractor, image, sample.reference, layers
                )
                object_states = np.stack((query_states, reference_states))
                object_coordinates = np.einsum("old,lad->ola", object_states, basis)
                all_axis_margins = project_separate_object_states(
                    query_states, reference_states, basis
                ).astype(np.float32)
                states[next_sample] = object_states
                coordinates[next_sample] = object_coordinates
                margins[next_sample] = all_axis_margins
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "sample_id": sample.sample_id,
                            "group": sample.group,
                            "gold_endpoint": sample.gold_endpoint,
                            "image_sha256": sample.image_sha256,
                            "query_evidence": query_evidence,
                            "reference_evidence": reference_evidence,
                            "state_index": next_sample,
                            "coordinate_index": next_sample,
                            "all_axis_margin_index": next_sample,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                if next_sample % checkpoint_every == 0:
                    checkpoint(records)
                    print(
                        f"{run_config['benchmark']} {next_sample}/{len(samples)}",
                        flush=True,
                    )
        checkpoint(records)
    if not all(np.isfinite(value).all() for value in (states, coordinates, margins)):
        raise FloatingPointError("Completed single-object outputs are non-finite")
    return np.asarray(states), np.asarray(coordinates), np.asarray(margins)


def _metric_rows(
    samples: Sequence[PairwiseEvaluationSample],
    all_axis_margins: np.ndarray,
    layer_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return overall, H/V/D, and six-endpoint accuracy at every layer."""
    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    group_margins = np.asarray(
        [
            all_axis_margins[row, :, axis_index[sample.group]]
            for row, sample in enumerate(samples)
        ]
    )
    metrics = score_layerwise_margins(samples, group_margins, layer_ids)
    rows = []
    for layer_position, metric in enumerate(metrics):
        predictions = []
        for row, sample in enumerate(samples):
            predictions.append(
                classify_pairwise_margin(
                    sample.group, group_margins[row, layer_position]
                )
            )
        endpoint_accuracy = {}
        for endpoint in ("left", "right", "above", "below", "far", "close"):
            selected = [
                index
                for index, sample in enumerate(samples)
                if sample.gold_endpoint == endpoint
            ]
            if not selected:
                raise ValueError(f"Benchmark has no {endpoint} samples")
            endpoint_accuracy[endpoint] = float(
                np.mean([predictions[index] == endpoint for index in selected])
            )
        row = metric.to_dict()
        row["endpoint_accuracy"] = endpoint_accuracy
        rows.append(row)
    return rows


def _write_metrics(
    output_dir: Path,
    rows: list[dict[str, Any]],
    selected_layer: int,
) -> dict[str, Any]:
    """Write complete metrics and return fixed-layer plus oracle summary."""
    with atomic_text(output_dir / "layerwise_metrics.csv", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layer",
                "overall",
                "horizontal",
                "vertical",
                "distance",
                "worst_axis",
                "left",
                "right",
                "above",
                "below",
                "far",
                "close",
            ]
        )
        for row in rows:
            endpoints = row["endpoint_accuracy"]
            writer.writerow(
                [
                    row["layer_id"],
                    row["overall_accuracy"],
                    *row["axis_accuracy"],
                    row["worst_axis_accuracy"],
                    *(
                        endpoints[name]
                        for name in ("left", "right", "above", "below", "far", "close")
                    ),
                ]
            )
    atomic_json(output_dir / "layerwise_metrics.json", {"per_layer": rows})
    best = max(
        rows,
        key=lambda row: (
            row["overall_accuracy"],
            row["worst_axis_accuracy"],
            -row["layer_id"],
        ),
    )
    fixed = next(row for row in rows if row["layer_id"] == selected_layer)
    summary = {
        "selected_layer": selected_layer,
        "selected_layer_metrics": fixed,
        "diagnostic_best_layer": best["layer_id"],
        "diagnostic_best_metrics": best,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Load one configured model and run its absolute-prompt controls."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    launch = load_prompt_sweep_config(args.config, project_root, "prompt_where")
    configs = _load_configs(launch.source_configs, project_root)
    if configs[0].model_adapter != launch.model_adapter:
        raise ValueError("Prompt config model differs from source configs")
    artifact = SSpaceArtifact.load(configs[0].artifact_dir)
    spec = RUNTIME_SPECS[configs[0].model_adapter]
    artifact.validate_model(
        ModelSpec(
            model_id=spec.model_id,
            revision=spec.revision,
            architecture=spec.architecture,
            tokenizer_id=spec.model_id,
            tokenizer_revision=spec.revision,
            hidden_size=spec.hidden_size,
            layer_ids=tuple(range(spec.layer_count)),
            selected_layer=artifact.manifest.model.selected_layer,
        )
    )

    random.seed(configs[0].seed)
    np.random.seed(configs[0].seed)
    torch.manual_seed(configs[0].seed)
    torch.cuda.manual_seed_all(configs[0].seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = load_model_runtime(
        configs[0].model_adapter, configs[0].model_path, configs[0].device
    )

    for source_path, config in zip(launch.source_configs, configs, strict=True):
        configure_image_max_pixels(runtime, config.image_max_pixels)
        samples = BenchmarkRegistry.load(config.benchmark, config.benchmark_source)
        if pairwise_dataset_fingerprint(samples) != config.expected_dataset_fingerprint:
            raise ValueError("Prompt source dataset fingerprint changed")
        samples = balanced_prompt_subset(samples, args.limit)
        output_dir = launch.output_dir.resolve() / config.benchmark
        launch_fingerprint = experiment_launch_fingerprint(
            args.config, project_root, args.limit
        )
        if completed_prompt_output(output_dir, launch_fingerprint):
            print(f"skip completed prompt output: {output_dir}", flush=True)
            continue
        resolved = _resolved_config(config, source_path, artifact, output_dir)
        resolved["sample_limit"] = args.limit
        bind_launch_identity(resolved, args.config, project_root, args.limit)
        recorder = RunRecorder(output_dir, resolved, project_root)
        stage = "single_object_where_layer_sweep"
        try:
            recorder.event(stage, "started", sample_count=len(samples))
            states, coordinates, margins = _extract_all_layers(
                runtime,
                samples,
                artifact.axes_tensor,
                artifact.manifest.model.layer_ids,
                output_dir,
                resolved,
                config.checkpoint_every,
            )
            rows = _metric_rows(samples, margins, artifact.manifest.model.layer_ids)
            summary = _write_metrics(
                output_dir, rows, artifact.manifest.model.selected_layer
            )
            best = diagnostic_best_layer(
                score_layerwise_margins(
                    samples,
                    np.asarray(
                        [
                            margins[row, :, AXIS_ORDER.index(sample.group)]
                            for row, sample in enumerate(samples)
                        ]
                    ),
                    artifact.manifest.model.layer_ids,
                )
            )
            if best != summary["diagnostic_best_layer"]:
                raise ValueError("Single-object best-layer calculations differ")
            recorder.event(
                stage,
                "complete",
                selected_layer=artifact.manifest.model.selected_layer,
                selected_accuracy=summary["selected_layer_metrics"]["overall_accuracy"],
                diagnostic_best_layer=best,
                diagnostic_best_accuracy=summary["diagnostic_best_metrics"][
                    "overall_accuracy"
                ],
            )
            recorder.complete(
                sample_count=len(samples),
                state_shape=list(states.shape),
                coordinate_shape=list(coordinates.shape),
                margin_shape=list(margins.shape),
                summary=summary,
            )
            print(
                f"complete {config.benchmark}: fixed L{artifact.manifest.model.selected_layer} "
                f"{summary['selected_layer_metrics']['overall_accuracy']:.4f}; "
                f"best L{best} {summary['diagnostic_best_metrics']['overall_accuracy']:.4f}",
                flush=True,
            )
        except BaseException as error:
            recorder.fail(error, stage)
            raise


if __name__ == "__main__":
    main()
